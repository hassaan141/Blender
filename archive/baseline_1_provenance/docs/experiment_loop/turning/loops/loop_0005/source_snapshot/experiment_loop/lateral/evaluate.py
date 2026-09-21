"""Long-horizon deterministic evaluator for LateralWalkEnv (Bingo-Lateral-*),
adapted from rl/bingo_rl/bingo_rl/natural_walk/evaluate.py (read, not modified --
that file lives under the protected rl/bingo_rl/bingo_rl/ tree). Same structure,
same metrics schema (so core.py's existing generic metric_gate() works
unchanged), with exactly these differences:

  * measures body-frame LATERAL velocity (root_lin_b[:,1], +Y=left per
    stage2/v4_kinematics.py) into the SAME "vx"-named log slot/metrics fields
    the base evaluator uses for forward velocity -- same trick backward's own
    evaluation already uses (reporting backward speed through the same
    vx_mean/vx_std/vx_max keys). This keeps every existing generic gate in
    core.py's metric_gate() working without modification.
  * drives base._cmd_vy (new attribute on LateralWalkEnv) instead of erroring
    on nonzero lateral command; base._cmd_vx is held at 0 (forward-drift
    suppression, wanted for a pure sidestep).
  * registers Bingo-Lateral-{Train,Play}-v0 via env.py before gym.make.

Reuses rl/bingo_rl/bingo_rl/natural_walk/diagnostics.QualityRecorder unchanged
(read-only import) for the natural_metrics.json/quality_metrics.json/
paw_trajectories.npz artifacts core.py's Runner.arm() requires -- that class is
gait-direction-agnostic (per-joint/contact/bobbing diagnostics only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

BLENDER_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = BLENDER_ROOT / "docs/experiment_loop/lateral/evaluation"

parser = argparse.ArgumentParser(description="Bingo lateral (sidestep) loop evaluator.")
parser.add_argument("--task", type=str, default="Bingo-Lateral-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_vy", type=float, required=True,
                     help="held lateral-speed command, m/s (+ = left, - = right).")
parser.add_argument("--duration_s", type=float, default=60.0)
parser.add_argument("--settle_seconds", type=float, default=1.0)
parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
parser.add_argument("--pd_only", action="store_true")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--diagnostics", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if not args_cli.no_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse  # noqa: E402

sys.path.insert(0, str(BLENDER_ROOT / "rl" / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env as lateral_env  # noqa: F401,E402  (registers Bingo-Lateral-{Train,Play}-v0)

EFFORT_LIMIT = 3.0  # v4 leg actuator ceiling; same constant natural_walk/evaluate.py uses.


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    settle_steps_est = int(round(args_cli.settle_seconds / (env_cfg.sim.dt * env_cfg.decimation)))
    duration_steps_est = int(round(args_cli.duration_s / (env_cfg.sim.dt * env_cfg.decimation)))
    env_cfg.episode_length_s = args_cli.settle_seconds + args_cli.duration_s + 5.0

    video = not args_cli.no_video
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if video else None)
    total_steps = settle_steps_est + duration_steps_est
    if video:
        import time
        env.reset()
        for _ in range(10):
            env.render()
            time.sleep(0.2)
        video_folder = out_dir / "_video_tmp"
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(video_folder),
            step_trigger=lambda s: s == 0,
            video_length=total_steps + 48, disable_logger=True)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    from skrl.utils.runner.torch import Runner
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wenv, agent_cfg)
    checkpoint_path = str(Path(args_cli.checkpoint).resolve())
    runner.agent.load(checkpoint_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    print(f"[[ policy loaded: {checkpoint_path}", flush=True)

    joint_names = list(base.ctrl_dof_names)
    step_dt = base.step_dt
    settle_steps = int(round(args_cli.settle_seconds / step_dt))
    duration_steps = int(round(args_cli.duration_s / step_dt))
    total_steps = settle_steps + duration_steps

    log = {k: [] for k in ("vx", "roll_deg", "pitch_deg", "qerr", "resid", "torque", "qdot", "done")}
    if args_cli.diagnostics:
        from bingo_rl.natural_walk.diagnostics import QualityRecorder
        diagnostic = QualityRecorder(base, joint_names, out_dir)

    obs, _ = wenv.reset()
    with torch.inference_mode():
        for k in range(total_steps):
            base._cmd_vx[:] = 0.0
            base._cmd_vy[:] = args_cli.cmd_vy
            base._cmd_timer[:] = 1e6  # hold: this script drives the command explicitly

            out = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
            action = out[-1].get("mean_actions", out[0])
            if args_cli.pd_only: action = torch.zeros_like(action)
            obs, _, term, tout, _ = wenv.step(action)

            if k < settle_steps:
                if args_cli.diagnostics:
                    diagnostic.warmup_falls += int(term.sum().item())
                continue

            d = base.robot.data
            root_quat = d.body_quat_w[:, base.ref_body_index]
            root_lin_w = d.body_lin_vel_w[:, base.ref_body_index]
            root_lin_b = quat_rotate_inverse(root_quat, root_lin_w)
            vy_b = root_lin_b[:, 1].cpu().numpy()   # LATERAL velocity, +Y=left
            vx_b_forward = root_lin_b[:, 0].cpu().numpy()  # kept for drift reporting only
            roll, pitch, _yaw = euler_xyz_from_quat(root_quat)
            roll_deg = torch.rad2deg(roll).cpu().numpy()
            pitch_deg = torch.rad2deg(pitch).cpu().numpy()

            cur_dof = d.joint_pos[:, base.robot_ctrl_indexes].cpu().numpy()
            ref_dof = base._ff_leg_q.cpu().numpy()
            qerr = np.abs(cur_dof - ref_dof)

            applied_action = getattr(base, "_filtered_action", action)
            resid = (base._res_scale * applied_action).cpu().numpy()
            torque = d.applied_torque[:, base.robot_ctrl_indexes].cpu().numpy()
            qdot = d.joint_vel[:, base.robot_ctrl_indexes].cpu().numpy()

            done = (term | tout).reshape(-1).cpu().numpy() if term is not None \
                else np.zeros(base.num_envs, bool)

            # "vx" slot carries the LATERAL speed -- see module docstring; forward
            # drift is tracked separately below and reported, not gated generically.
            log["vx"].append(vy_b); log["roll_deg"].append(roll_deg); log["pitch_deg"].append(pitch_deg)
            log["qerr"].append(qerr); log["resid"].append(resid); log["torque"].append(torque)
            log["qdot"].append(qdot); log["done"].append(done)
            if not hasattr(main, "_forward_drift"):
                main._forward_drift = []
            main._forward_drift.append(vx_b_forward)
            if args_cli.diagnostics:
                diagnostic.capture(base, action)

    D = {k: np.asarray(v) for k, v in log.items()}
    forward_drift = np.asarray(main._forward_drift) if hasattr(main, "_forward_drift") else np.zeros_like(D["vx"])
    metrics = _summarize(D, forward_drift, joint_names, step_dt, args_cli, checkpoint_path)
    if args_cli.diagnostics:
        diagnostic.finish(D, metrics, step_dt)
    env.close()

    if video:
        _finalize_video(video_folder, out_dir)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    report = _format(metrics)
    (out_dir / "report.txt").write_text(report + "\n")
    print(report, flush=True)
    print(f"[[ wrote {metrics_path}", flush=True)


def _summarize(D, forward_drift, joint_names, step_dt, args_cli, checkpoint_path):
    vy = D["vx"]                # (T,E) -- actually lateral speed, see module docstring
    roll = D["roll_deg"]
    pitch = D["pitch_deg"]
    qerr = D["qerr"]
    resid = D["resid"]
    torque = D["torque"]
    qdot = D["qdot"]
    done = D["done"]
    n_steps, n_envs = vy.shape

    ever_fell = np.cumsum(done.astype(int), axis=0) > 0
    alive = ~ever_fell
    fell_ever = ever_fell[-1] if n_steps else np.zeros(n_envs, bool)
    fall_step = np.where(fell_ever, alive.sum(axis=0), n_steps)
    survival_per_env = fall_step / float(n_steps)

    def stat(a, fn, mask=alive):
        return float(fn(a[mask])) if mask.any() else float("nan")

    d_resid = np.diff(resid, axis=0)
    d_qdot = np.diff(qdot, axis=0) / step_dt
    mask_d = alive[1:]

    torque_sat_per_joint = {}
    qerr_mean_per_joint = {}
    for j, name in enumerate(joint_names):
        m = alive
        torque_sat_per_joint[name] = float(
            100.0 * (np.abs(torque[:, :, j])[m] >= 0.99 * EFFORT_LIMIT).mean()
        ) if m.any() else float("nan")
        qerr_mean_per_joint[name] = stat(qerr[:, :, j], np.mean)

    metrics = {
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "num_envs": n_envs,
        "command": {
            "vx": 0.0, "vy": args_cli.cmd_vy, "yaw": 0.0,
            "note": "vx is the metric-schema-generic field name; this campaign's tracked "
                    "axis is lateral (vy). command.vy is the true held command.",
        },
        "duration_s": args_cli.duration_s,
        "settle_seconds": args_cli.settle_seconds,
        "steps_recorded": n_steps,
        "step_dt": step_dt,
        "effort_limit_Nm": EFFORT_LIMIT,

        "survival": float(survival_per_env.mean()),
        "any_fell": bool(fell_ever.any()),
        "fall_time_s": [float(fs * step_dt) if fell_ever[e] else None
                         for e, fs in enumerate(fall_step)],

        # generic schema fields (read by core.py's metric_gate) -- populated with
        # the LATERAL speed, matching the "vx" log slot; see module docstring.
        "vx_mean": stat(vy, np.mean),
        "vx_std": stat(vy, np.std),
        "vx_max": stat(vy, np.max),
        "vx_abs_max": stat(np.abs(vy), np.max),

        # explicit, unambiguous lateral-specific fields (for reports/gates that
        # want the real semantics rather than the generic schema names).
        "vy_mean": stat(vy, np.mean),
        "vy_std": stat(vy, np.std),
        "vy_abs_max": stat(np.abs(vy), np.max),
        "forward_drift_mean_m_s": stat(forward_drift, np.mean),
        "forward_drift_abs_max_m_s": stat(np.abs(forward_drift), np.max),

        "base_roll_rms_deg": stat(roll, lambda a: np.sqrt((a ** 2).mean())),
        "base_pitch_rms_deg": stat(pitch, lambda a: np.sqrt((a ** 2).mean())),

        "joint_accel_rms": float(np.sqrt((d_qdot[mask_d] ** 2).mean())) if mask_d.any() else float("nan"),
        "residual_action_rate_rms": float(np.sqrt((d_resid[mask_d] ** 2).mean())) if mask_d.any() else float("nan"),

        "torque_saturation_pct_per_joint": torque_sat_per_joint,

        "reference_tracking_error_rad": {
            "mean": stat(qerr, np.mean),
            "rms": stat(qerr, lambda a: np.sqrt((a ** 2).mean())),
            "max": stat(qerr, np.max),
            "mean_per_joint": qerr_mean_per_joint,
        },
    }
    return metrics


def _finalize_video(video_folder: Path, out_dir: Path):
    mp4s = sorted(video_folder.glob("*.mp4"))
    if not mp4s:
        print(f"[[ WARNING: no .mp4 found in {video_folder}, video not written", flush=True)
        return
    dest = out_dir / "eval.mp4"
    shutil.move(str(mp4s[0]), str(dest))
    shutil.rmtree(video_folder, ignore_errors=True)
    print(f"[[ wrote {dest}", flush=True)


def _format(m):
    out = ["", "=" * 100, "BINGO LATERAL (SIDESTEP) LOOP EVALUATION", "=" * 100, "",
           f"task={m['task']}  checkpoint={m['checkpoint']}",
           f"command: vy={m['command']['vy']:.3f} m/s ({m['command']['note']})",
           f"duration={m['duration_s']:.1f}s  settle={m['settle_seconds']:.1f}s  "
           f"steps_recorded={m['steps_recorded']}  num_envs={m['num_envs']}",
           "-" * 100,
           f"survival: {100*m['survival']:.1f}%   any_fell={m['any_fell']}  fall_time_s={m['fall_time_s']}",
           f"vy (lateral): mean={m['vy_mean']:.3f}  std={m['vy_std']:.3f}  abs_max={m['vy_abs_max']:.3f}  (cmd={m['command']['vy']:.3f})",
           f"forward drift: mean={m['forward_drift_mean_m_s']:.3f}  abs_max={m['forward_drift_abs_max_m_s']:.3f} m/s (want near 0)",
           f"base roll RMS: {m['base_roll_rms_deg']:.3f} deg   base pitch RMS: {m['base_pitch_rms_deg']:.3f} deg",
           f"joint accel RMS: {m['joint_accel_rms']:.3f} rad/s^2",
           f"residual action-rate RMS: {m['residual_action_rate_rms']:.4f} rad/step",
           f"reference tracking error (rad): mean={m['reference_tracking_error_rad']['mean']:.4f}  "
           f"rms={m['reference_tracking_error_rad']['rms']:.4f}  max={m['reference_tracking_error_rad']['max']:.4f}",
           "", "torque saturation % per joint:"]
    for name, pct in m["torque_saturation_pct_per_joint"].items():
        out.append(f"  {name:10s} {pct:6.1f}%")
    out.append("=" * 100)
    return "\n".join(out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        Path("/tmp/eval_lateral_loop_crash.txt").write_text(tb)
        print(tb, flush=True)
        raise
    simulation_app.close()
