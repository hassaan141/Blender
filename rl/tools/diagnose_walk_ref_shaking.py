"""Per-joint shaking/instability diagnostic for the reference-guided walk policy
(Bingo-WalkRef-v4-*). Extends the S6 torque-saturation diagnostic
(rl/tools/diagnose_torque_saturation.py) with the extra signals this env has that
S6 doesn't: q_ref(phase), the RL residual, and the loop-seam wrap event.

Logs per-step, per-leg-joint: q_ref, residual (= res_scale*action), q_target
(= q_ref + residual), actual q, qdot, applied torque, saturation, plus phase and
whether this step crossed the cyclic reference's loop seam (phase wrapped).
Drives a HELD NOMINAL-SPEED command throughout (blend weight == 1, i.e. q_ref ==
the raw cyclic reference, no stand-blend interference) since that's where the
policy is reported to fall/shake.

Reports, per joint: residual mean/max, action-rate RMS (|delta residual| step to
step), joint-acceleration RMS (|delta qdot|/dt), torque-sat %, and a loop-seam vs
non-seam breakdown of all of the above -- to separate "the reference itself is
rough at the wrap" from "the RL residual is oscillating independent of the
reference."

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p ~/Blender/rl/tools/diagnose_walk_ref_shaking.py \
        --checkpoint <run>/agent_XXXX.pt --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Bingo-WalkRef-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--settle_seconds", type=float, default=0.5)
parser.add_argument("--cmd_frac", type=float, default=1.0, help="fraction of NOMINAL_VX to hold")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import NOMINAL_VX  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.random_start_frame = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    from skrl.utils.runner.torch import Runner
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wenv, agent_cfg)
    runner.agent.load(str(Path(args_cli.checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    print(f"[[ loaded {args_cli.checkpoint}", flush=True)

    joint_names = list(base.ctrl_dof_names)
    n_leg = len(joint_names)
    effort_limit = 3.0
    dt = base.step_dt
    cmd_vx = args_cli.cmd_frac * float(NOMINAL_VX)
    steps = int(round(args_cli.seconds / dt))
    settle = int(round(args_cli.settle_seconds / dt))

    log = {k: [] for k in ("phase", "q_ref", "residual", "q_target", "q_act", "qdot",
                           "torque", "sat", "wrapped", "contact")}

    obs, _ = wenv.reset()
    prev_phase = None
    with torch.inference_mode():
        for k in range(steps):
            base._cmd_vx[:] = cmd_vx
            base._cmd_timer[:] = 1e6
            out = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
            action = out[-1].get("mean_actions", out[0])
            obs, _, term, tout, _ = wenv.step(action)

            if k < settle:
                prev_phase = float(base._clip_time[0] / base.motion_duration)
                continue

            phase = float(base._clip_time[0] / base.motion_duration)
            wrapped = prev_phase is not None and phase < prev_phase - 0.5
            prev_phase = phase

            q_ref = base._ff_leg_q[0].cpu().numpy().copy()
            # Use the ACTUALLY-APPLIED action if the env filters it (EMA residual
            # smoothing) -- otherwise this would silently report the raw pre-filter
            # signal, which understates what's really driving the robot.
            applied_action = getattr(base, "_filtered_action", action)
            resid = (base._res_scale * applied_action)[0].cpu().numpy().copy()
            q_target = q_ref + resid
            q_act = base.robot.data.joint_pos[0, base.robot_ctrl_indexes].cpu().numpy().copy()
            qdot = base.robot.data.joint_vel[0, base.robot_ctrl_indexes].cpu().numpy().copy()
            tq = base.robot.data.applied_torque[0, base.robot_ctrl_indexes].cpu().numpy().copy()
            sat = (np.abs(tq) >= 0.99 * effort_limit)
            tips_w, _ = base._foot_tips_local()
            contact = (tips_w[0, :, 2] < 0.03).cpu().numpy().copy()

            log["phase"].append(phase); log["q_ref"].append(q_ref); log["residual"].append(resid)
            log["q_target"].append(q_target); log["q_act"].append(q_act); log["qdot"].append(qdot)
            log["torque"].append(tq); log["sat"].append(sat); log["wrapped"].append(wrapped)
            log["contact"].append(contact)

            if bool(term[0]) or bool(tout[0]):
                print(f"[[ episode ended at step {k+1} (fell={bool(term[0])}, timeout={bool(tout[0])})",
                      flush=True)
                break

    D = {k: np.asarray(v) for k, v in log.items()}
    n = len(D["phase"])
    d_resid = np.zeros_like(D["residual"]); d_resid[1:] = np.diff(D["residual"], axis=0)
    d_qdot = np.zeros_like(D["qdot"]); d_qdot[1:] = np.diff(D["qdot"], axis=0) / dt  # joint accel
    wrap_idx = D["wrapped"]

    out_lines = []
    out_lines.append("")
    out_lines.append("=" * 128)
    out_lines.append(f"WALK-REF SHAKING DIAGNOSTIC  (cmd_vx={cmd_vx:.3f} m/s, {n} steps, "
                     f"{int(wrap_idx.sum())} loop-seam crossings)")
    out_lines.append("=" * 128)
    out_lines.append(f"{'joint':10s} {'resid_mean':>10s} {'resid_max':>9s} {'act_rate_rms':>12s} "
                     f"{'accel_rms':>10s} {'sat%':>6s} {'sat%@seam':>10s} {'accel_rms@seam':>14s} "
                     f"{'corr(|dresid|,sat)':>19s}")
    out_lines.append("-" * 128)
    for j, name in enumerate(joint_names):
        r = D["residual"][:, j]; dr = d_resid[:, j]; da = d_qdot[:, j]; s = D["sat"][:, j]
        sat_pct = 100.0 * s.mean()
        sat_seam = 100.0 * s[wrap_idx].mean() if wrap_idx.any() else float("nan")
        accel_seam = np.sqrt((da[wrap_idx] ** 2).mean()) if wrap_idx.any() else float("nan")
        corr = float(np.corrcoef(np.abs(dr), s.astype(float))[0, 1]) if dr.std() > 0 and s.std() > 0 else float("nan")
        out_lines.append(f"{name:10s} {r.mean():10.4f} {np.abs(r).max():9.4f} "
                         f"{np.sqrt((dr**2).mean()):12.4f} {np.sqrt((da**2).mean()):10.3f} "
                         f"{sat_pct:5.1f}% {sat_seam:9.1f}% {accel_seam:14.3f} {corr:19.3f}")

    out_lines.append("-" * 128)
    out_lines.append(f"overall accel RMS: non-seam {np.sqrt((d_qdot[~wrap_idx]**2).mean()):.3f}  "
                     f"seam {np.sqrt((d_qdot[wrap_idx]**2).mean()) if wrap_idx.any() else float('nan'):.3f}  "
                     f"(ratio {(np.sqrt((d_qdot[wrap_idx]**2).mean())/np.sqrt((d_qdot[~wrap_idx]**2).mean())) if wrap_idx.any() else float('nan'):.2f}x)")
    out_lines.append(f"overall |resid| mean: {np.abs(D['residual']).mean():.4f}  max: {np.abs(D['residual']).max():.4f}  "
                     f"(authority {float(base._res_scale.max()):.2f})")
    out_lines.append(f"overall action-rate RMS (all joints pooled): {np.sqrt((d_resid**2).mean()):.4f}")
    out_lines.append(f"overall torque-sat %: {100.0*D['sat'].mean():.1f}%")

    # q_ref jump magnitude at the seam vs elsewhere, to isolate reference-caused vs residual-caused
    d_qref = np.zeros_like(D["q_ref"]); d_qref[1:] = np.diff(D["q_ref"], axis=0)
    qref_jump_seam = np.abs(d_qref[wrap_idx]).max() if wrap_idx.any() else float("nan")
    qref_jump_other = np.abs(d_qref[~wrap_idx]).max() if (~wrap_idx).any() else float("nan")
    out_lines.append(f"\nq_ref max per-step jump: at seam {qref_jump_seam:.4f} rad  "
                     f"elsewhere {qref_jump_other:.4f} rad")
    resid_jump_seam = np.abs(d_resid[wrap_idx]).max() if wrap_idx.any() else float("nan")
    resid_jump_other = np.abs(d_resid[~wrap_idx]).max() if (~wrap_idx).any() else float("nan")
    out_lines.append(f"residual max per-step jump: at seam {resid_jump_seam:.4f} rad  "
                     f"elsewhere {resid_jump_other:.4f} rad")

    text = "\n".join(out_lines)
    print(text, flush=True)
    Path("/tmp/diagnose_walk_ref_shaking_report.txt").write_text(text)
    np.savez("/tmp/diagnose_walk_ref_shaking_data.npz", joint_names=np.array(joint_names), **D)


if __name__ == "__main__":
    main()
    simulation_app.close()
