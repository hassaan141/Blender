"""Deterministic evaluation battery for Bingo Locomotion 2 (dog-style AMP prior).

Adapts rl/tools/eval_velocity.py's methodology (SegmentRecorder pattern, held-command
segments, pass/fail gates) to this env's stack (skrl AMP DirectRLEnv, not rsl_rl
manager-based) and to the first controller's constrained scope (forward-only,
0.00-0.30 m/s, no turning/lateral -- see bingo_dog_amp_env_cfg.py).

Battery: stand + vx in {0.10, 0.20, 0.25, 0.30}, each held `--seconds_per_segment`
(default 5s, first `--settle_seconds` ignored as transient), plus one sustained
60s hold at vx=0.25 appended at the end.

Reports, per segment: survival %, vx mean/std/max, vx tracking RMSE, roll/pitch
RMS (deg), base-height std/ptp, joint-accel RMS, action-rate RMS, torque
saturation %, foot slip, left/right gait symmetry (front and rear pairs), task
reward, and AMP discriminator diagnostics (mean discriminator logit on policy
transitions THIS segment vs. on a held-out sample of expert transitions, plus
policy-vs-expert separation = |mean_expert - mean_policy|).

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p /pub0/muhammadf/Blender/rl/bingo_rl/scripts/eval_locomotion2_amp.py \
        --checkpoint <run>/checkpoints/agent_XXXX.pt --headless
    # add --video to render standing / slow-walk / 0.25 m/s clips
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bingo Locomotion 2 (AMP) evaluation battery.")
parser.add_argument("--task", type=str, default="Bingo-Locomotion2-AMP-Direct-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds_per_segment", type=float, default=5.0)
parser.add_argument("--settle_seconds", type=float, default=1.0)
parser.add_argument("--sustained_seconds", type=float, default=60.0)
parser.add_argument("--sustained_vx", type=float, default=0.25)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--report", type=str, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import functools  # noqa: E402
print = functools.partial(print, flush=True)  # noqa: A001,E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "rl" / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
from bingo_rl.locomotion2_amp.bingo_dog_amp_env_cfg import BINGO_LEG_SCALE  # noqa: E402

LEGS = ["fl", "fr", "bl", "br"]
EFFORT_LIMIT = 3.0  # v4 leg actuator ceiling (bingo_v4.py)

BATTERY = [
    ("stand", 0.00),
    ("vx_010", 0.10),
    ("vx_020", 0.20),
    ("vx_025", 0.25),
    ("vx_030", 0.30),
]


class SegmentRecorder:
    def __init__(self, name, cmd_vx):
        self.name = name
        self.cmd_vx = cmd_vx
        self.vx, self.roll, self.pitch, self.height = [], [], [], []
        self.torque, self.slip, self.limit_hits, self.actions = [], [], [], []
        self.joint_acc, self.task_reward, self.alive = [], [], []
        self.contact_duty = {leg: [] for leg in LEGS}
        self.disc_policy = []

    def add(self, vx, roll, pitch, height, torque, slip, limit_hit, action,
            joint_acc, task_reward, done, contacts, disc_val):
        self.vx.append(vx); self.roll.append(roll); self.pitch.append(pitch)
        self.height.append(height); self.torque.append(torque); self.slip.append(slip)
        self.limit_hits.append(limit_hit); self.actions.append(action)
        self.joint_acc.append(joint_acc); self.task_reward.append(task_reward)
        self.alive.append(done)
        for i, leg in enumerate(LEGS):
            self.contact_duty[leg].append(contacts[:, i].mean())
        self.disc_policy.append(disc_val)

    def summary(self):
        vx = np.asarray(self.vx)          # (T, N)
        roll = np.asarray(self.roll)      # (T,) already env-averaged
        pitch = np.asarray(self.pitch)
        height = np.asarray(self.height)  # (T, N)
        tq = np.asarray(self.torque)      # (T, N, 12)
        act = np.asarray(self.actions)    # (T, N, 12)
        jacc = np.asarray(self.joint_acc)
        task_r = np.asarray(self.task_reward)
        done = np.asarray(self.alive)     # (T, N)

        ever = np.cumsum(done.astype(int), axis=0) > 0
        m = ~ever
        fell = ever[-1] if len(ever) else np.zeros(0, bool)

        e_vx = vx - self.cmd_vx
        d_act = np.diff(act, axis=0) if len(act) > 1 else np.zeros_like(act)

        def stat(a, fn):
            return float(fn(a[m])) if m.any() else float("nan")

        slip_mean_mm_s = float(1000.0 * np.mean(self.slip)) if self.slip else float("nan")
        cd = {leg: float(np.mean(v)) for leg, v in self.contact_duty.items()}
        front_sym = abs(cd["fl"] - cd["fr"])
        rear_sym = abs(cd["bl"] - cd["br"])

        return {
            "cmd_vx": self.cmd_vx, "steps": int(len(vx)),
            "survival": float(1.0 - fell.mean()) if len(fell) else 0.0,
            "vx_mean": stat(vx, np.mean), "vx_std": stat(vx, np.std), "vx_max": stat(vx, np.max),
            "vx_rmse": stat(e_vx, lambda a: np.sqrt((a ** 2).mean())),
            "roll_rms_deg": float(np.degrees(np.sqrt((roll[m.any(1)] ** 2).mean()))) if m.any() else float("nan"),
            "pitch_rms_deg": float(np.degrees(np.sqrt((pitch[m.any(1)] ** 2).mean()))) if m.any() else float("nan"),
            "height_std_m": stat(height, np.std), "height_ptp_m": float(height[m].max() - height[m].min()) if m.any() else float("nan"),
            "joint_acc_rms": stat(jacc, lambda a: np.sqrt((a ** 2).mean())),
            "action_rate_rms": float(np.sqrt((d_act ** 2).mean())) if d_act.size else 0.0,
            "torque_mean_Nm": stat(tq, lambda a: np.abs(a).mean()),
            "torque_sat_pct": float(100.0 * (np.abs(tq[m]) >= 0.99 * EFFORT_LIMIT).mean()) if m.any() else float("nan"),
            "limit_hit_pct": float(100.0 * np.mean(self.limit_hits)),
            "slip_mean_mm_s": slip_mean_mm_s,
            "contact_duty": cd, "front_lr_symmetry_diff": front_sym, "rear_lr_symmetry_diff": rear_sym,
            "task_reward_mean": stat(task_r, np.mean),
            "disc_policy_mean": float(np.mean(self.disc_policy)) if self.disc_policy else float("nan"),
        }


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    total_s = len(BATTERY) * args_cli.seconds_per_segment + args_cli.sustained_seconds + 10.0
    env_cfg.episode_length_s = total_s
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    out_dir = Path(args_cli.out).parent if args_cli.out else Path.cwd()
    video_dir = Path(args_cli.video_dir) if args_cli.video_dir else out_dir / "eval_video"
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(video_dir), step_trigger=lambda s: s == 0,
            video_length=int(total_s * (1.0 / (env_cfg.sim.dt * env_cfg.decimation))), disable_logger=True)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(wenv, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    print(f"[[ loaded {args_cli.checkpoint}")

    base = env.unwrapped
    robot = base.robot
    dt = base.dt_ctrl
    leg_ids = base.robot_ctrl_indexes
    print(f"[[ battery: {len(BATTERY)} held segments x {args_cli.seconds_per_segment:.1f}s + "
          f"{args_cli.sustained_seconds:.0f}s sustained @ vx={args_cli.sustained_vx}, at {1/dt:.0f} Hz")

    full_battery = list(BATTERY) + [("sustained_60s", args_cli.sustained_vx)]
    steps_per_seg = {name: int(round(args_cli.seconds_per_segment / dt)) for name, _ in BATTERY}
    steps_per_seg["sustained_60s"] = int(round(args_cli.sustained_seconds / dt))
    settle_steps = int(round(args_cli.settle_seconds / dt))

    # sample a fixed held-out expert batch once for the discriminator diagnostic
    expert_batch = base.collect_reference_motions(4096)

    obs, _ = wenv.reset()
    recorders = []
    with torch.inference_mode():
        for name, vx in full_battery:
            rec = SegmentRecorder(name, vx)
            prev_dof_vel = robot.data.joint_vel[:, leg_ids].clone()
            for k in range(steps_per_seg[name]):
                base.commands[:, 0] = vx
                base.commands[:, 1] = 0.0
                base.commands[:, 2] = 0.0
                outputs = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, reward, terminated, truncated, _ = wenv.step(actions)
                done = (terminated | truncated).bool().cpu().numpy().reshape(-1) if terminated is not None \
                    else np.zeros(base.num_envs, bool)

                if k < settle_steps:
                    prev_dof_vel = robot.data.joint_vel[:, leg_ids].clone()
                    continue

                d = robot.data
                vx_m = d.root_lin_vel_b[:, 0].cpu().numpy()
                grav = d.projected_gravity_b.cpu().numpy()
                pitch = float(np.arcsin(np.clip(grav[:, 0], -1, 1)).mean())
                roll = float(np.arctan2(-grav[:, 1], -grav[:, 2]).mean())
                height = d.body_pos_w[:, base.ref_body_index, 2].cpu().numpy()
                tq = d.applied_torque[:, leg_ids].cpu().numpy()
                dof_vel = d.joint_vel[:, leg_ids]
                jacc = ((dof_vel - prev_dof_vel) / dt).cpu().numpy()
                prev_dof_vel = dof_vel.clone()

                _, _, paw_pos = base._leg_world_positions()
                fz = paw_pos[:, :, 2].cpu().numpy()
                fv = torch.linalg.norm(robot.data.body_lin_vel_w[:, base.knee_indexes, :2], dim=-1).cpu().numpy()
                planted = fz < 0.02
                slip = float(fv[planted].mean()) if planted.any() else 0.0

                q = d.joint_pos[:, leg_ids]
                lo = d.soft_joint_pos_limits[:, leg_ids, 0]
                hi = d.soft_joint_pos_limits[:, leg_ids, 1]
                at_limit = ((q <= lo + 1e-3) | (q >= hi - 1e-3)).float().mean().item()

                forces = base.contact_sensor.data.net_forces_w[:, base.contact_foot_idx]
                contacts = (forces.norm(dim=-1) > 0.5).float().cpu().numpy()

                amp_obs = base.amp_observation_buffer.view(-1, base.amp_observation_size)
                disc_out = runner.agent.discriminator.act(
                    {"observations": runner.agent._amp_observation_preprocessor(amp_obs)}, role="discriminator")
                disc_val = float(disc_out[0].mean().item())

                rec.add(vx_m, roll, pitch, height, tq, slip, at_limit,
                        actions.cpu().numpy(), jacc, reward.cpu().numpy().flatten(),
                        done, contacts, disc_val)
            recorders.append(rec)

    disc_expert_out = runner.agent.discriminator.act(
        {"observations": runner.agent._amp_observation_preprocessor(expert_batch)}, role="discriminator")
    disc_expert_mean = float(disc_expert_out[0].mean().item())

    rows = {r.name: r.summary() for r in recorders}
    for r in rows.values():
        r["disc_expert_mean"] = disc_expert_mean
        r["disc_separation"] = abs(disc_expert_mean - r["disc_policy_mean"])

    text = _format(rows, disc_expert_mean)
    print(text)
    if args_cli.report:
        Path(args_cli.report).write_text(text + "\n")
        print(f"[[ wrote {args_cli.report}")
    if args_cli.out:
        Path(args_cli.out).write_text(json.dumps(rows, indent=2))
        print(f"[[ wrote {args_cli.out}")
    env.close()


def _format(rows, disc_expert_mean):
    out = ["", "=" * 130, "BINGO LOCOMOTION 2 (AMP) EVALUATION BATTERY", "=" * 130, "",
           f"discriminator mean logit on held-out EXPERT transitions: {disc_expert_mean:+.3f} "
           "(reference point: higher = more 'real'-looking to the discriminator)", "",
           f"{'segment':14s} {'cmd_vx':>7s} {'vx_mean':>8s} {'vx_std':>7s} {'vx_max':>7s} {'RMSE':>6s} "
           f"{'surv':>5s} {'roll':>6s} {'pitch':>6s} {'h_std':>7s}",
           "-" * 130]
    for name, r in rows.items():
        out.append(f"{name:14s} {r['cmd_vx']:7.2f} {r['vx_mean']:8.3f} {r['vx_std']:7.3f} {r['vx_max']:7.3f} "
                   f"{r['vx_rmse']:6.3f} {100*r['survival']:4.0f}% {r['roll_rms_deg']:6.1f} "
                   f"{r['pitch_rms_deg']:6.1f} {r['height_std_m']:7.4f}")
    out += ["", f"{'segment':14s} {'slip mm/s':>10s} {'torque':>8s} {'sat%':>6s} {'limit%':>7s} "
                f"{'act_rate':>9s} {'jacc_rms':>9s} {'fr_lr_sym':>10s} {'rr_lr_sym':>10s} "
                f"{'task_r':>7s} {'disc_pol':>9s} {'disc_sep':>9s}", "-" * 130]
    for name, r in rows.items():
        out.append(f"{name:14s} {r.get('slip_mean_mm_s', float('nan')):10.1f} {r['torque_mean_Nm']:8.3f} "
                   f"{r['torque_sat_pct']:5.1f}% {r['limit_hit_pct']:6.1f}% {r['action_rate_rms']:9.4f} "
                   f"{r['joint_acc_rms']:9.3f} {r['front_lr_symmetry_diff']:10.3f} "
                   f"{r['rear_lr_symmetry_diff']:10.3f} {r['task_reward_mean']:7.3f} "
                   f"{r['disc_policy_mean']:9.3f} {r['disc_separation']:9.3f}")

    out += ["", "GATES (v1 first-controller scope)", "-" * 130]
    gates = []
    for name, target in [("vx_020", 0.20), ("vx_025", 0.25), ("vx_030", 0.30)]:
        r = rows.get(name, {})
        gates.append((f"{name} tracks commanded speed", r.get("vx_mean", 0) > 0.6 * target,
                     f"vx {r.get('vx_mean', float('nan')):.3f} vs {target:.2f} (needs >60%)"))
    st = rows.get("stand", {})
    gates.append(("stands still on zero command", abs(st.get("vx_mean", 1)) < 0.05,
                 f"vx {st.get('vx_mean', float('nan')):.3f} (needs <0.05)"))
    gates.append(("no falls anywhere", all(r["survival"] > 0.99 for r in rows.values()),
                 f"min survival {min((r['survival'] for r in rows.values()), default=0)*100:.0f}%"))
    sus = rows.get("sustained_60s", {})
    gates.append(("60s sustained walk survives", sus.get("survival", 0) > 0.99,
                 f"survival {sus.get('survival', 0)*100:.0f}%"))
    for label, ok, detail in gates:
        out.append(f"  [{'ok  ' if ok else 'FAIL'}] {label:32s} {detail}")
    n_ok = sum(1 for _, ok, _ in gates if ok)
    out += ["-" * 130, f"  {n_ok}/{len(gates)} gates passed."]
    return "\n".join(out)


if __name__ == "__main__":
    main()
    simulation_app.close()
