"""Deterministic evaluation battery for Bingo velocity locomotion (Task 1).

Task 1 is complete only when Bingo can actually be DRIVEN by the velocity command.
Surviving, or oscillating around one pose, is not success - so this battery does not
report episode return. It drives fixed command trajectories and measures whether the
robot went where it was told.

Battery (each segment is a held command; the last is a transition test):

    stand            [ 0.00,  0.00,  0.00]
    forward          [ 0.25,  0.00,  0.00]
    forward_fast     [ 0.40,  0.00,  0.00]
    reverse          [-0.20,  0.00,  0.00]
    turn_left        [ 0.00,  0.00,  0.80]
    turn_right       [ 0.00,  0.00, -0.80]
    turn_in_place    [ 0.00,  0.00,  1.20]
    lateral_left     [ 0.00,  0.20,  0.00]
    lateral_right    [ 0.00, -0.20,  0.00]
    diagonal         [ 0.20,  0.15,  0.00]
    arc              [ 0.25,  0.00,  0.60]
    start_move_stop  0 -> 0.25 -> 0  (three sub-segments)

Reported per segment and overall: commanded vs measured vx/vy/yaw_rate, RMSE,
survival/falls, root tilt, foot slip, joint-limit hits, mean/max torque, torque
saturation %, and action smoothness.

    cd ~/robotics/IsaacLab
    ./isaaclab.sh -p ~/Bingo/Blender/rl/tools/eval_velocity.py \
        --checkpoint <run>/model_1499.pt --headless
    # add --video to render the battery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bingo velocity evaluation battery.")
parser.add_argument("--task", type=str, default="Bingo-Velocity-Flat-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16,
                    help="the battery is run in parallel across envs and averaged")
parser.add_argument("--seconds_per_segment", type=float, default=5.0)
parser.add_argument("--settle_seconds", type=float, default=1.0,
                    help="ignored at the start of each segment, so the metric measures "
                         "the held command rather than the transient into it")
parser.add_argument("--out", type=str, default=None, help="write JSON metrics here")
parser.add_argument("--report", type=str, default=None, help="write the text report here")
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402

LEG_JOINTS = ("fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
              "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee")
FOOT_LINKS = ("fl_knee", "fr_knee", "bl_knee", "br_knee")

# (name, vx, vy, yaw_rate). A None command means "hold the previous one".
BATTERY = [
    ("stand",           0.00,  0.00,  0.00),
    ("forward",         0.25,  0.00,  0.00),
    ("forward_fast",    0.40,  0.00,  0.00),
    ("reverse",        -0.20,  0.00,  0.00),
    ("turn_left",       0.00,  0.00,  0.80),
    ("turn_right",      0.00,  0.00, -0.80),
    ("turn_in_place",   0.00,  0.00,  1.20),
    ("lateral_left",    0.00,  0.20,  0.00),
    ("lateral_right",   0.00, -0.20,  0.00),
    ("diagonal",        0.20,  0.15,  0.00),
    ("arc",             0.25,  0.00,  0.60),
    ("stop_a",          0.00,  0.00,  0.00),   # start -> move -> stop
    ("move_b",          0.25,  0.00,  0.00),
    ("stop_b",          0.00,  0.00,  0.00),
]


class SegmentRecorder:
    """Accumulates per-step measurements for one held command."""

    def __init__(self, name, cmd):
        self.name = name
        self.cmd = np.asarray(cmd, dtype=float)
        self.v = []          # measured base linear velocity, body frame
        self.w = []          # measured yaw rate
        self.tilt = []       # deg from vertical
        self.torque = []     # |applied torque| over the leg joints
        self.slip = []       # planted-foot horizontal speed
        self.limit_hits = [] # fraction of leg joints at a soft limit
        self.actions = []
        self.alive = []

    def add(self, v, w, tilt, torque, slip, limit_hit, action, done):
        self.v.append(v); self.w.append(w); self.tilt.append(tilt)
        self.torque.append(torque); self.slip.append(slip)
        self.limit_hits.append(limit_hit); self.actions.append(action)
        self.alive.append(done)     # per-step termination flags, resolved in summary()

    def summary(self, effort_limit):
        v = np.asarray(self.v)          # (T, num_envs, 2)
        w = np.asarray(self.w)          # (T, num_envs)
        tq = np.asarray(self.torque)    # (T, num_envs, 12)
        act = np.asarray(self.actions)  # (T, num_envs, 12)
        done = np.asarray(self.alive)   # (T, num_envs) - terminated ON this step
        tilt = np.asarray(self.tilt)    # (T,) - already averaged over envs
        slip = np.asarray(self.slip)    # (T,) - already averaged over planted feet

        # An env that terminates is auto-reset by the vec env, so from the NEXT step
        # its measurements belong to a fresh episode and must not be scored against
        # this segment's command. Mask out every step at or after the first
        # termination, per env - not just the final step's flag.
        ever = np.cumsum(done.astype(int), axis=0) > 0
        m = ~ever
        fell = ever[-1] if len(ever) else np.zeros(0, bool)

        vx, vy = v[..., 0], v[..., 1]
        e_vx = vx - self.cmd[0]
        e_vy = vy - self.cmd[1]
        e_w = w - self.cmd[2]

        # action smoothness: RMS of the first difference, per control step
        d_act = np.diff(act, axis=0) if len(act) > 1 else np.zeros_like(act)

        def stat(a, fn, scale=1.0):
            return float(scale * fn(a[m])) if m.any() else float("nan")

        return {
            "command": self.cmd.tolist(),
            "steps": int(len(v)),
            "survival": float(1.0 - fell.mean()) if len(fell) else 0.0,
            "falls": int(fell.sum()) if len(fell) else 0,
            "vx_mean": stat(vx, np.mean),
            "vy_mean": stat(vy, np.mean),
            "yaw_mean": stat(w, np.mean),
            "vx_rmse": stat(e_vx, lambda a: np.sqrt((a ** 2).mean())),
            "vy_rmse": stat(e_vy, lambda a: np.sqrt((a ** 2).mean())),
            "yaw_rmse": stat(e_w, lambda a: np.sqrt((a ** 2).mean())),
            # tilt and slip are already per-step scalars, so they take the step mask
            # (any env still alive) rather than the per-env one.
            "tilt_mean_deg": float(tilt[m.any(1)].mean()) if m.any() else float("nan"),
            "tilt_max_deg": float(tilt[m.any(1)].max()) if m.any() else float("nan"),
            "slip_mean_mm_s": float(1000 * slip[m.any(1)].mean()) if m.any() else float("nan"),
            "torque_mean_Nm": float(np.abs(tq[m]).mean()) if m.any() else float("nan"),
            "torque_max_Nm": float(np.abs(tq[m]).max()) if m.any() else float("nan"),
            "torque_sat_pct": float(100.0 * (np.abs(tq[m]) >= 0.99 * effort_limit).mean())
            if m.any() else float("nan"),
            "limit_hit_pct": float(100.0 * np.asarray(self.limit_hits).mean()),
            "action_rate_rms": float(np.sqrt((d_act ** 2).mean())) if d_act.size else 0.0,
        }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    # Deterministic: no pushes, no corruption, long episode so the whole battery
    # runs without a scheduled reset interrupting it.
    total_s = len(BATTERY) * args_cli.seconds_per_segment + 5.0
    env_cfg.episode_length_s = total_s
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    out_dir = Path(args_cli.out).parent if args_cli.out else Path.cwd()
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(out_dir / "eval_video"),
            step_trigger=lambda s: s == 0,
            video_length=int(total_s * 24), disable_logger=True)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[[ loaded {args_cli.checkpoint}")

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.step_dt
    leg_ids = [list(robot.data.joint_names).index(j) for j in LEG_JOINTS]
    foot_ids = [list(robot.data.body_names).index(b) for b in FOOT_LINKS]
    effort_limit = 3.0  # v4 leg actuator ceiling (bingo_v4.py)

    steps_per_seg = int(round(args_cli.seconds_per_segment / dt))
    settle_steps = int(round(args_cli.settle_seconds / dt))
    print(f"[[ battery: {len(BATTERY)} segments x {steps_per_seg} steps "
          f"({args_cli.seconds_per_segment:.1f} s at {1/dt:.0f} Hz), "
          f"first {settle_steps} steps of each ignored as transient")

    _run_battery(env, uenv, robot, policy, leg_ids, foot_ids,
                 effort_limit, steps_per_seg, settle_steps)


def _run_battery(env, uenv, robot, policy, leg_ids, foot_ids, effort_limit,
                 steps_per_seg, settle_steps):
    """Drive the battery and report."""
    obs, _ = env.reset()
    recorders = []

    with torch.inference_mode():
        for name, vx, vy, wz in BATTERY:
            rec = SegmentRecorder(name, (vx, vy, wz))
            cmd = torch.tensor([vx, vy, wz], device=uenv.device,
                               dtype=torch.float32).repeat(uenv.num_envs, 1)
            for k in range(steps_per_seg):
                uenv.command_manager.get_term("base_velocity").vel_command_b[:] = cmd
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                if k < settle_steps:
                    continue

                d = robot.data
                v_b = d.root_lin_vel_b[:, :2].cpu().numpy()
                w_b = d.root_ang_vel_b[:, 2].cpu().numpy()
                # tilt: angle between the body z axis and world up
                up = d.projected_gravity_b.cpu().numpy()
                tilt = np.degrees(np.arccos(np.clip(-up[:, 2], -1.0, 1.0)))
                tq = d.applied_torque[:, leg_ids].cpu().numpy()

                # foot slip: horizontal speed of feet that are on the ground
                fz = d.body_pos_w[:, foot_ids, 2].cpu().numpy()
                fv = np.linalg.norm(
                    d.body_lin_vel_w[:, foot_ids, :2].cpu().numpy(), axis=2)
                planted = fz < 0.02
                slip = float(fv[planted].mean()) if planted.any() else 0.0

                q = d.joint_pos[:, leg_ids]
                lo = d.soft_joint_pos_limits[:, leg_ids, 0]
                hi = d.soft_joint_pos_limits[:, leg_ids, 1]
                at_limit = ((q <= lo + 1e-3) | (q >= hi - 1e-3)).float().mean().item()

                done = dones.bool().cpu().numpy() if dones is not None \
                    else np.zeros(uenv.num_envs, bool)

                rec.add(v_b, w_b, tilt.mean(), tq, slip, at_limit,
                        actions.cpu().numpy(), done)
            recorders.append(rec)

    rows = {r.name: r.summary(effort_limit) for r in recorders}
    text = _format(rows)
    print(text)
    if args_cli.report:
        Path(args_cli.report).write_text(text + "\n")
        print(f"[[ wrote {args_cli.report}")
    if args_cli.out:
        Path(args_cli.out).write_text(json.dumps(rows, indent=2))
        print(f"[[ wrote {args_cli.out}")
    env.close()
    return 0


def _format(rows):
    out = ["", "=" * 118,
           "BINGO VELOCITY EVALUATION BATTERY", "=" * 118, "",
           f"{'segment':16s} {'cmd vx':>7s} {'meas':>7s} {'RMSE':>6s} | "
           f"{'cmd vy':>7s} {'meas':>7s} {'RMSE':>6s} | "
           f"{'cmd wz':>7s} {'meas':>7s} {'RMSE':>6s} | {'surv':>5s} {'tilt':>5s}",
           "-" * 118]
    for name, r in rows.items():
        c = r["command"]
        out.append(
            f"{name:16s} {c[0]:7.2f} {r['vx_mean']:7.3f} {r['vx_rmse']:6.3f} | "
            f"{c[1]:7.2f} {r['vy_mean']:7.3f} {r['vy_rmse']:6.3f} | "
            f"{c[2]:7.2f} {r['yaw_mean']:7.3f} {r['yaw_rmse']:6.3f} | "
            f"{100*r['survival']:4.0f}% {r['tilt_mean_deg']:5.1f}")
    out += ["", f"{'segment':16s} {'slip':>10s} {'torque mean':>12s} {'max':>7s} "
                f"{'sat%':>6s} {'limit%':>7s} {'act rate':>9s}", "-" * 118]
    for name, r in rows.items():
        out.append(f"{name:16s} {r['slip_mean_mm_s']:8.1f}mm/s {r['torque_mean_Nm']:12.3f} "
                   f"{r['torque_max_Nm']:7.3f} {r['torque_sat_pct']:5.1f}% "
                   f"{r['limit_hit_pct']:6.1f}% {r['action_rate_rms']:9.4f}")

    # --- the actual pass/fail gate ------------------------------------------
    out += ["", "TASK 1 COMPLETION GATE", "-" * 118]
    gates = []
    fwd = rows.get("forward", {})
    gates.append(("drives forward",
                  fwd.get("vx_mean", 0) > 0.6 * fwd.get("command", [1])[0],
                  f"vx {fwd.get('vx_mean', float('nan')):.3f} vs commanded "
                  f"{fwd.get('command', [0])[0]:.2f} (needs >60%)"))
    rev = rows.get("reverse", {})
    gates.append(("drives backward",
                  rev.get("vx_mean", 0) < 0.6 * rev.get("command", [-1])[0],
                  f"vx {rev.get('vx_mean', float('nan')):.3f} vs "
                  f"{rev.get('command', [0])[0]:.2f}"))
    tip = rows.get("turn_in_place", {})
    gates.append(("turns in place",
                  tip.get("yaw_mean", 0) > 0.6 * tip.get("command", [0, 0, 1])[2],
                  f"yaw {tip.get('yaw_mean', float('nan')):.3f} vs "
                  f"{tip.get('command', [0, 0, 0])[2]:.2f}"))
    lat = rows.get("lateral_left", {})
    gates.append(("moves laterally",
                  lat.get("vy_mean", 0) > 0.5 * lat.get("command", [0, 1])[1],
                  f"vy {lat.get('vy_mean', float('nan')):.3f} vs "
                  f"{lat.get('command', [0, 0])[1]:.2f} (needs >50%)"))
    st = rows.get("stand", {})
    gates.append(("stands still on zero command",
                  abs(st.get("vx_mean", 1)) < 0.05 and abs(st.get("vy_mean", 1)) < 0.05,
                  f"|v| = ({st.get('vx_mean', float('nan')):.3f}, "
                  f"{st.get('vy_mean', float('nan')):.3f}) (needs <0.05)"))
    gates.append(("no falls anywhere",
                  all(r["survival"] > 0.99 for r in rows.values()),
                  f"min survival {min((r['survival'] for r in rows.values()), default=0)*100:.0f}%"))
    gates.append(("torque not saturated",
                  all(r["torque_sat_pct"] < 5.0 for r in rows.values()),
                  f"max saturation {max((r['torque_sat_pct'] for r in rows.values()), default=0):.1f}%"))
    for label, ok, detail in gates:
        out.append(f"  [{'ok  ' if ok else 'FAIL'}] {label:32s} {detail}")
    n_ok = sum(1 for _, ok, _ in gates if ok)
    out += ["-" * 118,
            f"  {n_ok}/{len(gates)} gates passed."
            + ("  TASK 1 COMPLETE." if n_ok == len(gates) else
               "  Task 1 is NOT complete - the policy is not yet driveable.")]
    return "\n".join(out)


if __name__ == "__main__":
    main()
    simulation_app.close()
