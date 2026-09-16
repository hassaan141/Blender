"""Deterministic evaluation battery for the reference-guided walk policy
(Bingo-WalkRef-v4-*), scoped exactly to what this first experiment covers:
stand, slow/nominal/fast forward, start->walk, walk->stop. No reverse, strafe,
turn, or domain randomization (those aren't trained -- see bingo_walk_ref_env_cfg.py).

Reports, per segment: commanded vs measured forward velocity + RMSE, survival,
joint-reference error against the Stage 4 q_ref (mean/max, legs), residual
magnitude, action-rate smoothness, torque saturation, and foot-contact agreement
against the reference's own contact schedule -- same metric families as
eval_velocity.py's battery and eval_stage5.py's single-clip report, combined for
this cyclic/speed-scaled reference.

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p ~/Blender/rl/tools/eval_walk_ref.py \
        --checkpoint <run>/agent_XXXX.pt --headless [--video]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bingo walk-ref evaluation battery.")
parser.add_argument("--task", type=str, default="Bingo-WalkRef-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                     help="skrl checkpoint .pt. Omitted => zero residual (Stage 4 PD-only).")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seconds_per_segment", type=float, default=4.0)
parser.add_argument("--settle_seconds", type=float, default=0.5)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--report", type=str, default=None)
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

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import NOMINAL_VX  # noqa: E402

LEG_JOINTS = ("fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
              "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee")
FOOT_LINKS = ("fl_knee", "fr_knee", "bl_knee", "br_knee")

# (name, cmd_vx as a fraction of NOMINAL_VX). "start_walk"/"walk_stop" switch the
# command partway through their own segment (see the special-case in the loop).
BATTERY = [
    ("stand",         0.0),
    ("slow",          0.5),
    ("nominal",       1.0),
    ("fast",          1.3),
    ("start_walk",    None),   # 0 -> 1.0 nominal, switching at the segment midpoint
    ("walk_stop",     None),   # 1.0 nominal -> 0, switching at the segment midpoint
]


class SegmentRecorder:
    def __init__(self, name, cmd_vx):
        self.name = name
        self.cmd_vx = cmd_vx
        self.vx = []; self.qerr = []; self.resid = []; self.actions = []
        self.torque = []; self.contacts_match = []; self.alive = []

    def add(self, vx, qerr, resid, action, torque, cmatch, done):
        self.vx.append(vx); self.qerr.append(qerr); self.resid.append(resid)
        self.actions.append(action); self.torque.append(torque)
        self.contacts_match.append(cmatch); self.alive.append(done)

    def summary(self, effort_limit):
        vx = np.asarray(self.vx)            # (T,E)
        qerr = np.asarray(self.qerr)        # (T,E,12)
        resid = np.asarray(self.resid)      # (T,E,12)
        act = np.asarray(self.actions)      # (T,E,12)
        tq = np.asarray(self.torque)        # (T,E,12)
        cmatch = np.asarray(self.contacts_match)  # (T,E)
        done = np.asarray(self.alive)       # (T,E)

        ever = np.cumsum(done.astype(int), axis=0) > 0
        m = ~ever
        fell = ever[-1] if len(ever) else np.zeros(0, bool)

        cmd = self.cmd_vx if np.isscalar(self.cmd_vx) else np.asarray(self.cmd_vx)
        e_vx = vx - (cmd if np.isscalar(cmd) else cmd[:, None].T)
        d_act = np.diff(act, axis=0) if len(act) > 1 else np.zeros_like(act)

        def stat(a, fn):
            return float(fn(a[m])) if m.any() else float("nan")

        return {
            "command_vx": float(cmd) if np.isscalar(cmd) else "time-varying",
            "steps": int(len(vx)),
            "survival": float(1.0 - fell.mean()) if len(fell) else 0.0,
            "vx_mean": stat(vx, np.mean),
            "vx_rmse": stat(e_vx, lambda a: np.sqrt((a ** 2).mean())),
            "qerr_leg_mean": stat(qerr, np.mean),
            "qerr_leg_max": float(qerr[m].max()) if m.any() else float("nan"),
            "residual_mean": stat(np.abs(resid), np.mean),
            "residual_max": float(np.abs(resid)[m].max()) if m.any() else float("nan"),
            "action_rate_rms": float(np.sqrt((d_act ** 2).mean())) if d_act.size else 0.0,
            "torque_mean_Nm": stat(np.abs(tq), np.mean),
            "torque_max_Nm": float(np.abs(tq)[m].max()) if m.any() else float("nan"),
            "torque_sat_pct": float(100.0 * (np.abs(tq[m]) >= 0.99 * effort_limit).mean())
            if m.any() else float("nan"),
            "contact_agreement_pct": stat(cmatch, np.mean) * 100.0 if m.any() else float("nan"),
        }


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg,
                    render_mode="rgb_array" if args_cli.video else None)
    out_dir = Path(args_cli.out).parent if args_cli.out else Path.cwd()
    total_s = len(BATTERY) * args_cli.seconds_per_segment + 2.0
    if args_cli.video:
        # The camera/render pipeline initializes lazily and asynchronously; without
        # a checkpoint to load (which incidentally buys enough wall-clock for it to
        # finish) RecordVideo's step_trigger=(step==0) can fire before it's ready,
        # raising "Cannot render 'rgb_array' ... NO_GUI_OR_RENDERING" [MEASURED,
        # this session: reproduced twice with the pdonly/zero-residual path].
        import time
        env.reset()
        for _ in range(10):
            env.render()
            time.sleep(0.2)
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(out_dir / "eval_video"),
            step_trigger=lambda s: s == 0,
            video_length=int(total_s * 24), disable_logger=True)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    runner = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner
        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wenv, agent_cfg)
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)
        print(f"[[ policy loaded: {args_cli.checkpoint}")
    else:
        print("[[ no checkpoint: residual forced to ZERO (Stage 4 PD-only baseline)")

    effort_limit = 3.0  # v4 leg actuator ceiling (bingo_v4.py)
    steps_per_seg = int(round(args_cli.seconds_per_segment / base.step_dt))
    settle_steps = int(round(args_cli.settle_seconds / base.step_dt))
    nominal_vx = float(NOMINAL_VX)
    n_leg = len(base.ctrl_dof_names)
    zero_action = torch.zeros((base.num_envs, n_leg), device=base.device)

    obs, _ = wenv.reset()
    recorders = []
    with torch.inference_mode():
        for name, frac in BATTERY:
            rec = SegmentRecorder(name, 0.0 if frac is None else frac * nominal_vx)
            for k in range(steps_per_seg):
                if frac is None:
                    lo, hi = (0.0, nominal_vx) if name == "start_walk" else (nominal_vx, 0.0)
                    v = lo if k < steps_per_seg // 2 else hi
                else:
                    v = frac * nominal_vx
                base._cmd_vx[:] = v
                base._cmd_timer[:] = 1e6  # hold: eval drives the command explicitly

                if runner is not None:
                    out = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
                    action = out[-1].get("mean_actions", out[0])
                else:
                    action = zero_action
                ref_dof = base._ff_leg_q if hasattr(base, "_ff_leg_q") else None
                obs, _, term, tout, _ = wenv.step(action)

                if k < settle_steps:
                    continue

                d = base.robot.data
                root_quat = d.body_quat_w[:, base.ref_body_index]
                root_lin_w = d.body_lin_vel_w[:, base.ref_body_index]
                from isaaclab.utils.math import quat_rotate_inverse
                vx_b = quat_rotate_inverse(root_quat, root_lin_w)[:, 0].cpu().numpy()

                cur_dof = d.joint_pos[:, base.robot_ctrl_indexes].cpu().numpy()
                if ref_dof is not None:
                    qerr = np.abs(cur_dof - ref_dof.cpu().numpy())
                else:
                    qerr = np.zeros_like(cur_dof)

                resid = (base._res_scale.cpu().numpy() * action.cpu().numpy())
                tq = d.applied_torque[:, base.robot_ctrl_indexes].cpu().numpy()

                tips_w, _ = base._foot_tips_local()
                act_c = (tips_w[:, :, 2] < 0.03).float()
                if base._ref_contacts is not None:
                    times = base._current_times()
                    fidx = np.clip(np.round(times * base._ref_fps).astype(int),
                                    0, base._n_ref_frames - 1)
                    rc = base._ref_contacts[fidx].cpu().numpy()
                    cmatch = (act_c.cpu().numpy() == rc).mean(axis=1)
                else:
                    cmatch = np.zeros(base.num_envs)

                done = (term | tout).reshape(-1).cpu().numpy() if term is not None \
                    else np.zeros(base.num_envs, bool)
                rec.add(vx_b, qerr, resid, action.cpu().numpy(), tq, cmatch, done)
            recorders.append(rec)

    rows = {r.name: r.summary(effort_limit) for r in recorders}
    text = _format(rows)
    Path("/tmp/eval_walk_ref_report.txt").write_text(text)
    print(text, flush=True)
    sys.stdout.flush()
    if args_cli.report:
        Path(args_cli.report).write_text(text + "\n")
        print(f"[[ wrote {args_cli.report}")
    if args_cli.out:
        Path(args_cli.out).write_text(json.dumps(rows, indent=2))
        print(f"[[ wrote {args_cli.out}")
    env.close()


def _format(rows):
    out = ["", "=" * 118, "BINGO WALK-REF EVALUATION BATTERY", "=" * 118, "",
           f"{'segment':14s} {'cmd_vx':>8s} {'meas_vx':>8s} {'RMSE':>6s} {'surv':>5s} "
           f"{'qerr_mean':>10s} {'qerr_max':>9s} {'resid_mean':>11s} {'act_rate':>9s} "
           f"{'tq_mean':>8s} {'tq_sat%':>8s} {'contact%':>9s}", "-" * 118]
    for name, r in rows.items():
        cmd = r["command_vx"] if isinstance(r["command_vx"], float) else 0.0
        out.append(
            f"{name:14s} {cmd:8.3f} {r['vx_mean']:8.3f} {r['vx_rmse']:6.3f} "
            f"{100*r['survival']:4.0f}% {r['qerr_leg_mean']:10.4f} {r['qerr_leg_max']:9.4f} "
            f"{r['residual_mean']:11.4f} {r['action_rate_rms']:9.4f} "
            f"{r['torque_mean_Nm']:8.3f} {r['torque_sat_pct']:7.1f}% {r['contact_agreement_pct']:8.1f}%")
    out += ["", "SCOPED GATE (stand / slow / nominal / fast / start-walk / walk-stop only)", "-" * 118]
    gates = []
    st = rows.get("stand", {})
    gates.append(("stands still on zero command", abs(st.get("vx_mean", 1)) < 0.08,
                  f"vx {st.get('vx_mean', float('nan')):.3f} (needs <0.08)"))
    nom = rows.get("nominal", {})
    gates.append(("tracks nominal forward speed", nom.get("vx_mean", 0) > 0.6 * nom.get("command_vx", 1),
                  f"vx {nom.get('vx_mean', float('nan')):.3f} vs {nom.get('command_vx', 0):.3f} (needs >60%)"))
    gates.append(("no falls anywhere", all(r["survival"] > 0.99 for r in rows.values()),
                  f"min survival {min((r['survival'] for r in rows.values()), default=0)*100:.0f}%"))
    gates.append(("gait resembles Stage 4 (joint tracking)",
                  all(r["qerr_leg_mean"] < 0.35 for r in rows.values()),
                  f"max mean qerr {max((r['qerr_leg_mean'] for r in rows.values()), default=0):.4f} rad (needs <0.35)"))
    for label, ok, detail in gates:
        out.append(f"  [{'ok  ' if ok else 'FAIL'}] {label:40s} {detail}")
    n_ok = sum(1 for _, ok, _ in gates if ok)
    out += ["-" * 118, f"  {n_ok}/{len(gates)} scoped gates passed."]
    return "\n".join(out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        Path("/tmp/eval_walk_ref_crash.txt").write_text(tb)
        print(tb, flush=True)
        raise
    simulation_app.close()
