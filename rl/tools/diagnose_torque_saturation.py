"""Per-joint torque-saturation diagnostic for the Bingo velocity policy.

`eval_velocity.py`'s "torque not saturated" gate pools all 12 leg joints into a
single sat% per segment, which is enough to know the gate fails but not why.
This script drives the same command sequence up through the worst offender
(`forward_fast`, per TASK1_STATUS_AND_PLAN.md S6 run: 16% pooled) and, for that
segment only, breaks saturation down per joint and correlates it against
per-joint |delta action| - the two candidate causes are (a) specific stiff-gain
joints (SP/knee, Kp=120) sitting near their torque ceiling structurally, vs
(b) abrupt step-to-step action jumps spiking torque transiently regardless of
joint. Read-only: loads a checkpoint, runs no training.

    source /pub0/muhammadf/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p ~/Blender/rl/tools/diagnose_torque_saturation.py \
        --checkpoint ~/Blender/logs/rsl_rl/bingo_velocity_v4/Bingo-Velocity-Flat-v4-S6-v0/model_8994.pt \
        --headless --device cuda:1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Per-joint torque saturation diagnostic.")
parser.add_argument("--task", type=str, default="Bingo-Velocity-Flat-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds_per_segment", type=float, default=5.0)
parser.add_argument("--settle_seconds", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

_INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402

LEG_JOINTS = ("fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
              "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee")

# Same lead-in as eval_velocity.py's BATTERY, truncated to the worst offender -
# forward_fast needs the stand->forward transient behind it to be representative.
LEAD_IN = [
    ("stand",        0.00, 0.00, 0.00),
    ("forward",      0.25, 0.00, 0.00),
    ("forward_fast", 0.40, 0.00, 0.00),
]
RECORD_SEGMENT = "forward_fast"


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = len(LEAD_IN) * args_cli.seconds_per_segment + 5.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[[ loaded {args_cli.checkpoint}")

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.step_dt
    leg_ids = [list(robot.data.joint_names).index(j) for j in LEG_JOINTS]
    effort_limit = 3.0  # v4 leg actuator ceiling (bingo_v4.py)
    steps_per_seg = int(round(args_cli.seconds_per_segment / dt))
    settle_steps = int(round(args_cli.settle_seconds / dt))

    torques = []  # (T, num_envs, 12), forward_fast only, post-settle
    actions = []  # (T, num_envs, 12), forward_fast only, post-settle (for delta)
    prev_action = None  # last action of the PRIOR step, incl. pre-settle, for delta continuity

    obs, _ = env.reset()
    with torch.inference_mode():
        for name, vx, vy, wz in LEAD_IN:
            cmd = torch.tensor([vx, vy, wz], device=uenv.device, dtype=torch.float32).repeat(
                uenv.num_envs, 1)
            for k in range(steps_per_seg):
                uenv.command_manager.get_term("base_velocity").vel_command_b[:] = cmd
                act = policy(obs)
                obs, _, dones, _ = env.step(act)
                act_np = act.cpu().numpy()

                if name == RECORD_SEGMENT and k >= settle_steps:
                    torques.append(robot.data.applied_torque[:, leg_ids].cpu().numpy())
                    actions.append(act_np)
                prev_action = act_np
    env.close()

    tq = np.asarray(torques)     # (T, E, 12)
    act = np.asarray(actions)    # (T, E, 12)
    d_act = np.zeros_like(act)
    d_act[1:] = np.abs(np.diff(act, axis=0))  # step 0 has no in-segment predecessor -> 0

    sat = np.abs(tq) >= 0.99 * effort_limit  # (T, E, 12) bool

    print()
    print("=" * 100)
    print(f"PER-JOINT TORQUE SATURATION - {RECORD_SEGMENT} segment "
          f"({tq.shape[0]} steps x {tq.shape[1]} envs, post-settle)")
    print("=" * 100)
    print(f"{'joint':10s} {'mean|tq|':>9s} {'max|tq|':>8s} {'sat%':>7s}  "
          f"{'|dact| all':>11s} {'|dact| sat':>11s} {'|dact| unsat':>13s} {'corr(|dact|,sat)':>17s}")
    print("-" * 100)
    for j, jname in enumerate(LEG_JOINTS):
        tq_j = np.abs(tq[..., j]).ravel()
        sat_j = sat[..., j].ravel()
        dact_j = d_act[..., j].ravel()
        sat_pct = 100.0 * sat_j.mean()
        dact_all = dact_j.mean()
        dact_sat = dact_j[sat_j].mean() if sat_j.any() else float("nan")
        dact_unsat = dact_j[~sat_j].mean() if (~sat_j).any() else float("nan")
        # Pearson correlation between |delta action| and the binary saturation event.
        if dact_j.std() > 0 and sat_j.std() > 0:
            corr = float(np.corrcoef(dact_j, sat_j.astype(float))[0, 1])
        else:
            corr = float("nan")
        print(f"{jname:10s} {tq_j.mean():9.3f} {tq_j.max():8.3f} {sat_pct:6.1f}%  "
              f"{dact_all:11.4f} {dact_sat:11.4f} {dact_unsat:13.4f} {corr:17.3f}")

    print("-" * 100)
    print("by group (SY / SP / knee):")
    groups = {"SY": [0, 3, 6, 9], "SP": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
    for gname, idxs in groups.items():
        sat_g = sat[..., idxs].ravel()
        dact_g = d_act[..., idxs].ravel()
        dact_sat_g = dact_g[sat_g].mean() if sat_g.any() else float("nan")
        dact_unsat_g = dact_g[~sat_g].mean() if (~sat_g).any() else float("nan")
        print(f"  {gname:5s} sat%={100*sat_g.mean():5.1f}  |dact| sat={dact_sat_g:.4f}  "
              f"|dact| unsat={dact_unsat_g:.4f}")
    print(f"\npooled sat% (should match eval_velocity.py's forward_fast row): "
          f"{100.0 * sat.mean():.1f}%")


if __name__ == "__main__":
    main()
    simulation_app.close()
