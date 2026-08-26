"""Pre-flight: is the back-right leg kinematically mirrored like the others?

Holds Bingo at its default crouch pose (StandTest task, action scale 0) and prints
each foot position in the BASE frame. For a correctly mirrored quadruped, right feet
should be the y-mirror of the matching left feet:
    fr.y ~ -fl.y  (and same x,z)   <- front pair
    br.y ~ -bl.y  (and same x,z)   <- back pair
If br breaks this while fr holds, the br shoulder-pitch axis convention is inverted.

Usage:
  CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p check_leg_symmetry.py --headless --device cuda:0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Bingo-Improved-StandTest-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import functools
import os

print = functools.partial(print, flush=True)

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bingo_rl  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

FOOT_NAMES = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]
JOINTS = ["fl_SP_J", "fr_SP_J", "bl_SP_J", "br_SP_J", "fl_knee", "fr_knee", "bl_knee", "br_knee"]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    ue = env.unwrapped
    robot = ue.scene["robot"]

    # hold default pose: action scale is 0 in StandTest, so any action holds default
    act = torch.zeros((args_cli.num_envs, ue.action_space.shape[1]), device=ue.device)
    env.reset()
    for _ in range(args_cli.settle):
        env.step(act)

    base_pos = robot.data.root_pos_w[0]  # [3]
    base_quat = robot.data.root_quat_w[0]  # [4] wxyz
    # rotate world->base
    from isaaclab.utils.math import quat_apply_inverse

    foot_idx = [robot.body_names.index(n) for n in FOOT_NAMES]
    foot_w = robot.data.body_pos_w[0, foot_idx, :]  # [4,3]
    rel_w = foot_w - base_pos.unsqueeze(0)
    foot_b = quat_apply_inverse(base_quat.unsqueeze(0).expand(4, -1), rel_w)

    print("\n============ FOOT POSITIONS (base frame) ============")
    print(f"base height (world z) = {base_pos[2].item():.4f}")
    print(f"{'foot':8s} {'x':>8s} {'y':>8s} {'z':>8s}")
    d = {}
    for i, n in enumerate(FOOT_NAMES):
        x, y, z = foot_b[i].tolist()
        d[n] = (x, y, z)
        print(f"{n:8s} {x:8.4f} {y:8.4f} {z:8.4f}")

    def mirror_err(left, right):
        lx, ly, lz = d[left]
        rx, ry, rz = d[right]
        return abs(lx - rx), abs(ly + ry), abs(lz - rz)  # x same, y opposite, z same

    fe = mirror_err("fl_knee", "fr_knee")
    be = mirror_err("bl_knee", "br_knee")
    print("\n--- mirror error |L vs R|  (want all ~0; y checks opposite-sign) ---")
    print(f"FRONT (fl vs fr):  dx={fe[0]:.4f}  dy(sum)={fe[1]:.4f}  dz={fe[2]:.4f}")
    print(f"BACK  (bl vs br):  dx={be[0]:.4f}  dy(sum)={be[1]:.4f}  dz={be[2]:.4f}")

    # joint default positions actually applied
    print("\n--- default joint positions (rad) as applied in sim ---")
    for j in JOINTS:
        ji = robot.joint_names.index(j)
        print(f"{j:10s} {robot.data.joint_pos[0, ji].item():+.4f}")

    tol = 0.01
    front_ok = fe[0] < tol and fe[1] < tol and fe[2] < tol
    back_ok = be[0] < tol and be[1] < tol and be[2] < tol
    print("\n============ VERDICT ============")
    print(f"front pair mirrored: {front_ok}")
    print(f"back  pair mirrored: {back_ok}")
    if front_ok and not back_ok:
        print(">>> br leg is NOT mirrored like fr -> back-right axis/convention IS inverted.")
    elif front_ok and back_ok:
        print(">>> both pairs mirror cleanly -> kinematics symmetric; issue is policy/reward, not axis.")
    else:
        print(">>> front pair itself not mirrored -> investigate pose/convention broadly.")
    print("=================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
