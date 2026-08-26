"""Objective gait check for the trained Bingo AMP policy (skrl).

Loads a skrl AMP checkpoint, runs the env, and reports: forward speed, base
height/tilt, and per-foot fore-aft phase correlations (diagonal pairs should be
IN phase, lateral/front-back ANTI phase == a trot). This is the objective judge
that the fixed-wide record camera can't give (legs too small on screen).

Usage:
  CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p eval_amp_gait.py \
      --task Bingo-AMP-Trot-Direct-v0 --algorithm AMP --num_envs 16 --headless \
      --device cuda:0 --checkpoint <best_agent.pt> --steps 300
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Bingo-AMP-Trot-Direct-v0")
parser.add_argument("--algorithm", type=str, default="AMP")
parser.add_argument("--ml_framework", type=str, default="torch")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
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
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bingo_rl  # noqa: F401

FOOT_NAMES = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    ue = env.unwrapped
    robot = ue.scene["robot"]
    foot_idx = [robot.body_names.index(n) for n in FOOT_NAMES]
    base_idx = robot.body_names.index("origin")

    fx_hist = [[] for _ in range(4)]  # foot fore-aft rel base
    fz_hist = [[] for _ in range(4)]  # foot vertical rel base
    vx_sum = h_sum = tilt_sum = 0.0
    n = 0

    from isaaclab.utils.math import quat_apply_inverse

    obs, _ = env.reset()
    states = env.state()
    for t in range(args_cli.warmup + args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
            states = env.state()
        if t < args_cli.warmup:
            continue
        bpos = robot.data.body_pos_w[:, base_idx]           # [N,3]
        bq = robot.data.body_quat_w[:, base_idx]            # [N,4]
        fpos = robot.data.body_pos_w[:, foot_idx]           # [N,4,3]
        rel = quat_apply_inverse(bq.unsqueeze(1).expand(-1, 4, -1), fpos - bpos.unsqueeze(1))
        for i in range(4):
            fx_hist[i].append(rel[:, i, 0].mean().item())
            fz_hist[i].append(rel[:, i, 2].mean().item())
        vx_sum += robot.data.root_lin_vel_b[:, 0].mean().item()
        h_sum += bpos[:, 2].mean().item()
        tilt_sum += robot.data.projected_gravity_b[:, :2].norm(dim=-1).mean().item()
        n += 1

    fx = [torch.tensor(h) for h in fx_hist]
    fz = [torch.tensor(h) for h in fz_hist]
    print("\n================ AMP GAIT REPORT ================")
    print(f"steps={n}  envs={args_cli.num_envs}")
    print(f"forward speed vx = {vx_sum/n:.3f} m/s   (target 0.5)")
    print(f"base height      = {h_sum/n:.3f} m      (v7 ~0.19)")
    print(f"tilt |grav_xy|   = {tilt_sum/n:.3f}     (0 = level)")
    print("\nper-foot fore-aft amplitude (m) and vertical amplitude (m):")
    for i, name in enumerate(FOOT_NAMES):
        print(f"  {name:8s} fore-aft={fx[i].max()-fx[i].min():.3f}  vert={fz[i].max()-fz[i].min():.3f}")
    print("\nphase correlations (fore-aft) -- TROT signature:")
    print(f"  fl-br diagonal (want +): {corr(fx[0],fx[3]):+.2f}")
    print(f"  fr-bl diagonal (want +): {corr(fx[1],fx[2]):+.2f}")
    print(f"  fl-fr lateral  (want -): {corr(fx[0],fx[1]):+.2f}")
    print(f"  fl-bl frontback(want -): {corr(fx[0],fx[2]):+.2f}")
    print("=================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
