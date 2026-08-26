"""Objective per-foot gait metrics for a trained Bingo policy.

Loads a checkpoint (same way as play.py), runs a fixed forward command, and
reports per-foot contact fraction, cadence (steps/s), mean swing air-time, and
mean in-contact horizontal slip speed, plus base height/orientation/velocity
tracking. Lets us diagnose "leg X not working" with numbers instead of video.

Usage:
  ./isaaclab.sh -p eval_gait.py --task Bingo-Improved-Walking-Flat-Play-v0 \
      --num_envs 64 --headless [--checkpoint /abs/path/model_999.pt]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Bingo-Improved-Walking-Flat-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
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
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bingo_rl  # noqa: F401
import importlib.metadata as metadata

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")
FOOT_NAMES = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # pin command to a straight, constant forward push so per-foot metrics measure
    # gait quality/symmetry (not turning). Same probe across all versions.
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[eval] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    ue = env.unwrapped
    robot = ue.scene["robot"]
    contact = ue.scene["contact_forces"]
    dt = ue.step_dt

    foot_body_idx = [robot.body_names.index(n) for n in FOOT_NAMES]
    contact_body_idx = [contact.body_names.index(n) for n in FOOT_NAMES]
    dev = ue.device
    N = args_cli.num_envs

    # accumulators
    contact_steps = torch.zeros(4, device=dev)
    slip_sum = torch.zeros(4, device=dev)
    liftoffs = torch.zeros(4, device=dev)
    air_sum = torch.zeros(4, device=dev)  # accumulated completed-swing durations
    swing_count = torch.zeros(4, device=dev)
    cur_air = torch.zeros(N, 4, device=dev)
    prev_contact = torch.zeros(N, 4, dtype=torch.bool, device=dev)
    h_sum = h2_sum = 0.0
    grav_xy_sum = 0.0
    vx_err_sum = 0.0
    cmd_vx_sum = 0.0
    act_vx_sum = 0.0
    total = 0

    obs = env.get_observations()
    for t in range(args_cli.warmup + args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        if t < args_cli.warmup:
            continue

        forces = contact.data.net_forces_w[:, contact_body_idx, :]  # [N,4,3]
        in_contact = forces.norm(dim=-1) > 1.0  # [N,4]
        foot_vel = robot.data.body_lin_vel_w[:, foot_body_idx, :2]  # [N,4,2]
        foot_speed = foot_vel.norm(dim=-1)  # [N,4]

        contact_steps += in_contact.sum(dim=0).float()
        slip_sum += (foot_speed * in_contact.float()).sum(dim=0)

        # air-time bookkeeping
        cur_air += (~in_contact).float() * dt
        touchdown = in_contact & (~prev_contact)
        liftoff = (~in_contact) & prev_contact
        liftoffs += liftoff.sum(dim=0).float()
        # on touchdown, the just-completed swing air time is cur_air
        air_sum += (cur_air * touchdown.float()).sum(dim=0)
        swing_count += touchdown.sum(dim=0).float()
        cur_air = cur_air * (~in_contact).float()  # reset air on contact
        prev_contact = in_contact

        h = robot.data.root_pos_w[:, 2]
        h_sum += h.sum().item()
        h2_sum += (h * h).sum().item()
        grav_xy_sum += robot.data.projected_gravity_b[:, :2].norm(dim=-1).sum().item()
        cmd = ue.command_manager.get_command("base_velocity")  # [N,3] vx,vy,wz
        act_vx = robot.data.root_lin_vel_b[:, 0]
        vx_err_sum += (cmd[:, 0] - act_vx).abs().sum().item()
        cmd_vx_sum += cmd[:, 0].sum().item()
        act_vx_sum += act_vx.sum().item()
        total += N

    steps = args_cli.steps
    print("\n================= GAIT REPORT =================")
    print(f"envs={N}  steps={steps}  dt={dt:.4f}s  window={steps*dt:.2f}s")
    mean_h = h_sum / total
    std_h = (h2_sum / total - mean_h ** 2) ** 0.5
    print(f"base height  mean={mean_h:.3f}  std={std_h:.3f}  (target ~0.16-0.20)")
    print(f"orientation  mean|grav_xy|={grav_xy_sum/total:.3f}  (0=perfectly level)")
    print(f"cmd vx mean={cmd_vx_sum/total:.3f}  actual vx mean={act_vx_sum/total:.3f}  |err|={vx_err_sum/total:.3f}")
    print("\n  foot    contact%   cadence(Hz)  meanAir(s)  slip(m/s)")
    for i, name in enumerate(FOOT_NAMES):
        cfrac = 100.0 * contact_steps[i].item() / (steps * N)
        cadence = liftoffs[i].item() / (steps * dt * N)  # swings per second per foot
        mean_air = (air_sum[i].item() / swing_count[i].item()) if swing_count[i] > 0 else 0.0
        mean_slip = (slip_sum[i].item() / contact_steps[i].item()) if contact_steps[i] > 0 else 0.0
        print(f"  {name:7s} {cfrac:7.1f}   {cadence:9.2f}   {mean_air:9.3f}   {mean_slip:7.3f}")
    print("===============================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
