"""Watch a trained Bingo velocity policy, optionally driving it by hand.

    cd ~/robotics/IsaacLab
    ./isaaclab.sh -p ~/Bingo/Blender/rl/tools/play_velocity.py \
        --checkpoint <run>/model_1499.pt

    # hold one command for the whole run instead of the sampled distribution
    ./isaaclab.sh -p ~/Bingo/Blender/rl/tools/play_velocity.py \
        --checkpoint <ckpt> --cmd 0.25 0.0 0.0

For numbers rather than a picture, use eval_velocity.py - that is the script that
decides whether Task 1 is complete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a Bingo velocity policy.")
parser.add_argument("--task", type=str, default="Bingo-Velocity-Flat-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--cmd", type=float, nargs=3, default=None,
                    metavar=("VX", "VY", "YAW"),
                    help="hold this command every step instead of sampling")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(Path.cwd() / "play_video"),
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length, disable_logger=True)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[[ loaded {args_cli.checkpoint}")

    uenv = env.unwrapped
    hold = None
    if args_cli.cmd is not None:
        hold = torch.tensor(args_cli.cmd, device=uenv.device,
                            dtype=torch.float32).repeat(uenv.num_envs, 1)
        print(f"[[ holding command {args_cli.cmd}")

    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            if hold is not None:
                uenv.command_manager.get_term("base_velocity").vel_command_b[:] = hold
            obs, _, _, _ = env.step(policy(obs))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
