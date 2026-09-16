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
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to a .pt checkpoint, or the literal 'none' to skip "
                    "loading and use the untrained actor-critic as-is (only "
                    "meaningful on the StandTest task, where action scale is 0 so "
                    "the policy's output never reaches the joints - see HANDOFF.md "
                    "step 2, the stance gate)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--cmd", type=float, nargs=3, default=None,
                    metavar=("VX", "VY", "YAW"),
                    help="hold this command every step instead of sampling")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--closeup", action="store_true",
                    help="track env_index 0's robot with a close trailing camera "
                    "instead of the default static wide shot of the whole env "
                    "grid - use with a small --num_envs so the tracked robot "
                    "isn't lost in a crowd.")
parser.add_argument("--num_steps", type=int, default=480,
                    help="bounded run length (steps at 24 Hz control rate; 480 = "
                    "20 s). Headless has no window to close, so "
                    "simulation_app.is_running() never goes false on its own - "
                    "the loop must be bounded explicitly. If --video is set and "
                    "video_length > num_steps, num_steps is raised to match.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# See train_velocity.py for why this is needed: the locally installed rsl-rl-lib
# (5.0.1) deprecated the flat `policy=` runner field in favour of `actor`/`critic`,
# and Bingo's agent cfgs still use the old field. Without this translation,
# OnPolicyRunner crashes with KeyError: 'class_name' [MEASURED].
_INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.closeup:
        # Isaac Lab's ViewportCameraController re-centers eye/lookat on the
        # tracked asset's root every render step when origin_type="asset_root"
        # (isaaclab/envs/ui/viewport_camera_controller.py) - unlike "world"/"env",
        # which are static. Trailing offset chosen for Bingo's ~0.2 m stance
        # height: behind (-x) and to the side, looking slightly down at the body.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-0.55, 0.55, 0.30)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.05)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

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
    if args_cli.checkpoint == "none":
        print("[[ --checkpoint none: skipping load, using the untrained "
              "actor-critic. Only valid where action scale is 0 (StandTest) - "
              "the policy's output is scaled to zero and never reaches the "
              "joints, so an untrained network is equivalent to a trained one.")
    else:
        runner.load(args_cli.checkpoint)
        print(f"[[ loaded {args_cli.checkpoint}")
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    hold = None
    if args_cli.cmd is not None:
        hold = torch.tensor(args_cli.cmd, device=uenv.device,
                            dtype=torch.float32).repeat(uenv.num_envs, 1)
        print(f"[[ holding command {args_cli.cmd}")

    num_steps = max(args_cli.num_steps, args_cli.video_length if args_cli.video else 0)
    robot = uenv.scene["robot"]
    n_envs = uenv.num_envs   # cache: uenv.scene is gone once env.close() runs below
    heights = []
    tilts_deg = []

    obs, _ = env.reset()
    with torch.inference_mode():
        step = 0
        while simulation_app.is_running() and step < num_steps:
            if hold is not None:
                uenv.command_manager.get_term("base_velocity").vel_command_b[:] = hold
            obs, _, _, _ = env.step(policy(obs))
            heights.append(robot.data.root_pos_w[:, 2].clone())
            # projected_gravity_b -> (0,0,-1) when upright; tilt = angle off that.
            gz = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
            tilts_deg.append(torch.rad2deg(torch.acos(-gz)))
            step += 1
    env.close()

    heights = torch.stack(heights)   # [steps, num_envs]
    tilts_deg = torch.stack(tilts_deg)
    print(f"[[ ran {step} steps ({step / 24.0:.1f} s at 24 Hz) over {n_envs} envs")
    print(f"[[ base height  min={heights.min():.4f}  mean={heights.mean():.4f}  "
          f"max={heights.max():.4f}  m  (per-env final: "
          f"{[round(v, 4) for v in heights[-1].tolist()]})")
    print(f"[[ tilt off-vertical  mean={tilts_deg.mean():.2f}  max={tilts_deg.max():.2f}  deg")
    if args_cli.video:
        video_path = Path.cwd() / "play_video"
        print(f"[[ video written under: {video_path.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
