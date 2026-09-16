"""Train the Bingo velocity locomotion policy (Task 1).

Standard Isaac Lab external-project training entry point: launch the app first,
then import everything that touches ``isaaclab``.

    cd ~/robotics/IsaacLab
    ./isaaclab.sh -p ~/Bingo/Blender/rl/tools/train_velocity.py \
        --task Bingo-Velocity-Flat-v4-S1-v0 --num_envs 4096 --headless \
        --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"

Curriculum. Train stage 1 first, then warm-start each later stage from the previous
checkpoint with ``--resume_from``. The observation and action shapes are identical
across all 11 stages by construction, so a checkpoint always loads:

    S1  forward only, nominal physics      S7   observation noise
    S2  + stand                            S8   + friction variation
    S3  + wider velocity range             S9   + mass variation
    S4  + turning                          S10  + reset-state variation
    S5  + reverse / lateral                S11  + pushes
    S6  + start/stop transitions

Do NOT jump to S11. The brief is explicit that aggressive randomisation before a
gait exists is how these runs fail, and improved_walking_cfg.py records the same
lesson from this project's own history.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Bingo velocity locomotion.")
parser.add_argument("--task", type=str, default="Bingo-Velocity-Flat-v4-S1-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume_from", type=str, default=None,
                    help="checkpoint .pt to warm-start from (use between stages)")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--video", action="store_true", help="record training videos")
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--video_length", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below must come AFTER the app launch -------------------------
import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

# The rl/README.md-referenced rl/bingo_rl/scripts/train.py this project's tools were
# meant to mirror is not on disk (see TASK1_DESIGN.md "MEDIUM" risks / API_CORROBORATION.txt),
# so isaaclab_rl.rsl_rl.OnPolicyRunner/RslRlVecEnvWrapper/parse_env_cfg/
# load_cfg_from_registry/dump_yaml were never exercised against the local install
# before this session. [MEASURED] Confirmed working as imported above. One gap
# found running this script for the first time: the locally installed rsl-rl-lib
# (5.0.1) deprecated the flat `policy=RslRlPpoActorCriticCfg(...)` runner field in
# favour of separate `actor`/`critic` RslRlMLPModelCfg objects; without translating
# the old-style cfg, rsl_rl.algorithms.ppo.construct_algorithm crashes with
# `KeyError: 'class_name'` because `actor`/`critic` are left as MISSING sentinels.
# Isaac Lab ships a compat shim for exactly this - it's what its own bundled
# scripts/reinforcement_learning/rsl_rl/train.py calls - so use it here too rather
# than hand-rolling the actor/critic translation.
_INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bingo_rl"))
import bingo_rl  # noqa: F401,E402  (registers the Bingo tasks)

LOG_ROOT = Path(__file__).resolve().parents[2] / "logs" / "rsl_rl"


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.device is not None:
        # Match Isaac Lab's own bundled train.py: --device picks the sim device,
        # and the rsl_rl runner needs the same device or obs/network end up split
        # across GPUs. This machine has two GPUs (2080 Ti at cuda:0, 4090 at
        # cuda:1); default AppLauncher device is cuda:0, so --device cuda:1 is
        # needed to actually use the 4090.
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # translate the deprecated policy=RslRlPpoActorCriticCfg(...) field into the
    # actor=/critic= fields rsl-rl 5.0.1 actually reads (see comment above).
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)

    log_dir = LOG_ROOT / agent_cfg.experiment_name / args_cli.task
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[[ logging to {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(log_dir / "videos"),
            step_trigger=lambda s: s % args_cli.video_interval == 0,
            video_length=args_cli.video_length, disable_logger=True)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir),
                            device=agent_cfg.device)
    if args_cli.resume_from:
        print(f"[[ warm-starting from {args_cli.resume_from}")
        runner.load(args_cli.resume_from)

    dump_yaml(str(log_dir / "env_cfg.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent_cfg.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
