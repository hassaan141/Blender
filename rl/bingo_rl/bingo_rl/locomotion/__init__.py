"""Bingo free / command-conditioned locomotion (Task 1).

Registers the velocity-tracking tasks on the validated v4 physics robot. Separate
from the Stage 1-5 animation pipeline in every respect: no motion reference is
loaded and no pipeline file is read or written.

Task ids are suffixed ``-v4-`` to keep them distinct from the pre-existing
``Bingo-Velocity-Flat-v0``, which is registered in bingo_rl/__init__.py against the
older rev_1/rev_3 robot and is left untouched.
"""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Bingo-Velocity-Flat-v4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bingo_velocity_env_cfg:BingoVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BingoVelocityPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Velocity-Flat-v4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bingo_velocity_env_cfg:BingoVelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BingoVelocityPPORunnerCfg_PLAY",
    },
)

# Stance gate: action scale 0, so the robot can only hold STAND_SOLVED. Run this
# before training - if Bingo cannot hold its own stance, no reward will fix it.
gym.register(
    id="Bingo-Velocity-StandTest-v4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bingo_velocity_env_cfg:BingoVelocityStandTestCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BingoVelocityPPORunnerCfg_PLAY",
    },
)

# One task per curriculum stage (see bingo_velocity_env_cfg.TRAINING_STAGES).
# Stage 1 is forward-only on nominal physics; stage 11 is the fully randomised env.
# Train stage N, then warm-start stage N+1 from its checkpoint - the observation and
# action shapes are identical across all stages by construction.
for _s in range(1, 12):
    gym.register(
        id=f"Bingo-Velocity-Flat-v4-S{_s}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.bingo_velocity_env_cfg:BingoVelocityFlatEnvCfg_S{_s}",
            "rsl_rl_cfg_entry_point":
                f"{__name__}.agents.rsl_rl_ppo_cfg:BingoVelocityPPORunnerCfg",
        },
    )
