"""Bingo quadruped RL tasks. Importing this package registers the gym environments."""

import gymnasium as gym

from . import amp  # noqa: F401  (registers Bingo-AMP-Trot-Direct-v0)
from . import track  # noqa: F401  (registers Bingo-Track-Deadpan-Direct-v0)
from . import track_expr  # noqa: F401  (registers Bingo-TrackExpr-Deadpan-Direct-v0)
from . import track_v4  # noqa: F401  (registers Bingo-TrackV4-Deadpan-Direct-v0, 21 DOF incl. ears)
from . import stage5  # noqa: F401  (registers Bingo-Stage5-Timid-Direct-v0, residual RL on Stage 4)

gym.register(
    id="Bingo-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.env_cfg:BingoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "bingo_rl.agents:BingoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.env_cfg:BingoFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "bingo_rl.agents:BingoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Improved-Walking-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedFlatPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Improved-Walking-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedFlatPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Improved-StandTest-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedStandTestCfg_PLAY",
        "rsl_rl_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedFlatPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.env_cfg:BingoRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "bingo_rl.agents:BingoRoughPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Improved-Walking-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedRoughPPORunnerCfg",
    },
)

gym.register(
    id="Bingo-Improved-Walking-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "bingo_rl.improved_walking_cfg:BingoImprovedRoughPPORunnerCfg",
    },
)
