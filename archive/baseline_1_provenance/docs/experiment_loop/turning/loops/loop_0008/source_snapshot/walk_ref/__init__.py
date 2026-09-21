"""Reference-guided walk task registration: residual RL on a cyclic, speed-scaled
Stage 4 walk (vs stage5's single-pass Timid tracker). PPO config REUSED as-is from
bingo_rl.track.agents (skrl_ppo_cfg.yaml) -- same convention as stage5, no new
agent config file.
"""

import gymnasium as gym

from bingo_rl.track import agents as _track_agents

gym.register(
    id="Bingo-WalkRef-v4-A-v0",
    entry_point="bingo_rl.walk_ref.bingo_walk_ref_env:BingoWalkRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.walk_ref.bingo_walk_ref_env_cfg:BingoWalkRefEnvCfg",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-WalkRef-v4-B-v0",
    entry_point="bingo_rl.walk_ref.bingo_walk_ref_env:BingoWalkRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.walk_ref.bingo_walk_ref_env_cfg:BingoWalkRefEnvCfg_B",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-WalkRef-v4-C-v0",
    entry_point="bingo_rl.walk_ref.bingo_walk_ref_env:BingoWalkRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.walk_ref.bingo_walk_ref_env_cfg:BingoWalkRefEnvCfg_C",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-WalkRef-v4-D-v0",
    entry_point="bingo_rl.walk_ref.bingo_walk_ref_env:BingoWalkRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.walk_ref.bingo_walk_ref_env_cfg:BingoWalkRefEnvCfg_D",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-WalkRef-v4-Play-v0",
    entry_point="bingo_rl.walk_ref.bingo_walk_ref_env:BingoWalkRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.walk_ref.bingo_walk_ref_env_cfg:BingoWalkRefPlayEnvCfg",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
