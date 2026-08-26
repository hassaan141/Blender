"""Stage 5 task registration: residual RL on the validated Stage 4 Timid reference.

The PPO config is REUSED as-is from bingo_rl.track.agents (skrl_ppo_cfg.yaml), the
same one that produced the project's best tracker (0.099 rad on deadpan). No new
agent config file is introduced.
"""

import gymnasium as gym

# reuse the existing tracker's PPO config package (no duplicate agent cfg)
from bingo_rl.track import agents as _track_agents

gym.register(
    id="Bingo-Stage5-Timid-Direct-v0",
    entry_point="bingo_rl.stage5.bingo_stage5_env:BingoStage5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.stage5.bingo_stage5_env_cfg:BingoStage5TimidEnvCfg",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-Stage5-Timid-Play-v0",
    entry_point="bingo_rl.stage5.bingo_stage5_env:BingoStage5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.stage5.bingo_stage5_env_cfg:BingoStage5TimidPlayEnvCfg",
        "skrl_cfg_entry_point": f"{_track_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
