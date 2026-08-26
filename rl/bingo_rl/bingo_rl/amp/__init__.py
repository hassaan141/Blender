"""Bingo AMP (dog-style imitation) task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Bingo-AMP-Trot-Direct-v0",
    entry_point="bingo_rl.amp.bingo_amp_env:BingoAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.amp.bingo_amp_env_cfg:BingoAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
