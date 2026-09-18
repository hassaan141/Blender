"""Bingo Locomotion 2 (AMP dog-style prior + velocity-command PPO) task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Bingo-Locomotion2-AMP-Direct-v0",
    entry_point="bingo_rl.locomotion2_amp.bingo_dog_amp_env:BingoDogAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.locomotion2_amp.bingo_dog_amp_env_cfg:BingoDogAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

gym.register(
    id="Bingo-Locomotion2-AMP-Direct-Play-v0",
    entry_point="bingo_rl.locomotion2_amp.bingo_dog_amp_env:BingoDogAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.locomotion2_amp.bingo_dog_amp_env_cfg:BingoDogAmpEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
