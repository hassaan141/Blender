"""Bingo DeepMimic-style single-clip tracking task registration.

Reproduces a SPECIFIC authored performance (e.g. deadpan) on the robot in sim, as
opposed to AMP which matches a style distribution. Reuses the same rev_3 USD, the
same .npz MotionLoader, and the same 12-DOF action mapping as the AMP env; only the
observation (adds phase) and the reward (phase-based imitation, not a discriminator)
differ.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Bingo-Track-Deadpan-Direct-v0",
    entry_point="bingo_rl.track.bingo_track_env:BingoTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.track.bingo_track_env_cfg:BingoTrackEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-TrackRes-Deadpan-Direct-v0",
    entry_point="bingo_rl.track.bingo_track_env:BingoTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.track.bingo_track_env_cfg:BingoTrackResidualEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bingo-TrackResKneeLock-Deadpan-Direct-v0",
    entry_point="bingo_rl.track.bingo_track_env:BingoTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.track.bingo_track_env_cfg:BingoTrackResKneeLockEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
