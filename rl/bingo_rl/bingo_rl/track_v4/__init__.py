"""Bingo v4 (21-DOF) expression tracker: legs RL + head/tail + EARS from the reference.

Uses the converted v4 USD (21 joints incl. 4 ears). The 9 expressive DOF are driven
feed-forward from the reference (head/tail from the clip's head_tail_positions; ears from a
baked ear signal); the policy learns the 12 legs.
"""
import gymnasium as gym

gym.register(
    id="Bingo-TrackV4-Deadpan-Direct-v0",
    entry_point="bingo_rl.track_v4.bingo_track_v4_env:BingoTrackV4Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.track_v4.bingo_track_v4_env_cfg:BingoTrackV4EnvCfg",
        "skrl_cfg_entry_point": "bingo_rl.track.agents:skrl_ppo_cfg.yaml",
    },
)
