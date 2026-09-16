"""Bingo expression tracker: legs learned (RL) + head/tail driven from the reference.

On the rev_3 USD the robot has 17 joints (12 legs + head_pitch/yaw/roll + tail_pitch/yaw);
there are NO ear joints (those exist only in the v4 URDF, for which no USD exists yet). So this
task drives the **5 head/tail** expressive DOF feed-forward from the clip while the policy keeps
learning the 12 legs. Reuses the tracker's spec-B.4.1 reward/termination unchanged.
"""

import gymnasium as gym

gym.register(
    id="Bingo-TrackExpr-Deadpan-Direct-v0",
    entry_point="bingo_rl.track_expr.bingo_track_expr_env:BingoTrackExprEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bingo_rl.track_expr.bingo_track_expr_env_cfg:BingoTrackExprEnvCfg",
        "skrl_cfg_entry_point": "bingo_rl.track.agents:skrl_ppo_cfg.yaml",
    },
)
