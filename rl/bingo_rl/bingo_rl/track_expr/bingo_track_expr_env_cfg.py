"""Config for the Bingo expression tracker (legs RL + head/tail feed-forward)."""

from __future__ import annotations

from isaaclab.utils import configclass

from bingo_rl.track.bingo_track_env_cfg import BingoTrackEnvCfg


@configclass
class BingoTrackExprEnvCfg(BingoTrackEnvCfg):
    # obs = 17 dof_pos + 17 dof_vel + root_h(1) + quat_tan_norm(6) + root_lin(3)
    #       + root_ang(3) + 4 feet*3(12) + phase(2) = 61
    observation_space = 61
    action_space = 12  # policy still controls only the 12 legs; head/tail are fed from the ref
