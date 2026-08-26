"""Config for the v4 21-DOF expression tracker."""
from __future__ import annotations

import os

from isaaclab.utils import configclass

from bingo_rl.track.bingo_track_env_cfg import BingoTrackEnvCfg
from bingo_rl.bingo_v4 import BINGO_V4_CFG


@configclass
class BingoTrackV4EnvCfg(BingoTrackEnvCfg):
    # obs = 21 dof_pos + 21 dof_vel + root_h(1) + quat(6) + root_lin(3) + root_ang(3)
    #       + 4 feet*3(12) + phase(2) = 69
    observation_space = 69
    action_space = 12  # policy controls the 12 legs; 9 expressive DOF fed from the reference
    ear_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions", "deadpan_ears.npz")

    robot = BINGO_V4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        # v4 default pose already set in BINGO_V4_CFG; keep it
        pass
