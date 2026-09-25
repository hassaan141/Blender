"""Lateral (left/right sidestep) locomotion: one new signed command, `cmd_vy`,
added on top of the unmodified BingoWalkRefEnv -- no observation-space change.

See docs/experiment_loop/lateral/AUTONOMOUS_LATERAL_PROTOCOL.md for the full
research and rationale. Summary: BingoWalkRefEnv's reward never uses the
reference clip's absolute root trajectory (only joint pose / foot-tip-local /
contacts, all root-relative), so a reference authored to encode a genuine
sideways shuffle IN PLACE, combined with a new reward term pulling body-frame
lateral velocity toward `cmd_vy`, produces real sideways translation without
touching the 70-dim observation vector or the champion checkpoint's shape.

Body-frame convention (stage2/v4_kinematics.py, verified): +X forward, +Y left,
+Z up. So `root_lin_b[:,1] > 0` is leftward, `< 0` is rightward. `cmd_vx` is
pinned to 0 so the EXISTING forward-tracking reward term becomes "suppress
forward drift", which is exactly wanted for a pure sidestep.

Does not import or modify anything under rl/bingo_rl/bingo_rl/ (protected).
"""
from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse

from bingo_rl.walk_ref.bingo_walk_ref_env import BingoWalkRefEnv
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import BingoWalkRefEnvCfg_C, BingoWalkRefPlayEnvCfg

ROOT = Path(__file__).resolve().parents[4]
REFERENCE = os.environ.get(
    "BINGO_LATERAL_REFERENCE", str(ROOT / "docs/experiment_loop/lateral/right/reference_seed.npz")
)

# lateral-tracking reward: same functional shape/weight as the base env's own
# forward-tracking term (0.20 * exp(-8.0 * err^2)), applied to the Y axis instead.
LATERAL_TRACK_WEIGHT = 0.20
LATERAL_TRACK_SHARPNESS = 8.0


class LateralWalkEnv(BingoWalkRefEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cmd_vy = torch.full((self.num_envs,), float(self.cfg.cmd_vy), device=self.device)

    def _get_rewards(self):
        reward = super()._get_rewards()  # includes the now-zero-target forward-drift-suppression term
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_lin_b = quat_rotate_inverse(root_quat, root_lin_w)
        vy_err_sq = (root_lin_b[:, 1] - self._cmd_vy) ** 2
        return reward + LATERAL_TRACK_WEIGHT * torch.exp(-LATERAL_TRACK_SHARPNESS * vy_err_sq)


@configclass
class LateralTrainCfg(BingoWalkRefEnvCfg_C):
    """Fixed-magnitude lateral command; no forward, no standing, no resampling
    variety -- mirrors backward's deliberate simplicity for a first campaign."""
    motion_file: str = REFERENCE
    ear_file: str = REFERENCE
    vx_range = (0.0, 0.0)
    stand_prob = 0.0
    command_resample_s = 1000.0
    cmd_vy: float = float(os.environ.get("BINGO_LATERAL_CMD_VY", "-0.15"))  # default: rightward


@configclass
class LateralPlayCfg(BingoWalkRefPlayEnvCfg):
    motion_file: str = REFERENCE
    ear_file: str = REFERENCE
    vx_range = (0.0, 0.0)
    stand_prob = 0.0
    cmd_vy: float = float(os.environ.get("BINGO_LATERAL_CMD_VY", "-0.15"))


for _name, _cfg in [("Train", LateralTrainCfg), ("Play", LateralPlayCfg)]:
    gym.register(
        id=f"Bingo-Lateral-{_name}-v0",
        entry_point=LateralWalkEnv,  # direct callable: avoids dotted-path ambiguity
        # (rl/bingo_rl/experiment_loop/ is a sibling of the bingo_rl package, not
        # a subpackage of it -- "bingo_rl.experiment_loop...." would not resolve).
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg,
            "skrl_cfg_entry_point": "bingo_rl.track.agents:skrl_ppo_cfg.yaml",
        },
    )
