"""Bingo expression tracker env: legs learned by the policy, head+tail driven feed-forward
from the reference clip. Subclasses BingoTrackEnv so the spec-B.4.1 reward, termination, and
RSI reset are inherited unchanged; only the observation (17-DOF proprio) and the action
application (add head/tail targets) are extended.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

from bingo_rl.amp.bingo_amp_env import compute_obs
from bingo_rl.track.bingo_track_env import BingoTrackEnv, KEY_BODY_NAMES
from .bingo_track_expr_env_cfg import BingoTrackExprEnvCfg

# 5 head/tail joints, in canonical order (indices 12-16 of the 21-DOF schema).
# rev_3 has these; ears (17-20) do not exist on rev_3 and are omitted.
HT_JOINT_NAMES = ["head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw"]


class BingoTrackExprEnv(BingoTrackEnv):
    cfg: BingoTrackExprEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.ht_joint_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in HT_JOINT_NAMES],
            device=self.device, dtype=torch.long,
        )
        # 17-DOF observation order: 12 legs (ctrl order) then the 5 head/tail joints
        self.obs_dof_indexes = torch.cat([self.robot_ctrl_indexes, self.ht_joint_indexes])

        # head/tail reference (not carried by MotionLoader) — sampled by frame index
        _raw = np.load(self.cfg.motion_file, allow_pickle=True)
        assert "head_tail_positions" in _raw.files, "clip has no head_tail_positions"
        self.ht_ref = torch.tensor(_raw["head_tail_positions"], device=self.device, dtype=torch.float32)
        self.n_ht = self.ht_ref.shape[0]  # (n_frames, 5)

    def _ht_targets(self, times_np: np.ndarray) -> torch.Tensor:
        fidx = np.clip(np.round(times_np * self._ref_fps).astype(int), 0, self.n_ht - 1)
        return self.ht_ref[fidx]  # (N,5)

    def _apply_action(self):
        super()._apply_action()  # leg position targets from the policy
        ht = self._ht_targets(self._current_times())
        self.robot.set_joint_position_target(ht, joint_ids=self.ht_joint_indexes)

    def _get_observations(self) -> dict:
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        foot_tips = knee_pos + quat_apply(knee_quat, offset)
        proprio = compute_obs(
            self.robot.data.joint_pos[:, self.obs_dof_indexes],   # 17
            self.robot.data.joint_vel[:, self.obs_dof_indexes],   # 17
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            foot_tips,
        )
        phase = torch.remainder(
            self.start_times + self.episode_length_buf.float() * self.step_dt, self.motion_duration
        ) / self.motion_duration
        phase_enc = torch.stack([torch.sin(2 * np.pi * phase), torch.cos(2 * np.pi * phase)], dim=-1)
        return {"policy": torch.cat([proprio, phase_enc], dim=-1)}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)  # sets root + leg state to the reference; picks start_times
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # seed head/tail joints to the reference at each env's start phase
        start_np = self.start_times[env_ids].detach().cpu().numpy()
        ht = self._ht_targets(start_np)
        self.robot.write_joint_state_to_sim(
            ht, torch.zeros_like(ht), self.ht_joint_indexes, env_ids
        )
