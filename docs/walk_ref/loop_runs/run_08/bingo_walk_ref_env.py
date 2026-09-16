"""Reference-guided walk env: residual RL on a CYCLIC, speed-scaled Stage 4 walk.

    legs (12):        q_target = q_ref(phase) + residual_scale * action
    expressive (9):   q_target = q_ref(phase)   (fed from the reference, unchanged)

Differs from Stage 5 (bingo_rl.stage5), which tracks a single non-cyclic performance
clip frame-exactly, in exactly the ways the task requires:

  * `phase` is NOT wall-clock time in the clip. It is a per-env virtual clip-time
    that advances by `step_dt * (commanded_vx / NOMINAL_VX)`, wrapped modulo the
    clip duration -- so commanding a slower/faster forward speed literally slows
    down/speeds up the gait cycle, and the reference loops indefinitely instead of
    ending after one pass.
  * At low |commanded_vx| the leg reference blends from the cyclic walk pose to the
    robot's static default (standing) pose, so a zero command produces standing,
    not a frozen mid-stride pose.
  * The reward drops Stage 5's absolute root-position/orientation tracking term
    (ill-defined once the reference loops many times per episode -- the reference
    root position resets discontinuously at each wrap while the robot keeps
    walking forward) and replaces it with forward-velocity-command tracking, which
    is the actual objective here, plus a lightweight upright term from the root
    quaternion directly (not compared against the reference).
  * Reward adds small torque / action-rate / residual-magnitude penalties (absent
    from Stage 5) so the residual stays small and the gait stays smooth -- Stage 4's
    own PD-only tracking of this clip already torque-saturates several leg joints
    5-28% of frames (see bingo_walk_ref_env_cfg.py docstring); the residual and
    these penalties are what's supposed to close that gap without fighting the
    reference.

Everything else (21-DOF joint remap, expressive feed-forward, MotionLoader-based
continuous reference sampling, foot-tip/contact bookkeeping) is inherited from
bingo_rl.track_v4.BingoTrackV4Env / bingo_rl.track.BingoTrackEnv unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from bingo_rl.amp.bingo_amp_env import compute_obs
from bingo_rl.track_v4.bingo_track_v4_env import BingoTrackV4Env

from .bingo_walk_ref_env_cfg import NOMINAL_VX, BingoWalkRefEnvCfg

# The 12 controlled leg joints, in the reference's own column order. NOTE (same
# issue Stage 5 hit): BingoTrackEnv.__init__ derives ctrl_dof_names from the
# motion file's OWN dof_names, which for a v4 clip is all 21 joints (legs +
# head/tail + ears), not just the 12 the policy actually controls -- so it must
# be re-derived explicitly here, same as bingo_rl.stage5.bingo_stage5_env.
LEG_JOINT_NAMES = [
    "fl_SY_J", "fl_SP_J", "fl_knee",
    "fr_SY_J", "fr_SP_J", "fr_knee",
    "bl_SY_J", "bl_SP_J", "bl_knee",
    "br_SY_J", "br_SP_J", "br_knee",
]


class BingoWalkRefEnv(BingoTrackV4Env):
    cfg: BingoWalkRefEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.ctrl_dof_names = list(LEG_JOINT_NAMES)
        self.robot_ctrl_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in self.ctrl_dof_names],
            device=self.device, dtype=torch.long,
        )
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.ctrl_dof_names)
        lo = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 0]
        hi = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 1]
        self.action_offset = 0.5 * (hi + lo)
        self.action_scale = hi - lo
        self.obs_dof_indexes = torch.cat([self.robot_ctrl_indexes, self.expr_indexes])
        self._res_scale = torch.full(
            (len(self.ctrl_dof_names),), float(self.cfg.residual_scale), device=self.device
        )

        self._nominal_vx = float(NOMINAL_VX)
        self._stand_blend_vx = self._nominal_vx * float(self.cfg.stand_blend_vx_frac)
        self._fall_z = 0.5 * float(self._motion_loader.body_positions[0, self.motion_ref_body_index, 2])
        self._tilt_limit = float(getattr(self.cfg, "tilt_limit_deg", 70.0))

        # per-env virtual clip time (the "phase" state), NOT the wall-clock
        # start_times/episode_length_buf combination the parent classes use.
        self._clip_time = torch.zeros(self.num_envs, device=self.device)
        self._cmd_vx = torch.zeros(self.num_envs, device=self.device)
        self._cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._filtered_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_filtered_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_qdot = torch.zeros(self.num_envs, len(LEG_JOINT_NAMES), device=self.device)
        self._residual_ema_alpha = float(self.cfg.residual_ema_alpha)
        self._stand_pose = self.robot.data.default_joint_pos[:, self.robot_ctrl_indexes].clone()

        self._vx_lo, self._vx_hi = float(self.cfg.vx_range[0]), float(self.cfg.vx_range[1])
        self._stand_prob = float(self.cfg.stand_prob)
        self._resample_s = float(self.cfg.command_resample_s)

    # ------------------------------------------------------------------ command
    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        vx = self._vx_lo + (self._vx_hi - self._vx_lo) * torch.rand(n, device=self.device)
        stand = torch.rand(n, device=self.device) < self._stand_prob
        vx = torch.where(stand, torch.zeros_like(vx), vx)
        self._cmd_vx[env_ids] = vx
        self._cmd_timer[env_ids] = self._resample_s

    # ------------------------------------------------------------------ reference time
    def _current_times(self) -> np.ndarray:
        return self._clip_time.detach().cpu().numpy()

    # ------------------------------------------------------------------ RL loop
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = getattr(self, "actions", torch.zeros_like(actions))
        self.actions = actions.clone()

        # EMA low-pass on the residual actually applied to the robot (see
        # bingo_walk_ref_env_cfg.RESIDUAL_EMA_ALPHA docstring): the shaking
        # diagnostic measured the raw policy action jumping up to 77% of its own
        # authority in a single control step. `self.actions` (raw) is kept
        # unfiltered for logging/ONNX-parity purposes; `_filtered_action` is what
        # `_apply_action` and the smoothness reward terms actually use.
        self._prev_filtered_action = self._filtered_action.clone()
        a = self._residual_ema_alpha
        self._filtered_action = a * self.actions + (1.0 - a) * self._prev_filtered_action

        self._cmd_timer -= self.step_dt
        due = (self._cmd_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() > 0:
            self._resample_command(due)

        speed_ratio = 1.4 * self._cmd_vx / self._nominal_vx
        self._clip_time = torch.remainder(
            self._clip_time + self.step_dt * speed_ratio, self.motion_duration
        )

        ref_dof, *_ = self._sample_ref(self._current_times())
        w = torch.clamp(torch.abs(self._cmd_vx) / self._stand_blend_vx, 0.0, 1.0).unsqueeze(-1)
        self._ff_leg_q = w * ref_dof + (1.0 - w) * self._stand_pose

    def _apply_action(self):
        """Overrides BingoTrackEnv/BingoTrackV4Env's _apply_action to drive the
        EMA-FILTERED residual (self._filtered_action), not the raw policy output --
        see _pre_physics_step."""
        target = self._ff_leg_q + self._res_scale * self._filtered_action
        self.robot.set_joint_position_target(target, joint_ids=self.robot_ctrl_indexes)
        expr = self._expr_targets(self._current_times())
        self.robot.set_joint_position_target(expr, joint_ids=self.expr_indexes)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        foot_tips = knee_pos + quat_apply(knee_quat, offset)
        proprio = compute_obs(
            self.robot.data.joint_pos[:, self.obs_dof_indexes],
            self.robot.data.joint_vel[:, self.obs_dof_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            foot_tips,
        )
        phase = self._clip_time / self.motion_duration
        phase_enc = torch.stack([torch.sin(2 * np.pi * phase), torch.cos(2 * np.pi * phase)], dim=-1)
        cmd_enc = (self._cmd_vx / self._nominal_vx).unsqueeze(-1)
        return {"policy": torch.cat([proprio, phase_enc, cmd_enc], dim=-1)}

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        times = self._current_times()
        ref_dof, ref_dofv, _ref_root_pos, _ref_root_quat, _ref_lin, _ref_ang, ref_key_local = \
            self._sample_ref(times)

        cur_dof = self.robot.data.joint_pos[:, self.robot_ctrl_indexes]
        cur_dofv = self.robot.data.joint_vel[:, self.robot_ctrl_indexes]
        tips_w, cur_key_local = self._foot_tips_local()

        pose_sq = torch.sum((cur_dof - ref_dof) ** 2, dim=1)
        vel_sq = torch.sum((cur_dofv - ref_dofv) ** 2, dim=1)
        ee_sq = torch.sum(torch.sum((cur_key_local - ref_key_local) ** 2, dim=-1), dim=1)

        if self._ref_contacts is not None:
            fidx = np.clip(np.round(times * self._ref_fps).astype(int), 0, self._n_ref_frames - 1)
            rc = self._ref_contacts[fidx]
            act_c = (tips_w[:, :, 2] < 0.03).float()
            contact_match = (act_c == rc).float().mean(dim=1)
        else:
            contact_match = torch.zeros(self.num_envs, device=self.device)

        # forward-velocity-command tracking, body frame (the actual objective here)
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_lin_b = quat_rotate_inverse(root_quat, root_lin_w)
        vx_err_sq = (root_lin_b[:, 0] - self._cmd_vx) ** 2

        # upright stability from the root quaternion directly (NOT vs. the looping
        # reference's own orientation -- see module docstring)
        r22 = 1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
        upright = torch.clamp(r22, 0.0, 1.0)

        torque = self.robot.data.applied_torque[:, self.robot_ctrl_indexes]
        torque_pen = torch.mean((torque / 3.0) ** 2, dim=1)  # 3.0 N*m = leg effort limit
        # smoothness penalties operate on the FILTERED residual (what's actually
        # applied), not the raw network output -- otherwise PPO could learn a
        # jittery raw action "for free" since the EMA filter absorbs it before it
        # reaches the robot. [MEASURED, this session: shaking diagnostic found the
        # raw residual jumping up to 0.23 rad/step and joint-accel RMS 67-133 rad/s^2,
        # ~2x higher at the loop seam -- see bingo_walk_ref_env_cfg.py docstring.]
        act_rate_pen = torch.sum((self._filtered_action - self._prev_filtered_action) ** 2, dim=1)
        residual_pen = torch.sum(self._filtered_action ** 2, dim=1)
        qdot_accel = (cur_dofv - self._prev_qdot) / self.step_dt
        accel_pen = torch.mean((qdot_accel / 50.0) ** 2, dim=1)  # 50 rad/s^2 ~ typical non-seam RMS
        self._prev_qdot = cur_dofv.detach().clone()

        r = (
            0.20 * torch.exp(-2.0 * pose_sq)
            + 0.05 * torch.exp(-0.1 * vel_sq)
            + 0.15 * torch.exp(-40.0 * ee_sq)
            + 0.10 * contact_match
            + 0.20 * torch.exp(-8.0 * vx_err_sq)
            + 0.10 * upright
            - 0.04 * torque_pen
            - 0.02 * act_rate_pen
            - 0.01 * residual_pen
            - 0.15 * accel_pen
        )
        return r

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.early_termination:
            return torch.zeros_like(time_out), time_out
        q = self.robot.data.root_quat_w
        r22 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        tilt = torch.rad2deg(torch.arccos(torch.clamp(r22, -1.0, 1.0)))
        root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fell = (root_z < self._fall_z) | (tilt > self._tilt_limit)
        return fell, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        """Bypasses BingoTrackEnv/BingoTrackV4Env's start_times-based RSI (this env
        has no start_times -- see _current_times) and writes legs + expressive DOF
        from the same sampled clip-time in one pass."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        n = len(env_ids)
        if getattr(self.cfg, "random_start_frame", True):
            t = torch.rand(n, device=self.device) * self.motion_duration
        else:
            t = torch.zeros(n, device=self.device)
        self._clip_time[env_ids] = t
        self._resample_command(env_ids)
        self._prev_actions[env_ids] = 0.0
        self._filtered_action[env_ids] = 0.0
        self._prev_filtered_action[env_ids] = 0.0
        self._prev_qdot[env_ids] = 0.0

        times_np = t.detach().cpu().numpy()
        ref_dof, ref_dofv, root_pos, root_quat, root_lin, root_ang, _ = self._sample_ref(times_np)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_pos + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.02  # tiny lift to avoid ground penetration on reset
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_lin
        root_state[:, 10:13] = root_ang

        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.robot_ctrl_indexes] = ref_dof
        dof_vel[:, self.robot_ctrl_indexes] = ref_dofv
        dof_pos[:, self.expr_indexes] = self._expr_targets(times_np)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
