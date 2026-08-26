"""Bingo DeepMimic-style single-clip tracking env.

Trains a policy to REPRODUCE a specific authored performance (e.g. deadpan) on the
robot under physics, using a phase-based imitation reward (pose + velocity + end-
effector + root). Reuses the AMP env's robot cfg, MotionLoader, 12-DOF action
mapping, and proprio obs builder; the only differences are:
  - obs adds a 2-d phase encoding (sin, cos) so the policy knows where in the clip it is
  - reward is DeepMimic imitation against the reference at the current phase (no discriminator)
  - reference-state initialization (RSI) + pose-deviation early termination
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from isaaclab_tasks.direct.humanoid_amp.motions.motion_loader import MotionLoader

from bingo_rl.amp.bingo_amp_env import compute_obs
from .bingo_track_env_cfg import BingoTrackEnvCfg

KEY_BODY_NAMES = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]
SHANK_LEN = 0.12  # shank-tip offset below the knee hinge (matches the AMP env)


class BingoTrackEnv(DirectRLEnv):
    cfg: BingoTrackEnvCfg

    def __init__(self, cfg: BingoTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
        # NOTE: DirectRLEnv already exposes read-only `self.step_dt` == sim.dt * decimation; use it.
        self.motion_duration = float(self._motion_loader.duration)

        # reference per-foot contact flags (not carried by MotionLoader) for the contact reward
        _raw = np.load(self.cfg.motion_file, allow_pickle=True)
        self._ref_contacts = (
            torch.tensor(_raw["contacts"], device=self.device, dtype=torch.float32)
            if "contacts" in _raw.files else None
        )
        self._ref_fps = float(_raw["fps"]) if "fps" in _raw.files else 120.0
        self._n_ref_frames = self._ref_contacts.shape[0] if self._ref_contacts is not None else 0

        # 12 controlled leg joints, in the motion's canonical order
        self.ctrl_dof_names = list(self._motion_loader.dof_names)
        self.robot_ctrl_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in self.ctrl_dof_names],
            device=self.device, dtype=torch.long,
        )
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.ctrl_dof_names)

        dof_lower = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = dof_upper - dof_lower
        sy_idx = [i for i, n in enumerate(self.ctrl_dof_names) if n.endswith("_SY_J")]
        self.action_scale[sy_idx] *= 0.3

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(n) for n in KEY_BODY_NAMES]
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(KEY_BODY_NAMES)

        self.shank_offset = torch.tensor([[0.0, 0.0, -SHANK_LEN]] * len(KEY_BODY_NAMES), device=self.device)

        # per-joint residual authority (ResMimic residual = ff + res_scale * action).
        # Default: uniform cfg.residual_scale. Optional per-type override lets us restrict knee
        # residual (keep knees on the reference) while giving SP/SY more balance authority.
        rs = float(getattr(self.cfg, "residual_scale", 0.0))
        self._res_scale = torch.full((len(self.ctrl_dof_names),), rs, device=self.device)
        byt = getattr(self.cfg, "residual_scale_by_type", None)
        if byt:
            for i, n in enumerate(self.ctrl_dof_names):
                if "SY_J" in n and "SY" in byt: self._res_scale[i] = byt["SY"]
                elif "SP_J" in n and "SP" in byt: self._res_scale[i] = byt["SP"]
                elif "knee" in n and "knee" in byt: self._res_scale[i] = byt["knee"]

        # per-env motion start time (RSI); current phase time = start + steps * dt
        self.start_times = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ helpers
    def _current_times(self) -> np.ndarray:
        t = self.start_times + self.episode_length_buf.float() * self.step_dt
        t = torch.remainder(t, self.motion_duration)
        return t.detach().cpu().numpy()

    def _sample_ref(self, times: np.ndarray):
        """Return reference (dof_pos, dof_vel, root_pos, root_quat, root_lin, root_ang,
        key_pos_local) all on device, sliced to the controlled dofs / key bodies."""
        dp, dv, bp, br, blv, bav = self._motion_loader.sample(num_samples=len(times), times=times)
        ref_dof = dp[:, self.motion_dof_indexes]
        ref_dofv = dv[:, self.motion_dof_indexes]
        root_pos = bp[:, self.motion_ref_body_index]
        root_quat = br[:, self.motion_ref_body_index]
        root_lin = blv[:, self.motion_ref_body_index]
        root_ang = bav[:, self.motion_ref_body_index]
        # NOTE: retarget.py stores body_positions[*_knee] = the SHANK TIP (tips_w), already the
        # contact point — NOT the knee hinge (verified: ref knee-body z ~0.04 m, near ground).
        # So use it directly; the actual side (_foot_tips_local) turns the sim knee HINGE into the
        # sim shank tip, so both are shank tips. (Adding the shank offset here again was a
        # ~0.12 m double-offset bug that inflated foot-tip error and corrupted the ee reward.)
        key_pos = bp[:, self.motion_key_body_indexes]   # (N,4,3) shank tip, world
        key_local = quat_rotate_inverse(
            root_quat.unsqueeze(1).expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
            (key_pos - root_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(-1, len(KEY_BODY_NAMES), 3)
        return ref_dof, ref_dofv, root_pos, root_quat, root_lin, root_ang, key_local

    def _foot_tips_local(self):
        """Robot shank-tip positions in the root frame (N,4,3)."""
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        tips_w = knee_pos + quat_apply(knee_quat, offset)
        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        local = quat_rotate_inverse(
            root_quat.unsqueeze(1).expand(-1, len(KEY_BODY_NAMES), -1).reshape(-1, 4),
            (tips_w - root_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(-1, len(KEY_BODY_NAMES), 3)
        return tips_w, local

    # ------------------------------------------------------------------ RL loop
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # ResMimic-style residual control: feedforward the (joint-trackable) reference and let
        # the policy learn only a small correction. Sampled once per control step here.
        if getattr(self.cfg, "residual_scale", 0.0) > 0.0:
            ref_dof, *_ = self._sample_ref(self._current_times())
            self._ff_leg_q = ref_dof  # (N,12) reference leg angles at this phase

    def _apply_action(self):
        if getattr(self.cfg, "residual_scale", 0.0) > 0.0:
            target = self._ff_leg_q + self._res_scale * self.actions
        else:
            target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self.robot_ctrl_indexes)

    def _get_observations(self) -> dict:
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        foot_tips = knee_pos + quat_apply(knee_quat, offset)
        proprio = compute_obs(
            self.robot.data.joint_pos[:, self.robot_ctrl_indexes],
            self.robot.data.joint_vel[:, self.robot_ctrl_indexes],
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

    def _get_rewards(self) -> torch.Tensor:
        times = self._current_times()
        ref_dof, ref_dofv, ref_root_pos, ref_root_quat, ref_lin, ref_ang, ref_key_local = self._sample_ref(times)

        cur_dof = self.robot.data.joint_pos[:, self.robot_ctrl_indexes]
        cur_dofv = self.robot.data.joint_vel[:, self.robot_ctrl_indexes]
        tips_w, cur_key_local = self._foot_tips_local()
        cur_root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        # actual root in the clip-local frame (strip the per-env grid offset)
        cur_root_local = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins

        # DeepMimic imitation reward, spec B.4.1 (SUM over dims, peaky -> real tracking gradient)
        pose_sq = torch.sum((cur_dof - ref_dof) ** 2, dim=1)
        vel_sq = torch.sum((cur_dofv - ref_dofv) ** 2, dim=1)
        ee_sq = torch.sum(torch.sum((cur_key_local - ref_key_local) ** 2, dim=-1), dim=1)
        root_pos_sq = torch.sum((cur_root_local - ref_root_pos) ** 2, dim=1)
        rot_err = 1.0 - torch.sum(cur_root_quat * ref_root_quat, dim=-1) ** 2

        # contact match (fraction of feet whose contact state matches the reference)
        if self._ref_contacts is not None:
            fidx = np.clip(np.round(times * self._ref_fps).astype(int), 0, self._n_ref_frames - 1)
            rc = self._ref_contacts[fidx]
            act_c = (tips_w[:, :, 2] < 0.03).float()
            contact_match = (act_c == rc).float().mean(dim=1)
        else:
            contact_match = torch.zeros(self.num_envs, device=self.device)

        r = (
            0.50 * torch.exp(-2.0 * pose_sq)
            + 0.05 * torch.exp(-0.1 * vel_sq)
            + 0.20 * torch.exp(-40.0 * ee_sq)
            + 0.15 * torch.exp(-20.0 * root_pos_sq - 10.0 * rot_err)
            + 0.10 * contact_match
        )
        # cache for termination + logging
        self._root_track_dist = torch.sqrt(root_pos_sq).detach()
        return r

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            fell = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            root_dist = getattr(self, "_root_track_dist", torch.zeros(self.num_envs, device=self.device))
            drifted = root_dist > self.cfg.root_track_max  # spec B.4.1: terminate on tracking error
            died = fell | drifted
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # RSI: random start phase within the clip
        start = self._motion_loader.sample_times(n)
        self.start_times[env_ids] = torch.tensor(start, device=self.device, dtype=torch.float32)

        ref_dof, ref_dofv, root_pos, root_quat, root_lin, root_ang, _ = self._sample_ref(start)

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

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
