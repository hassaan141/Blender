"""Bingo Locomotion 2 AMP env: velocity-command PPO + dog-style AMP discriminator reward.

See bingo_dog_amp_env_cfg.py's docstring for the strategy pivot rationale and
docs/locomotion2/amp/extract_dog_amp_features.py for the expert-side half of the
61-dim morphology-tolerant style feature schema this env computes on the policy side.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform

from .bingo_dog_amp_env_cfg import (
    ACTION_SCALE, BINGO_LEG_SCALE, DOF_ORDER, HIP_BODIES, KNEE_BODIES, LEGS, SHANK_LEN,
    BingoDogAmpEnvCfg,
)


class BingoDogAmpEnv(DirectRLEnv):
    cfg: BingoDogAmpEnvCfg

    def __init__(self, cfg: BingoDogAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt_ctrl = self.cfg.sim.dt * self.cfg.decimation

        self.ctrl_dof_names = list(DOF_ORDER)
        self.robot_ctrl_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in self.ctrl_dof_names],
            device=self.device, dtype=torch.long,
        )
        scale_list = []
        for n in self.ctrl_dof_names:
            key = ".*_SY_J" if n.endswith("_SY_J") else (".*_SP_J" if n.endswith("_SP_J") else ".*_knee")
            scale_list.append(ACTION_SCALE[key])
        self.action_scale = torch.tensor(scale_list, device=self.device)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.hip_indexes = [self.robot.data.body_names.index(n) for n in HIP_BODIES]
        self.knee_indexes = [self.robot.data.body_names.index(n) for n in KNEE_BODIES]
        self.contact_foot_idx = [self.contact_sensor.body_names.index(n) for n in KNEE_BODIES]
        self.shank_offset = torch.tensor([[0.0, 0.0, -SHANK_LEN]] * 4, device=self.device)

        # ---- expert transition buffer (feat_t, feat_t+1), see extract_dog_amp_features.py
        expert = np.load(self.cfg.expert_npz)
        self.expert_feat_t = torch.tensor(expert["feat_t"], device=self.device, dtype=torch.float32)
        self.expert_feat_tp1 = torch.tensor(expert["feat_tp1"], device=self.device, dtype=torch.float32)
        assert self.expert_feat_t.shape[-1] == self.cfg.amp_observation_space, (
            f"expert feature dim {self.expert_feat_t.shape[-1]} != cfg.amp_observation_space "
            f"{self.cfg.amp_observation_space}"
        )

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self.commands = torch.zeros((self.num_envs, 3), device=self.device)  # vx, vy, yaw_rate
        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_dof_vel = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_paw_rel = torch.zeros((self.num_envs, 4, 3), device=self.device)

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        default = self.robot.data.default_joint_pos[:, self.robot_ctrl_indexes]
        target = default + self.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self.robot_ctrl_indexes)

    def _leg_world_positions(self):
        """(hip, knee, paw) world positions, each (N, 4, 3), leg order fl,fr,bl,br."""
        hip_pos = self.robot.data.body_pos_w[:, self.hip_indexes]
        knee_pos = self.robot.data.body_pos_w[:, self.knee_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.knee_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        paw_pos = knee_pos + quat_apply(knee_quat, offset)
        return hip_pos, knee_pos, paw_pos

    def _compute_style_features(self, update_paw_vel_buffer: bool) -> torch.Tensor:
        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        N = root_pos.shape[0]

        root_vel_local = self.robot.data.root_lin_vel_b / BINGO_LEG_SCALE
        ang_vel_local = self.robot.data.root_ang_vel_b
        gravity_local = self.robot.data.projected_gravity_b

        hip_pos, knee_pos, paw_pos = self._leg_world_positions()
        root_quat_4 = root_quat.unsqueeze(1).expand(-1, 4, -1)

        d1_world = knee_pos - hip_pos
        d1_local = quat_apply_inverse(root_quat_4, d1_world)
        d1_local = d1_local / (d1_local.norm(dim=-1, keepdim=True) + 1e-9)
        d2_world = paw_pos - knee_pos
        d2_local = quat_apply_inverse(root_quat_4, d2_world)
        d2_local = d2_local / (d2_local.norm(dim=-1, keepdim=True) + 1e-9)
        seg_dirs = torch.stack([d1_local, d2_local], dim=2)  # (N,4,2,3)

        paw_rel_world = paw_pos - root_pos.unsqueeze(1)
        paw_rel_local = quat_apply_inverse(root_quat_4, paw_rel_world) / BINGO_LEG_SCALE  # (N,4,3)

        paw_vel_local = (paw_rel_local - self.prev_paw_rel) / self.dt_ctrl
        if update_paw_vel_buffer:
            self.prev_paw_rel = paw_rel_local.detach()

        forces = self.contact_sensor.data.net_forces_w[:, self.contact_foot_idx]  # (N,4,3)
        contacts = (forces.norm(dim=-1) > 0.5).float()

        feat = torch.cat([
            root_vel_local, ang_vel_local, gravity_local,
            seg_dirs.reshape(N, 24), paw_rel_local.reshape(N, 12), paw_vel_local.reshape(N, 12), contacts,
        ], dim=-1)
        return feat

    def _get_observations(self) -> dict:
        amp_obs = self._compute_style_features(update_paw_vel_buffer=True)
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        policy_obs = torch.cat([amp_obs, self.commands], dim=-1)
        self.prev_actions = self.actions.clone()
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        vx = self.robot.data.root_lin_vel_b[:, 0]
        vy = self.robot.data.root_lin_vel_b[:, 1]
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        cmd_vx, cmd_vy, cmd_yaw = self.commands[:, 0], self.commands[:, 1], self.commands[:, 2]

        vx_track = torch.exp(-((vx - cmd_vx) ** 2) / self.cfg.tracking_sigma_vx)
        vy_track = torch.exp(-((vy - cmd_vy) ** 2) / self.cfg.tracking_sigma_vy)
        yaw_track = torch.exp(-((yaw_rate - cmd_yaw) ** 2) / self.cfg.tracking_sigma_yaw)
        task_reward = vx_track + 0.3 * vy_track + 0.3 * yaw_track

        gravity_local = self.robot.data.projected_gravity_b
        upright = torch.exp(-5.0 * torch.sum(gravity_local[:, :2] ** 2, dim=-1))

        action_rate_pen = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)

        dof_vel = self.robot.data.joint_vel[:, self.robot_ctrl_indexes]
        joint_acc = (dof_vel - self.prev_dof_vel) / self.dt_ctrl
        joint_acc_pen = torch.sum(joint_acc ** 2, dim=-1)
        self.prev_dof_vel = dof_vel.clone()

        torque = self.robot.data.applied_torque[:, self.robot_ctrl_indexes]
        torque_pen = torch.sum(torque ** 2, dim=-1)

        air = self.contact_sensor.data.current_air_time[:, self.contact_foot_idx]
        contact_t = self.contact_sensor.data.current_contact_time[:, self.contact_foot_idx]
        gait_bound_pen = (
            torch.sum(torch.clamp(air - self.cfg.max_air_time, min=0.0), dim=-1)
            + torch.sum(torch.clamp(contact_t - self.cfg.max_contact_time, min=0.0), dim=-1)
        )

        reward = (
            task_reward
            + self.cfg.upright_weight * upright
            + self.cfg.action_rate_weight * action_rate_pen
            + self.cfg.joint_acc_weight * joint_acc_pen
            + self.cfg.torque_weight * torque_pen
            + self.cfg.gait_bound_weight * gait_bound_pen
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            low = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            tilt_cos = -self.robot.data.projected_gravity_b[:, 2]
            bad_orient = tilt_cos < np.cos(self.cfg.bad_orientation_limit)
            died = low | bad_orient
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        vx_lo, vx_hi = self.cfg.cmd_vx_range
        vy_lo, vy_hi = self.cfg.cmd_vy_range
        yaw_lo, yaw_hi = self.cfg.cmd_yaw_range
        fx = os.environ.get("BINGO_CMD_VX")
        fy = os.environ.get("BINGO_CMD_VY")
        fyaw = os.environ.get("BINGO_CMD_YAW")
        if fx is not None:
            self.commands[env_ids, 0] = float(fx)
        else:
            stand = torch.rand(n, device=self.device) < self.cfg.stand_prob
            walk_vx = sample_uniform(vx_lo, vx_hi, (n,), self.device)
            self.commands[env_ids, 0] = torch.where(stand, torch.zeros_like(walk_vx), walk_vx)
        self.commands[env_ids, 1] = float(fy) if fy is not None else sample_uniform(vy_lo, vy_hi, (n,), self.device)
        self.commands[env_ids, 2] = float(fyaw) if fyaw is not None else sample_uniform(yaw_lo, yaw_hi, (n,), self.device)

        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.prev_actions[env_ids] = 0.0
        self.prev_dof_vel[env_ids] = joint_vel[:, self.robot_ctrl_indexes]

        # settle the paw-velocity finite-difference buffer BEFORE computing features, so a
        # freshly-reset env reads paw_vel ~ 0 instead of a spike against the stale
        # end-of-previous-episode value (requires an up-to-date sim/data read, which
        # write_*_to_sim above provides).
        self.prev_paw_rel[env_ids] = self._leg_paw_rel_local()[env_ids]
        feat = self._compute_style_features(update_paw_vel_buffer=False)
        self.amp_observation_buffer[env_ids] = feat[env_ids].unsqueeze(1).repeat(1, self.cfg.num_amp_observations, 1)

    def _leg_paw_rel_local(self) -> torch.Tensor:
        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        _, _, paw_pos = self._leg_world_positions()
        root_quat_4 = root_quat.unsqueeze(1).expand(-1, 4, -1)
        paw_rel_world = paw_pos - root_pos.unsqueeze(1)
        return quat_apply_inverse(root_quat_4, paw_rel_world) / BINGO_LEG_SCALE

    def _reset_strategy_default(self, env_ids: torch.Tensor):
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        n = len(env_ids)
        if not self.cfg.deterministic_spawn:
            root_state[:, 0] += sample_uniform(-0.1, 0.1, (n,), self.device)
            root_state[:, 1] += sample_uniform(-0.1, 0.1, (n,), self.device)
            yaw = sample_uniform(-np.pi, np.pi, (n,), self.device)
            half = yaw / 2
            root_state[:, 3] = torch.cos(half)
            root_state[:, 4] = 0.0
            root_state[:, 5] = 0.0
            root_state[:, 6] = torch.sin(half)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Called by skrl's AMP agent (Runner wires `env.collect_reference_motions`
        directly, single positional arg) to sample expert (feat_t, feat_t+1) pairs."""
        idx = torch.randint(0, self.expert_feat_t.shape[0], (num_samples,), device=self.device)
        return torch.cat([self.expert_feat_t[idx], self.expert_feat_tp1[idx]], dim=-1)
