"""Bingo AMP env — adapted from IsaacLab's HumanoidAmpEnv for a 12-DOF quadruped.

Differences from humanoid: key bodies are the 4 feet; reference/root body is "origin"
(configurable) rather than a hardcoded "torso". Obs/AMP-obs construction is identical
in structure (compute_obs), just sized for 12 DOF + 4 key bodies -> 49.
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
from isaaclab.utils.math import quat_apply

# reuse the humanoid AMP MotionLoader (import the module directly to avoid triggering
# the humanoid task package's gym registration)
from isaaclab_tasks.direct.humanoid_amp.motions.motion_loader import MotionLoader

from .bingo_amp_env_cfg import BingoAmpEnvCfg

KEY_BODY_NAMES = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]

# Naturalness fix (bingoAMPNaturalMovement.md): the knee-link body origin is the knee
# HINGE, whose height depends only on SY+SP (barely varies) — the knee flexion that
# actually lifts the paw happens BELOW it. So we report the SHANK TIP as the AMP "foot":
# hinge + this downward offset rotated by the knee-link orientation. Must equal SHANK_LEN
# in retarget_dog_to_bingo.py so the reference and policy foot-height channels match
# (an unmatched channel lets the discriminator win for free and kills the style gradient).
SHANK_LEN = 0.12


class BingoAmpEnv(DirectRLEnv):
    cfg: BingoAmpEnvCfg

    def __init__(self, cfg: BingoAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # Bingo's articulation has 17 joints (12 legs + 5 head/tail); the motion and
        # action space cover only the 12 leg joints. Restrict everything to those,
        # in the motion's DOF order.
        self.ctrl_dof_names = list(self._motion_loader.dof_names)  # 12 leg joints, canonical order
        self.robot_ctrl_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in self.ctrl_dof_names],
            device=self.device, dtype=torch.long,
        )
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.ctrl_dof_names)

        # action offset/scale from the 12 controlled joints only
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.robot_ctrl_indexes, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        # cap SY (abduction) authority so the robot keeps its legs UNDER the body and
        # can't splay out into a low spider-crouch (SY joints are at ctrl indices 0,3,6,9)
        sy_idx = [i for i, n in enumerate(self.ctrl_dof_names) if n.endswith("_SY_J")]
        self.action_scale[sy_idx] *= 0.3

        # body indexes
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in KEY_BODY_NAMES]
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(KEY_BODY_NAMES)

        # contact-sensor foot indexes (for the gait-cycling reward)
        self.contact_foot_idx = [self.contact_sensor.body_names.index(n) for n in KEY_BODY_NAMES]

        # shank-tip offset in each knee link's frame (matches SHANK_LEN in the retargeter):
        # turns the tracked knee HINGE into the real contact point so the AMP foot-height
        # channel carries paw lift on the policy side too.
        self.shank_offset = torch.tensor(
            [[0.0, 0.0, -SHANK_LEN]] * len(KEY_BODY_NAMES), device=self.device
        )  # (4, 3)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # per-env velocity commands [vx, yaw], sampled at reset (v14 command conditioning)
        self.commands = torch.zeros((self.num_envs, 2), device=self.device)

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
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self.robot_ctrl_indexes)

    def _get_observations(self) -> dict:
        # shank tips (not the raw knee-hinge origins) as the AMP "feet" — see SHANK_LEN
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]    # (N,4,3) hinge
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]  # (N,4,4)
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        foot_tips = knee_pos + quat_apply(knee_quat, offset)
        amp_obs = compute_obs(
            self.robot.data.joint_pos[:, self.robot_ctrl_indexes],
            self.robot.data.joint_vel[:, self.robot_ctrl_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            foot_tips,
        )
        # discriminator sees ONLY the 49 style features (command-agnostic); the policy also
        # sees the 2 velocity commands -> 51-dim policy obs.
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        policy_obs = torch.cat([amp_obs, self.commands], dim=-1)
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # v14: PURE velocity-tracking task reward (AMP_for_hardware recipe). The gait style
        # (diagonal trot / lateral pace / foot lift / posture) is supplied entirely by the
        # AMP discriminator matching the multi-gait dog reference distribution -- NO
        # hand-crafted gait/height shaping (those hand terms are what AMP is meant to
        # replace, and they caused the emergent leg-sacrifice asymmetry). The command is in
        # the policy obs, so the policy tracks whatever [vx, yaw] it was given.
        vx = self.robot.data.root_lin_vel_b[:, 0]
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        cmd_vx, cmd_yaw = self.commands[:, 0], self.commands[:, 1]
        lin_track = torch.exp(-(vx - cmd_vx) ** 2 / self.cfg.tracking_sigma_vx)
        yaw_track = torch.exp(-(yaw_rate - cmd_yaw) ** 2 / self.cfg.tracking_sigma_yaw)
        return lin_track + 0.5 * yaw_track

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # resample per-episode velocity commands [vx, yaw]. For eval/recording, env vars
        # BINGO_CMD_VX / BINGO_CMD_YAW pin the command to a fixed value (else sample range).
        n = len(env_ids)
        vx_lo, vx_hi = self.cfg.cmd_vx_range
        yaw_lo, yaw_hi = self.cfg.cmd_yaw_range
        fx, fy = os.environ.get("BINGO_CMD_VX"), os.environ.get("BINGO_CMD_YAW")
        if fx is not None:
            self.commands[env_ids, 0] = float(fx)
        else:
            self.commands[env_ids, 0] = torch.rand(n, device=self.device) * (vx_hi - vx_lo) + vx_lo
        if fy is not None:
            self.commands[env_ids, 1] = float(fy)
        else:
            self.commands[env_ids, 1] = torch.rand(n, device=self.device) * (yaw_hi - yaw_lo) + yaw_lo

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_default(self, env_ids: torch.Tensor):
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(self, env_ids: torch.Tensor, start: bool = False):
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.03  # small lift to avoid ground penetration on reset
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]
        # expand the 12 controlled DOFs into the robot's full 17-DOF state (head/tail
        # stay at their defaults)
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.robot_ctrl_indexes] = dof_positions[:, self.motion_dof_indexes]
        dof_vel[:, self.robot_ctrl_indexes] = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)
        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
