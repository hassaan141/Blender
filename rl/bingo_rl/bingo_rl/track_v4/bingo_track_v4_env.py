"""v4 21-DOF expression tracker env: legs RL + 9 expressive DOF (head/tail + ears) fed
feed-forward from the reference. Subclasses BingoTrackEnv (inherits the spec-B.4.1 reward,
termination, RSI). On the v4 USD the robot has 21 joints incl. 4 ears.
"""
from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from bingo_rl.amp.bingo_amp_env import compute_obs
from bingo_rl.track.bingo_track_env import BingoTrackEnv
from .bingo_track_v4_env_cfg import BingoTrackV4EnvCfg

# 9 expressive joints in reference-column order: 5 head/tail (from head_tail_positions)
# then 4 ears (from ear_positions).
EXPR_JOINT_NAMES = [
    "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
    "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
]


class BingoTrackV4Env(BingoTrackEnv):
    cfg: BingoTrackV4EnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.expr_indexes = torch.tensor(
            [self.robot.data.joint_names.index(n) for n in EXPR_JOINT_NAMES],
            device=self.device, dtype=torch.long,
        )
        # 21-DOF obs order: 12 legs (ctrl order) + 9 expressive
        self.obs_dof_indexes = torch.cat([self.robot_ctrl_indexes, self.expr_indexes])

        # reference for the 9 expressive DOF: head/tail (5) from the clip, ears (4) from the bake
        raw = np.load(self.cfg.motion_file, allow_pickle=True)
        ht = raw["head_tail_positions"].astype(np.float32)          # (n,5)
        ears = np.load(self.cfg.ear_file, allow_pickle=True)["ear_positions"].astype(np.float32)  # (m,4)
        n = min(len(ht), len(ears))
        expr = np.concatenate([ht[:n], ears[:n]], axis=1)           # (n,9)
        self.expr_ref = torch.tensor(expr, device=self.device)
        self.n_expr = n

    def _setup_scene(self):
        # v4 USD lacks contact-reporter APIs on its links, and this tracker's reward uses
        # foot-tip height as the contact proxy (not the sensor), so skip the ContactSensor.
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0)),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _expr_targets(self, times_np: np.ndarray) -> torch.Tensor:
        fidx = np.clip(np.round(times_np * self._ref_fps).astype(int), 0, self.n_expr - 1)
        return self.expr_ref[fidx]  # (N,9)

    def _apply_action(self):
        super()._apply_action()  # legs from policy
        self.robot.set_joint_position_target(self._expr_targets(self._current_times()), joint_ids=self.expr_indexes)

    def _get_observations(self) -> dict:
        knee_pos = self.robot.data.body_pos_w[:, self.key_body_indexes]
        knee_quat = self.robot.data.body_quat_w[:, self.key_body_indexes]
        offset = self.shank_offset.unsqueeze(0).expand(knee_pos.shape[0], -1, -1)
        foot_tips = knee_pos + quat_apply(knee_quat, offset)
        proprio = compute_obs(
            self.robot.data.joint_pos[:, self.obs_dof_indexes],   # 21
            self.robot.data.joint_vel[:, self.obs_dof_indexes],   # 21
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
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        expr = self._expr_targets(self.start_times[env_ids].detach().cpu().numpy())
        self.robot.write_joint_state_to_sim(expr, torch.zeros_like(expr), self.expr_indexes, env_ids)
