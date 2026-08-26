"""Stage 5 residual RL env: PD on the Stage 4 reference + a small learned correction.

    legs (12):        q_target = q_ref(t) + residual_scale * action
    expressive (9):   q_target = q_ref(t)                    (fed from the reference)

The policy can NEVER command an absolute joint target: `_apply_action` only ever
adds `residual_scale * action` to the reference feed-forward.

Everything reused from bingo_rl.track (spec-B.4.1 reward, RSI, phase obs, foot-tip
key bodies) and bingo_rl.track_v4 (21-DOF split: legs RL, head/tail/ears from the
reference). This subclass exists to hold the reference/physics conventions of the
Stage 4 baseline (rl/tools/track_v4_physics.py) so that a ZERO residual reproduces
Stage 4 and the comparison is apples-to-apples:

  * sim dt 1/120, decimation 5 -> 24 Hz control == reference fps
  * FIRST-ORDER HOLD of the reference across the decimation window, exactly as
    Stage 4 does (f = (k+1)/decim). A 24 Hz zero-order-hold target against
    Kp = 120 is a ~0.33 rad step, i.e. an impulsive kick 24x/second.
  * joint VELOCITY target left at zero (Stage 4 runs without --vel-ff, which
    measurably regresses because Kd*qdot_ref exceeds the 3 N m effort ceiling)
  * default GroundPlaneCfg friction (0.5/0.5), not the deadpan tracker's 1.0/1.0
  * fall criterion: root_z < 0.5*z_ref[0]  or  tilt > 70 deg
  * frame-exact reference indexing (no continuous resampling, no wrap-around:
    Timid is not a cyclic clip)
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse

from bingo_rl.track.bingo_track_env import KEY_BODY_NAMES
from bingo_rl.track_v4.bingo_track_v4_env import BingoTrackV4Env

from .bingo_stage5_env_cfg import BingoStage5TimidEnvCfg

# The 12 controlled leg joints, in the reference's own column order.
# NOTE: BingoTrackEnv derives ctrl_dof_names from the motion file's dof_names,
# which for a v4 clip is ALL 21 joints (legs + head/tail + ears). The policy
# controls only the legs, so the mapping is re-derived explicitly here.
LEG_JOINT_NAMES = [
    "fl_SY_J", "fl_SP_J", "fl_knee",
    "fr_SY_J", "fr_SP_J", "fr_knee",
    "bl_SY_J", "bl_SP_J", "bl_knee",
    "br_SY_J", "br_SP_J", "br_knee",
]


class BingoStage5Env(BingoTrackV4Env):
    cfg: BingoStage5TimidEnvCfg

    # ------------------------------------------------------------------ setup
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- re-derive the 12-leg control mapping (see LEG_JOINT_NAMES note) ---
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
        # 21-DOF observation order: 12 legs (ctrl order) then 9 expressive
        self.obs_dof_indexes = torch.cat([self.robot_ctrl_indexes, self.expr_indexes])

        # per-joint residual authority, from cfg.residual_scale (+ optional per-type)
        n_leg = len(self.ctrl_dof_names)
        self._res_scale = torch.full(
            (n_leg,), float(self.cfg.residual_scale), device=self.device
        )
        byt = getattr(self.cfg, "residual_scale_by_type", None)
        if byt:
            for i, n in enumerate(self.ctrl_dof_names):
                if "SY_J" in n and "SY" in byt:
                    self._res_scale[i] = byt["SY"]
                elif "SP_J" in n and "SP" in byt:
                    self._res_scale[i] = byt["SP"]
                elif "knee" in n and "knee" in byt:
                    self._res_scale[i] = byt["knee"]

        # --- frame-exact reference tables -------------------------------------
        self._T = int(self._motion_loader.num_frames)
        self._fps = 1.0 / float(self._motion_loader.dt)
        self._ref_leg_q = self._motion_loader.dof_positions[:, self.motion_dof_indexes].contiguous()
        self._ref_leg_qd = self._motion_loader.dof_velocities[:, self.motion_dof_indexes].contiguous()
        self._ref_all_q = self._motion_loader.dof_positions.contiguous()
        self._ref_all_qd = self._motion_loader.dof_velocities.contiguous()
        self._ref_root_p = self._motion_loader.body_positions[:, self.motion_ref_body_index].contiguous()
        self._ref_root_q = self._motion_loader.body_rotations[:, self.motion_ref_body_index].contiguous()
        self._ref_root_lv = self._motion_loader.body_linear_velocities[:, self.motion_ref_body_index].contiguous()
        self._ref_root_av = self._motion_loader.body_angular_velocities[:, self.motion_ref_body_index].contiguous()
        self._ref_key_p = self._motion_loader.body_positions[:, self.motion_key_body_indexes].contiguous()
        # map the reference's 21 dof columns onto Isaac's joint order (Isaac orders
        # DOFs breadth-first; the npz is per-leg, so positional indexing scrambles legs)
        self._all_dof_isaac = torch.tensor(
            [self.robot.data.joint_names.index(str(n)) for n in self._motion_loader.dof_names],
            device=self.device, dtype=torch.long,
        )

        # Stage 4 fall criterion
        self._fall_z = 0.5 * float(self._ref_root_p[0, 2])
        self._tilt_limit = float(getattr(self.cfg, "tilt_limit_deg", 70.0))

        self._start_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._substep = 0
        # populated in _get_rewards, read by _get_dones (one step stale, as upstream)
        self._root_track_dist = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        """Same as BingoTrackV4Env but with the DEFAULT ground material, so the
        contact physics matches the validated Stage 4 baseline (0.5/0.5)."""
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ reference
    def _frames(self, offset: int = 0) -> torch.Tensor:
        """Reference frame index per env, clamped (no wrap: the clip is not cyclic)."""
        f = self._start_frames + self.episode_length_buf.long() + offset
        return torch.clamp(f, 0, self._T - 1)

    def _current_times(self) -> np.ndarray:
        """Time of the LAST COMMANDED reference frame (used by the inherited reward)."""
        return (self._frames(0).float() / self._fps).detach().cpu().numpy()

    def _sample_ref(self, times: np.ndarray):
        """Frame-exact override of BingoTrackEnv._sample_ref (same return contract)."""
        f = torch.as_tensor(np.round(np.asarray(times) * self._fps).astype(np.int64),
                            device=self.device)
        f = torch.clamp(f, 0, self._T - 1)
        root_pos = self._ref_root_p[f]
        root_quat = self._ref_root_q[f]
        key_pos = self._ref_key_p[f]
        k = len(KEY_BODY_NAMES)
        key_local = quat_rotate_inverse(
            root_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            (key_pos - root_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(-1, k, 3)
        return (self._ref_leg_q[f], self._ref_leg_qd[f], root_pos, root_quat,
                self._ref_root_lv[f], self._ref_root_av[f], key_local)

    # ------------------------------------------------------------------ RL loop
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        f0 = self._frames(0)
        f1 = self._frames(1)
        # feed-forward endpoints for the first-order hold across the decimation window
        self._ff_leg0, self._ff_leg1 = self._ref_leg_q[f0], self._ref_leg_q[f1]
        self._ff_expr0 = self.expr_ref[torch.clamp(f0, 0, self.n_expr - 1)]
        self._ff_expr1 = self.expr_ref[torch.clamp(f1, 0, self.n_expr - 1)]
        self._substep = 0

    def _apply_action(self):
        # Stage 4 interpolation: f = (k+1)/decim for k = 0..decim-1
        self._substep += 1
        a = self._substep / float(self.cfg.decimation)
        leg_ff = (1.0 - a) * self._ff_leg0 + a * self._ff_leg1
        # THE residual: the policy only ever adds to the reference, never replaces it
        self.robot.set_joint_position_target(
            leg_ff + self._res_scale * self.actions, joint_ids=self.robot_ctrl_indexes
        )
        expr = (1.0 - a) * self._ff_expr0 + a * self._ff_expr1
        self.robot.set_joint_position_target(expr, joint_ids=self.expr_indexes)
        # joint velocity target deliberately left at zero (Stage 4 has --vel-ff off)

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        f = self._start_frames + self.episode_length_buf.long()
        time_out = (f >= (self._T - 1)) | (self.episode_length_buf >= self.max_episode_length - 1)
        if not self.cfg.early_termination:
            return torch.zeros_like(time_out), time_out
        q = self.robot.data.root_quat_w                      # (N,4) wxyz
        r22 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)      # = R[2,2]
        tilt = torch.rad2deg(torch.arccos(torch.clamp(r22, -1.0, 1.0)))
        root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fell = (root_z < self._fall_z) | (tilt > self._tilt_limit)
        drifted = self._root_track_dist > self.cfg.root_track_max
        return fell | drifted, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        """Reference-state initialization on integer frames.

        Deliberately bypasses BingoTrackEnv._reset_idx (continuous sample_times and
        a +0.02 m lift) so that a frame-0 reset reproduces the Stage 4 baseline init.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        n = len(env_ids)
        if getattr(self.cfg, "random_start_frame", True):
            fr = torch.randint(0, self._T - 1, (n,), device=self.device, dtype=torch.long)
        else:
            fr = torch.zeros(n, dtype=torch.long, device=self.device)
        self._start_frames[env_ids] = fr
        self.start_times[env_ids] = fr.float() / self._fps
        # clear the drift buffer: it is filled by _get_rewards, which runs AFTER
        # _get_dones, so a stale value would immediately re-terminate the fresh episode
        self._root_track_dist[env_ids] = 0.0

        use_v = bool(getattr(self.cfg, "rsi_use_reference_velocity", True))

        root_state = torch.zeros((n, 13), device=self.device)
        root_state[:, 0:3] = self._ref_root_p[fr] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = self._ref_root_q[fr]
        if use_v:
            root_state[:, 7:10] = self._ref_root_lv[fr]
            root_state[:, 10:13] = self._ref_root_av[fr]

        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = torch.zeros_like(dof_pos)
        dof_pos[:, self._all_dof_isaac] = self._ref_all_q[fr]
        if use_v:
            dof_vel[:, self._all_dof_isaac] = self._ref_all_qd[fr]

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
