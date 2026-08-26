"""Config for the Bingo DeepMimic-style single-clip tracking env."""

from __future__ import annotations

import os

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions")

# Bilaterally symmetric default pose (per-leg axis SIGN MAP; matches the AMP env).
_SYMMETRIC_POSE = {
    ".*_SY_J": 0.0,
    "fl_SP_J": -0.3, "bl_SP_J": -0.3, "fr_SP_J": +0.3, "br_SP_J": -0.3,
    "fl_knee": +0.6, "bl_knee": +0.6, "fr_knee": -0.6, "br_knee": -0.6,
}


@configclass
class BingoTrackEnvCfg(DirectRLEnvCfg):
    """Bingo tracking env (reproduce a single authored clip)."""

    # env
    episode_length_s = 8.0
    decimation = 4

    # obs: proprio(49) + phase(sin,cos)=2 = 51.  proprio = dof_pos(12)+dof_vel(12)+
    #   root_h(1)+quat_tan_norm(6)+root_lin(3)+root_ang(3)+4feet*3(12)
    observation_space = 51
    action_space = 12
    state_space = 0

    # DeepMimic imitation reward is spec B.4.1 exactly (sum-based, hard-coded in the env):
    #   0.50 exp(-2 Σq_err²) + 0.05 exp(-0.1 Σq̇_err²) + 0.20 exp(-40 Σee_err²)
    #   + 0.15 exp(-20 root_pos_err² - 10 rot_err) + 0.10 contact_match

    early_termination = True
    termination_height = 0.12      # hard floor: robot has fallen
    root_track_max = 0.15          # m: terminate if root strays > this from the reference (spec B.4.1)

    # ResMimic-style residual control. 0 = absolute PD targets (baseline). >0 = command is
    # reference_feedforward + residual_scale * action (policy learns only the balance correction).
    residual_scale = 0.0
    # optional per-joint-type override (keys "SY","SP","knee") of the residual authority, so the
    # knees can be held near the reference while SP/SY carry the balance correction.
    residual_scale_by_type = None

    # deadpan3: identical clip to deadpan but authored at root 0.1934 m (vs 0.1749) -> matches
    # the robot's nominal stance, so the tracker isn't forced into a low crouch.
    motion_file: str = os.path.join(MOTIONS_DIR, "bingo_deadpan3.npz")
    reference_body = "origin"

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_knee", history_length=3, track_air_time=True
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.8, -1.3, 0.5), lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root", asset_name="robot", env_index=0,
    )

    robot = BINGO_IMPROVED_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        self.robot.init_state.joint_pos = dict(_SYMMETRIC_POSE)
        self.robot.init_state.pos = (0.0, 0.0, 0.22)


@configclass
class BingoTrackResidualEnvCfg(BingoTrackEnvCfg):
    """ResMimic-style residual control: command = reference feedforward + residual_scale·action."""
    residual_scale = 0.3


@configclass
class BingoTrackResKneeLockEnvCfg(BingoTrackEnvCfg):
    """Residual control with knee residual restricted (knees stay near the reference) while
    SP/SY carry the balance correction. Targets the knee-dominated residual error directly."""
    residual_scale = 0.4
    residual_scale_by_type = {"SY": 0.4, "SP": 0.4, "knee": 0.15}
