"""Bingo AMP (Adversarial Motion Priors) env config — dog-style locomotion.

Direct-workflow env mirroring IsaacLab's humanoid_amp, adapted to Bingo:
  - 12-DOF quadruped, key bodies = the 4 feet (*_knee).
  - reference body = base link "origin".
  - reference motion = retargeted dog trot (bingo_trot.npz).
Pure imitation (task_reward_scale 0 in the skrl cfg): the AMP discriminator supplies
the dog style; there is no hand-built gait shaping. The v7 symmetry penalties are
intentionally absent — the dog data encodes symmetry.
"""

from __future__ import annotations

import os

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions")

# Bilaterally symmetric default pose (per-leg axis SIGN MAP; see retargeter).
# fr SP&knee flipped, br knee flipped, relative to the left legs.
_SYMMETRIC_POSE = {
    ".*_SY_J": 0.0,
    "fl_SP_J": -0.3, "bl_SP_J": -0.3, "fr_SP_J": +0.3, "br_SP_J": -0.3,
    "fl_knee": +0.6, "bl_knee": +0.6, "fr_knee": -0.6, "br_knee": -0.6,
}


@configclass
class BingoAmpEnvCfg(DirectRLEnvCfg):
    """Bingo AMP environment (dog-trot imitation)."""

    # env
    episode_length_s = 10.0
    decimation = 4

    # AMP style features (discriminator obs): dof_pos(12)+dof_vel(12)+root_h(1)+
    #   quat_tan_norm(6)+root_lin(3)+root_ang(3)+4feet*3(12) = 49  (command-AGNOSTIC)
    amp_observation_space = 49
    num_amp_observations = 2
    # policy obs = 49 style features + 2 velocity commands (vx, yaw) = 51.
    # v14: command conditioning (AMP_for_hardware recipe) -> the policy tracks a commanded
    # forward speed + turn rate; the discriminator never sees the command (style is
    # command-agnostic). This is what lets a distribution of dog gaits + a pure velocity
    # task reward produce natural, controllable dog locomotion instead of one fixed cadence.
    observation_space = 51
    action_space = 12
    state_space = 0

    # velocity command ranges (sampled per episode)
    cmd_vx_range = (0.15, 0.8)     # m/s forward
    cmd_yaw_range = (-0.6, 0.6)    # rad/s turn
    tracking_sigma_vx = 0.25       # exp(-(vx-cmd)^2/sigma)
    tracking_sigma_yaw = 0.30

    early_termination = True
    termination_height = 0.15  # hard floor: no low belly-crawl (the gait shaping made it
    #                            crouch to 0.12; force it to keep a tall dog stance)

    # velocity-tracking task reward: without it, pure AMP style -> the robot trots in
    # place (matches the leg rhythm without translating). This gives it a reason to
    # travel forward; the task/style ratio is tuned in the skrl agent cfg.
    target_speed = 0.5   # m/s forward, aligned with the retargeted reference root speed
    vel_reward_sigma = 0.25

    # Gait-cycling shaping (the fix for dragging back legs + veer): the AMP "feet" are
    # knee joints that barely move vertically, so the discriminator can't tell a
    # stepping leg from a dragging one. Symmetric air/contact-time caps (v7's trick)
    # force EVERY foot onto a bounded lift/plant cycle -> back legs must step, and the
    # symmetry kills the left/right veer. A yaw penalty keeps it going straight.
    max_air_time = 0.35
    max_contact_time = 0.35
    gait_penalty_scale = 0.7   # exp(-scale * bounded_over) multiplies task reward (floor ~0.35)
    yaw_penalty_scale = 0.5    # exp(-scale * bounded_|yaw|) (floor ~0.37)
    target_height = 0.18       # nudge toward a tall dog stance (termination floor is 0.15)
    height_penalty_scale = 8.0 # exp(-scale * max(target_height - base_h, 0))

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_knee", history_length=3, track_air_time=True
    )

    motion_file: str = os.path.join(MOTIONS_DIR, "bingo_multigait.npz")  # v14: trot + pace distribution
    reference_body = "origin"
    reset_strategy = "random"  # default, random, random-start

    # simulation — 30 Hz control (dt 1/120 * decimation 4), matches motion cadence well
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # close-up camera that FOLLOWS env-0's robot (side view) so recordings show the
    # actual gait, not tiny robots on a fixed-wide grid
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.8, -1.3, 0.5), lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root", asset_name="robot", env_index=0,
    )

    # robot: reuse the v7 asset (rev_3 USD + measured drive gains), symmetric default pose
    robot = BINGO_IMPROVED_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        # apply the symmetric default pose to the reused robot cfg
        self.robot.init_state.joint_pos = dict(_SYMMETRIC_POSE)
        self.robot.init_state.pos = (0.0, 0.0, 0.22)
