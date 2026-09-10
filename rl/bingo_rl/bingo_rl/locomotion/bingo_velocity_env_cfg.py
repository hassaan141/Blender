"""Bingo command-conditioned locomotion (Task 1) on the validated v4 physics robot.

    command [vx, vy, yaw_rate] -> policy -> 12 leg joint targets -> Bingo physics

This is NOT imitation learning. No personality clip is used as a reference, and
nothing in the Stage 1-5 animation pipeline is read, written or modified.

Design report: docs/locomotion/TASK1_DESIGN.md. Every number below that is not a
stock Isaac Lab default is justified either there or inline here.

What is reused unchanged
------------------------
* ``BINGO_V4_CFG`` - the validated Stage-4 physics: v4 USD, masses, collision
  geometry, joint limits, IdealPD actuators and their measured gains.
* The stock manager-based velocity task, the same one ``env_cfg.py`` and
  ``improved_walking_cfg.py`` already subclass successfully against this Isaac Lab
  install.

What is deliberately different from ``improved_walking_cfg.py``
---------------------------------------------------------------
That config is the project's original walking success, but it runs on the **rev_3**
robot with implicit actuators and weak drives. This one runs the **v4** robot with
explicit IdealPD actuators, 21 joints, the solved stance, and per-joint action
scales. The reward *structure* and the two anti-degenerate-gait terms are inherited
from it because those were measured on Bingo; the robot and the action interface
are not.
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from ..bingo_v4 import BINGO_V4_CFG
from . import bingo_velocity_mdp as bingo_mdp

# ---------------------------------------------------------------- robot topology
BASE_LINK = "origin"
# v4 has NO ankle or paw joint: the paw is rigid collision geometry on the
# shank/knee link, so the knee link IS the contact body. Same convention as
# env_cfg.FOOT_BODIES and stage4/contact_model.PAW_LINKS.
FOOT_BODIES = ".*_knee"
LEG_JOINTS = "(fl|fr|bl|br)_.*"
EXPR_JOINTS = "(head_.*|tail_.*|.*_ear_.*)"

# ---------------------------------------------------------------- control timing
# 120 Hz physics / decimation 5 = 24 Hz policy. Identical to the Stage 4 baseline
# (rl/tools/track_v4_physics.py) and Stage 5 (stage5/bingo_stage5_env_cfg.py).
# The brief fixes this while the baseline is established. NOTE this is slower than
# the 50 Hz most Isaac Lab locomotion tasks use, so any weight inherited from a
# 50 Hz config that scales with control rate - action_rate_l2 above all - is a
# candidate for re-measurement, not a validated value.
_DECIMATION = 5
_PHYSICS_DT = 1.0 / 120.0

# ---------------------------------------------------------------- standing pose
# Solved by stage4/solve_stand_pose.py against the REAL collision hulls and carried
# in stage4/stand_test.py as STAND_SOLVED. Verified independently in this session:
# all four paw hulls sit at exactly -0.1800 m in the base frame (spread 0.00 mm),
# support polygon x -0.059..+0.059, y -0.053..+0.053, no joint outside its limits.
#
# This REPLACES the pose currently in BINGO_V4_CFG.init_state, which is the
# inherited rev_3 pose (SP +-0.3, knee +-0.6). Measured against the same hulls that
# pose puts one paw 207 mm in front of the base while another is at 24 mm, with paws
# at unequal heights (8.14 mm spread); stand_test.py records that it "drove fr_SP_J
# into its +1.56 limit and jammed the leg against the floor".
#
# bingo_v4.py is NOT edited - it is validated Stage-4 physics and Stage 5 depends on
# it (RSI overwrites the init state every reset, so the stale pose is harmless
# there). The override is local to this task.
STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}
EXPR_NEUTRAL = {"head_.*": 0.0, "tail_.*": 0.0, ".*_ear_.*": 0.0}

# Lowest paw hull is 0.180 m below the base at this pose; spawn 2 mm clear.
STAND_BASE_HEIGHT = 0.182

# ---------------------------------------------------------------- action scales
# Derived, not inherited. Usable travel from STAND_SOLVED to the soft limit
# (soft_joint_pos_limit_factor = 0.9) is, per joint type and taking the worst leg:
#     SY   0.378 rad      SP   0.593 rad      knee  0.510 rad
# Choosing the scale so that |action| = 3 lands exactly on the soft limit lets the
# policy use the full range of its unit-variance Gaussian without ever commanding a
# target outside the soft limits.
#
# This is deliberately NOT Stage 5's single 0.3. At 0.3, |a| = 3 asks SY for 0.90 rad
# against 0.378 rad of headroom - 2.4x past its stop. That is the SY overrun MEMORY.md
# records as warned-but-unsolved. One scalar cannot serve a robot whose SY range is
# 3.7x smaller than its SP range.
ACTION_SCALE = {".*_SY_J": 0.125, ".*_SP_J": 0.195, ".*_knee": 0.170}


# =============================================================================
# Rewards
# =============================================================================
@configclass
class BingoVelocityRewards(RewardsCfg):
    """Stock velocity rewards, plus the terms Bingo measurably needs."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )

    # anti-tripod: a leg held permanently aloft is never punished by feet_air_time.
    air_time_over = RewTerm(
        func=bingo_mdp.air_time_over_limit,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "max_air_time": 0.35,
        },
    )

    # anti-prop: a foot parked on the ground as a static support is likewise free.
    contact_time_over = RewTerm(
        func=bingo_mdp.contact_time_over_limit,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "max_contact_time": 0.35,
        },
    )

    # Makes the exact-zero command mean "hold the stance", not "march on the spot"
    # (which scores a perfect base-velocity tracking reward).
    stand_still = RewTerm(
        func=bingo_mdp.stand_still_on_zero_command,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS]),
            "command_threshold": 0.05,
        },
    )

    # Limit penalty on the LEG joints only. Averaged over all 21 it would be
    # dominated by joints nowhere near their stops; SY is the one that binds.
    leg_pos_limits = RewTerm(
        func=bingo_mdp.joint_position_limit_margin,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])},
    )


@configclass
class BingoVelocityTerminations(TerminationsCfg):
    """Stock terminations plus real fall detection."""

    # 0.14 m is 77% of the measured 0.182 m stance height. Below this the robot has
    # collapsed ~40 mm and cannot be walking.
    low_base = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.14},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},   # 45.8 deg
    )


# =============================================================================
# Flat environment
# =============================================================================
@configclass
class BingoVelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Bingo-Velocity-Flat-v4-v0: command-conditioned locomotion on flat ground."""

    rewards: BingoVelocityRewards = BingoVelocityRewards()
    terminations: BingoVelocityTerminations = BingoVelocityTerminations()

    sim: SimulationCfg = SimulationCfg(
        dt=_PHYSICS_DT,
        render_interval=_DECIMATION,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------- timing
        self.decimation = _DECIMATION
        self.episode_length_s = 20.0
        self.sim.dt = _PHYSICS_DT
        self.sim.render_interval = _DECIMATION
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # ---------------------------------------------------------------- robot
        # Validated v4 physics, with only the stance pose overridden (see above).
        robot = BINGO_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.init_state = BINGO_V4_CFG.init_state.replace(
            pos=(0.0, 0.0, STAND_BASE_HEIGHT),
            joint_pos={**STAND_SOLVED, **EXPR_NEUTRAL},
        )
        self.scene.robot = robot

        # ---------------------------------------------------------------- terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------------- actions
        # 12 leg joints only. The 9 expressive joints are left out of the action
        # space entirely and are held at their neutral targets by their own drives,
        # so Task 2 can drive them without changing the action shape.
        self.actions.joint_pos.joint_names = [LEG_JOINTS]
        self.actions.joint_pos.scale = dict(ACTION_SCALE)
        # q_target = default_joint_pos + scale * action, i.e. offset by the stance.
        self.actions.joint_pos.use_default_offset = True

        # ------------------------------------------------------------ observations
        # The stock policy observation is kept as-is. Its joint_pos_rel / joint_vel_rel
        # terms use the default asset_cfg, which is ALL joints, so all 21 are observed
        # while only 12 are actioned - exactly what the brief asks for, and what lets
        # Task 2 move the expressive joints without an observation-shape change.
        #
        # Layout: base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 +
        #         velocity_commands 3 + joint_pos 21 + joint_vel 21 + actions 12 = 66.
        # Cross-check: improved_walking_cfg records 58 for the 17-joint rev_3 robot,
        # and 3+3+3+3+17+17+12 = 58. No world XYZ is exposed to the actor.
        self.observations.policy.enable_corruption = False   # stage 7 turns this on

        # ---------------------------------------------------------------- commands
        self.commands.base_velocity = bingo_mdp.BingoVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 6.0),
            rel_standing_envs=0.0,      # "stand" is its own category instead
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.25, 0.40),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-1.2, 1.2),
            ),
        )

        # ---------------------------------------------------------------- events
        # Stage 1 is nominal physics: no randomisation until a gait exists. Stages
        # 7-11 re-enable these one at a time via apply_training_stage().
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = [BASE_LINK]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_LINK]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in
                               ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # ---------------------------------------------------------------- rewards
        # Weights carried from improved_walking_cfg where that config MEASURED them
        # on Bingo; see the design report for which are inherited vs. stock.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.feet_air_time.weight = 0.25
        # Any non-foot link touching the ground is a fall in progress. These are the
        # real v4 link names (verified against the physics URDF's 22 links) and are
        # restricted to the links that actually carry collision geometry in
        # stage4/out/collision_hulls.npz - the ear and head_pitch/head_yaw links have
        # none, so a contact term on them would never fire. "origin" is left out
        # because terminations.base_contact already ends the episode on it.
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
                ".*_shoulder_yaw", ".*_shoulder_pitch", "head_roll", "tail_yaw",
            ]
            self.rewards.undesired_contacts.weight = -1.0
        # The stock dof_pos_limits covers all joints; leg_pos_limits above is the
        # one that matters, so leave the stock term off to avoid double-counting.
        if getattr(self.rewards, "dof_pos_limits", None) is not None:
            self.rewards.dof_pos_limits.weight = 0.0

        # ------------------------------------------------------------ terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_LINK]


@configclass
class BingoVelocityFlatEnvCfg_PLAY(BingoVelocityFlatEnvCfg):
    """Deterministic playback / evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.episode_length_s = 40.0
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }


@configclass
class BingoVelocityStandTestCfg_PLAY(BingoVelocityFlatEnvCfg_PLAY):
    """Stance gate: action scale 0, so the robot can only hold STAND_SOLVED.

    Run this BEFORE any training. If Bingo cannot hold this pose under its own
    drives it is not a reward problem, and no amount of PPO will fix it. Mirrors
    stage4/stand_test.py's role, inside the RL env so the env's own physics,
    spawn height and contact setup are what gets tested.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.actions.joint_pos.scale = 0.0
        self.terminations.low_base = None
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None
        self.commands.base_velocity.category_weights = {"stand": 1.0}


# =============================================================================
# Training curriculum
# =============================================================================
# The brief's 11-step progression. Stages 1-6 shape the command distribution and
# rewards; stages 7-11 add randomisation, one source at a time. Nothing aggressive
# is enabled before a gait exists.
TRAINING_STAGES = {
    1: "flat floor, nominal physics, forward-only, narrow commands",
    2: "reliable forward + stand",
    3: "wider forward/backward velocity range",
    4: "turning",
    5: "reverse and lateral",
    6: "start/stop transitions (shorter command resampling)",
    7: "observation noise",
    8: "friction variation",
    9: "mass / COM variation",
    10: "actuator variation",
    11: "pushes",
}


def apply_training_stage(cfg: BingoVelocityFlatEnvCfg, stage: int):
    """Mutate ``cfg`` in place to the given curriculum stage (cumulative).

    Each stage keeps everything the previous stages enabled. Applied after
    ``__post_init__``, i.e. to an already-constructed cfg object.
    """
    if stage < 1 or stage > 11:
        raise ValueError(f"stage must be 1..11, got {stage}")

    w = {c: 0.0 for c in bingo_mdp.CATEGORIES}

    # --- stages 1-6: command distribution -----------------------------------
    if stage >= 1:
        w["forward"] = 1.0
        cfg.commands.base_velocity.category_ranges["forward_vx"] = (0.10, 0.25)
    if stage >= 2:
        w["stand"] = 0.25
    if stage >= 3:
        cfg.commands.base_velocity.category_ranges["forward_vx"] = (0.10, 0.40)
        w["backward"] = 0.3
    if stage >= 4:
        w["turn_in_place"] = 0.5
        w["turn_and_move"] = 0.7
    if stage >= 5:
        w["lateral"] = 0.4
        w["diagonal"] = 0.4
    if stage >= 6:
        # Shorter resampling means more command CHANGES per episode, which is what
        # actually trains start/stop transitions.
        cfg.commands.base_velocity.resampling_time_range = (2.0, 4.0)
    cfg.commands.base_velocity.category_weights = w

    # --- stages 7-11: randomisation, one source at a time --------------------
    if stage >= 7:
        cfg.observations.policy.enable_corruption = True
    if stage >= 8:
        cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.9)
    if stage >= 9:
        # Bingo's base is ~2.45 kg, so +-0.3 kg is a ~12% payload perturbation.
        cfg.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.3)
    if stage >= 10:
        cfg.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        cfg.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)
    if stage >= 11:
        cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3),
                                       "yaw": (-0.4, 0.4)}},
        )
    return cfg


# Stage-specialised cfg classes, so a curriculum stage is selectable as a task id
# rather than a flag that has to be threaded through the training script. Each is
# the flat cfg with apply_training_stage() run at the end of __post_init__.
def _make_stage_cfg(stage: int):
    @configclass
    class _StageCfg(BingoVelocityFlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            apply_training_stage(self, stage)

    _StageCfg.__name__ = f"BingoVelocityFlatEnvCfg_S{stage}"
    _StageCfg.__qualname__ = _StageCfg.__name__
    _StageCfg.__doc__ = f"Stage {stage}: {TRAINING_STAGES[stage]}"
    return _StageCfg


BingoVelocityFlatEnvCfg_S1 = _make_stage_cfg(1)
BingoVelocityFlatEnvCfg_S2 = _make_stage_cfg(2)
BingoVelocityFlatEnvCfg_S3 = _make_stage_cfg(3)
BingoVelocityFlatEnvCfg_S4 = _make_stage_cfg(4)
BingoVelocityFlatEnvCfg_S5 = _make_stage_cfg(5)
BingoVelocityFlatEnvCfg_S6 = _make_stage_cfg(6)
BingoVelocityFlatEnvCfg_S7 = _make_stage_cfg(7)
BingoVelocityFlatEnvCfg_S8 = _make_stage_cfg(8)
BingoVelocityFlatEnvCfg_S9 = _make_stage_cfg(9)
BingoVelocityFlatEnvCfg_S10 = _make_stage_cfg(10)
BingoVelocityFlatEnvCfg_S11 = _make_stage_cfg(11)
