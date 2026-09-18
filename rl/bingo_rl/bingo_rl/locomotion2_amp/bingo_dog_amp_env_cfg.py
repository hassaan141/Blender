"""Bingo Locomotion 2 AMP env config -- dog-style-prior + velocity-command PPO.

Strategy pivot from exact-retarget Locomotion 2 (docs/locomotion2/LOCOMOTION2_PIPELINE_REPORT.md,
which produced a kinematically valid but dynamically infeasible reference): rather than
solving Bingo joint angles to match a dog pose frame-by-frame, an AMP discriminator
compares a morphology-tolerant STYLE FEATURE VECTOR (root-local velocities, gravity,
per-leg segment directions, normalized paw kinematics, contacts -- see
docs/locomotion2/amp/extract_dog_amp_features.py for the exact schema and the dog-side
half of this pipeline) computed independently on the dog reference and on Bingo's own
simulated state. Neither side is ever mapped into the other's joint space.

Built on the VALIDATED v4 physics asset (``BINGO_V4_CFG`` from ``bingo_v4.py``, protected,
not modified) rather than the old rev_3 asset the prior (July-2026, pre-v4) AMP
experiment (``rl/bingo_rl/bingo_rl/amp/``) used -- that package is kept as-is for
reference/history but is not on the v4 asset and is not extended further.
"""
from __future__ import annotations

import os

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ..bingo_v4 import BINGO_V4_CFG

AMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "..", "docs", "locomotion2", "amp")
EXPERT_NPZ = os.path.join(AMP_DIR, "cache", "dog_amp_expert.npz")

# ---------------------------------------------------------------------------
# Copied (with attribution) from Task 1's validated v4 stance/action-scale work:
# rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_env_cfg.py (STAND_SOLVED, ACTION_SCALE).
# STAND_SOLVED was solved against v4's real collision hulls (stage4/solve_stand_pose.py);
# ACTION_SCALE gives |action|=3 exactly at each joint TYPE's soft-limit headroom from
# that stance (SY's range is 3.7x smaller than SP's, so one scalar can't serve both).
# ---------------------------------------------------------------------------
STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}
EXPR_NEUTRAL = {
    "head_.*": 0.0, "tail_.*": 0.0, ".*_ear_pitch": 0.0,
    "l_ear_roll": -0.75, "r_ear_roll": +0.75,
}
STAND_BASE_HEIGHT = 0.182
ACTION_SCALE = {".*_SY_J": 0.125, ".*_SP_J": 0.195, ".*_knee": 0.170}

LEGS = ["fl", "fr", "bl", "br"]
DOF_ORDER = [f"{l}_{j}" for l in LEGS for j in ("SY_J", "SP_J", "knee")]
HIP_BODIES = [f"{l}_shoulder_pitch" for l in LEGS]
KNEE_BODIES = [f"{l}_knee" for l in LEGS]
SHANK_LEN = 0.120  # m, knee hinge -> paw contact point (matches docs/locomotion2 retarget)

# measured from the v4 physics URDF via Urdf.leg_reach() (docs/locomotion2/dataset/retarget_to_bingo.py)
BINGO_LEG_SCALE = 0.2036


@configclass
class BingoDogAmpEnvCfg(DirectRLEnvCfg):
    """Bingo Locomotion 2: dog-style AMP prior + velocity-command task reward."""

    episode_length_s = 12.0
    decimation = 4  # 120 Hz physics / 4 = 30 Hz control, matches the expert buffer's resampled rate

    # AMP style features: see extract_dog_amp_features.py's 61-dim schema (root-local
    # vel(3)/leg_scale, ang_vel(3), gravity(3), seg_dirs(24), paw_rel(12)/leg_scale,
    # paw_vel(12)/leg_scale, contacts(4)). Command-AGNOSTIC (no vx/vy/yaw in it).
    amp_observation_space = 61
    num_amp_observations = 2
    # policy obs = 61 style features + 3 velocity commands (vx, vy, yaw_rate) = 64.
    observation_space = 64
    action_space = 12
    state_space = 0

    # ---- first-controller command scope (per the brief): forward-only, no domain
    # randomization. vy and yaw_rate stay fixed at 0 but are kept as observed/architected
    # command dims so a later stage can widen the ranges without an obs-shape change.
    cmd_vx_range = (0.15, 0.30)
    cmd_vy_range = (0.0, 0.0)
    cmd_yaw_range = (0.0, 0.0)
    stand_prob = 0.20  # fraction of episodes commanded to stand (vx=0) rather than walk
    deterministic_spawn = False  # PLAY cfg sets True: no xy/yaw spawn noise, for reproducible eval
    tracking_sigma_vx = 0.06
    tracking_sigma_vy = 0.05
    tracking_sigma_yaw = 0.10

    early_termination = True
    termination_height = 0.14  # 77% of STAND_BASE_HEIGHT, matches Task 1's low_base gate
    bad_orientation_limit = 0.8  # rad, matches Task 1

    # light regularizers only -- AMP supplies the gait SHAPE; these are safety nets,
    # not the primary reward, per the brief ("do not let handcrafted gait rewards
    # dominate so strongly that AMP becomes irrelevant").
    upright_weight = 0.10
    action_rate_weight = -0.01
    joint_acc_weight = -2.5e-7
    torque_weight = -5.0e-5

    # anti-degenerate-gait safety net (run_03): Bingo has a DOCUMENTED history of
    # dragging/parking legs as a cheap way to satisfy a velocity reward without
    # real stepping (rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_mdp.py's
    # air_time_over_limit/contact_time_over_limit, written after MEASURING this on
    # rev_3/v6). run_01 (AMP alone, no such term) reproduced exactly that failure --
    # 2 of 4 feet at 0.0 contact duty, visually confirmed static-drag gait. These
    # bounded caps only fire on an ALREADY-excessive air/contact time (0.35s), so
    # they don't dictate cadence/style (still AMP's job), only rule out the
    # permanently-lifted or permanently-planted degenerate solutions.
    max_air_time = 0.35
    max_contact_time = 0.35
    gait_bound_weight = -0.3

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/origin/.*_knee", history_length=3, track_air_time=True
    )

    expert_npz: str = EXPERT_NPZ
    reference_body = "origin"

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.8, -1.3, 0.5), lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root", asset_name="robot", env_index=0,
    )

    robot: ArticulationCfg = BINGO_V4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        self.robot.init_state = BINGO_V4_CFG.init_state.replace(
            pos=(0.0, 0.0, STAND_BASE_HEIGHT),
            joint_pos={**STAND_SOLVED, **EXPR_NEUTRAL},
        )


@configclass
class BingoDogAmpEnvCfg_PLAY(BingoDogAmpEnvCfg):
    """Deterministic playback / evaluation: small env count, no command randomization
    beyond what a caller pins via BINGO_CMD_VX/VY/YAW, fixed spawn."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.stand_prob = 0.0
        self.deterministic_spawn = True
        self.episode_length_s = 60.0

