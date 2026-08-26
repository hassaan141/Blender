"""Improved walking config for Bingo — fixes the drag/collapse failure mode.

Separate task from the baseline (baseline files untouched). Changes vs baseline:
  1. Crouched default standing pose (baseline all-zeros could not be balanced).
  2. Real fall detection: terminate on low base height or bad orientation.
  3. Large termination penalty so falling is worse than continuing upright.
  4. feet_air_time threshold 0.5 -> 0.15 s; add feet_slide penalty.
  5. Stage-1 curriculum: forward-only command, tighter tracking std, no yaw,
     randomization disabled -> teach one deterministic config to stand & walk first.

Retrain from scratch. Do NOT start a full run until the standing test passes.
"""

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    RewardsCfg,
    TerminationsCfg,
)


# --- custom anti-tripod reward -------------------------------------------------------
def air_time_over_limit(env, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Penalize a foot for staying airborne longer than max_air_time.

    feet_air_time only *rewards* feet that touch down; a foot held permanently in the
    air (the tripod-limp failure mode) contributes nothing and so is never punished.
    This term charges the accumulating current_air_time above a cap, so a leg that
    never comes down is heavily penalized -> forces all four feet into the gait.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(air - max_air_time, min=0.0), dim=1)


def contact_time_over_limit(env, sensor_cfg: SceneEntityCfg, max_contact_time: float) -> torch.Tensor:
    """Penalize a foot for staying grounded longer than max_contact_time.

    Symmetric complement to air_time_over: without it PPO parks one foot on the
    ground as a static prop (v6 front-left: 85% grounded, 0.077 s air) while the
    other three step. Charging contact_time above a cap forces every foot to lift
    on a bounded cycle -> even, trot-like duty factor. No left/right index mapping
    needed, so it fixes whichever leg PPO tries to sacrifice.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_t = cs.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(contact_t - max_contact_time, min=0.0), dim=1)

from .agents import BingoFlatPPORunnerCfg
from .bingo import BINGO_CFG
from .env_cfg import BASE_LINK, FOOT_BODIES, BingoFlatEnvCfg

# --- Crouched standing pose (symmetric across all 4 legs) -----------------------------
# Joints: shoulder_yaw (SY_J) -> shoulder_pitch (SP_J) -> knee; joint 0 == straight leg.
# If the standing test shows knees folding the WRONG way, negate SP_J and/or knee here.
DEFAULT_JOINT_POS = {
    ".*_SY_J": 0.0,     # shoulder yaw (abduction)
    ".*_SP_J": -0.3,    # shoulder pitch (hip): NEGATIVE swings the leg forward.
    ".*_knee": 0.6,     # knee flexion. Verified standing via pose sweep (candidate B);
    "head_.*": 0.0,     # deeper crouches (e.g. -0.5/1.0) sag because the drives are weak.
    "tail_.*": 0.0,
}

# rev_3 asset: corrected symmetric leg masses, real effort values (legs 3/10, head-tail
# 1.5/8), moveable base, natural-frequency-25 / damping-1 position drives.
import copy
from pathlib import Path

_BLENDER_ROOT = Path(__file__).resolve().parents[3]  # Blender/
REV3_USD_PATH = str(
    _BLENDER_ROOT / "URDF/bingo_urdf_rev_3/urdf/bingo_scene_real_values_3/bingo_scene_real_values_3.usd"
)

BINGO_IMPROVED_CFG = copy.deepcopy(BINGO_CFG)
BINGO_IMPROVED_CFG.spawn.usd_path = REV3_USD_PATH
BINGO_IMPROVED_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.22),
    joint_pos=DEFAULT_JOINT_POS,
)
# The CLI natural-frequency conversion produced drives ~55x weaker than the rev_1 GUI
# import (measured: rev_1 knee k~2.1 vs rev_3 k~0.038), so rev_3's crouch collapsed.
# Override with rev_1's actual per-joint gains -> reproduces the working rev_1 dynamics
# with the corrected symmetric mass. (Read from 11_isaac_real_values.usd.)
BINGO_IMPROVED_CFG.actuators["legs"].stiffness = {".*_SY_J": 1.15, ".*_SP_J": 1.82, ".*_knee": 2.10}
BINGO_IMPROVED_CFG.actuators["legs"].damping = {".*_SY_J": 0.092, ".*_SP_J": 0.146, ".*_knee": 0.166}
BINGO_IMPROVED_CFG.actuators["head_tail"].stiffness = 1.5
BINGO_IMPROVED_CFG.actuators["head_tail"].damping = 0.12


@configclass
class BingoImprovedRewards(RewardsCfg):
    """Base locomotion rewards + termination penalty + feet_slide."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )

    # anti-tripod: charge any foot that stays in the air past max_air_time (kills the
    # "hold one leg up forever and drag on three" limp seen in v3).
    air_time_over = RewTerm(
        func=air_time_over_limit,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "max_air_time": 0.4,
        },
    )

    # anti-prop: charge any foot parked on the ground past max_contact_time (kills the
    # over-planted static support leg seen in v6, front-left 85% grounded).
    contact_time_over = RewTerm(
        func=contact_time_over_limit,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "max_contact_time": 0.35,
        },
    )


@configclass
class BingoImprovedTerminations(TerminationsCfg):
    """Base terminations + real fall detection (flat terrain)."""

    # neutral standing height for the default pose is ~0.202 m (measured, action=0).
    # v3 dragged at 0.144, ~6 cm below where it could stand -> raise the death floor
    # to 0.165 so the low tilted drag terminates while a real stance survives.
    low_base = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.165},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


@configclass
class BingoImprovedFlatEnvCfg(BingoFlatEnvCfg):
    rewards: BingoImprovedRewards = BingoImprovedRewards()
    terminations: BingoImprovedTerminations = BingoImprovedTerminations()

    def __post_init__(self):
        super().__post_init__()

        # crouched default pose, one deterministic config to start
        self.scene.robot = BINGO_IMPROVED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # stage-3 robustification: perturbations + varied initial conditions break
        # the lopsided limit-cycle attractor (v4/v5 leaned left, back-right foot only
        # ~25% grounded). Random lateral pushes force the policy to bear load evenly
        # on both sides -> symmetric, natural gait. This is the domain randomization
        # standard Go2 training uses that the earlier Bingo stages had switched off.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.4, 0.4)}},
        )
        # varied initial pose/velocity so no single deterministic phase is memorized
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-0.5, 0.5),
        }
        # keep add_base_mass / base_external_force_torque enabled (small base-mass
        # noise from the base cfg) for extra robustness.

        # stage-2 command: forward + bidirectional turning. Making the policy steer
        # both left and right is a strong symmetry driver -> it cannot structurally
        # favor one side (v4 leaned left: left feet planted 67-77%, right 24-43%).
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # tracking: don't over-reward nailing velocity, or PPO drags to hit it.
        # v3 used weight 2.0 / std 0.2 and learned a low tilted 3-legged drag that
        # tracked vx perfectly; loosen so gait-quality terms can compete.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # step-promoting rewards: reward every foot cycling (raises the opportunity
        # cost of holding a leg up), and punish sliding harder.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_slide.weight = -0.4
        self.rewards.air_time_over.weight = -1.0
        # tighten the airborne cap: v4's right-rear foot averaged 0.47 s aloft (only
        # 24% grounded); 0.35 s pulls under-planted feet down toward even support.
        self.rewards.air_time_over.params["max_air_time"] = 0.35
        # anti-prop cap (v7): forces the over-planted foot to lift on a bounded cycle.
        self.rewards.contact_time_over.weight = -1.0
        self.rewards.contact_time_over.params["max_contact_time"] = 0.35

        # soften energy penalties while gait forms; keep base level (kills 18° tilt).
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.flat_orientation_l2.weight = -2.5


@configclass
class BingoImprovedFlatEnvCfg_PLAY(BingoImprovedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class BingoImprovedStandTestCfg_PLAY(BingoImprovedFlatEnvCfg_PLAY):
    """Standing test: action scale 0 -> robot can only hold DEFAULT_JOINT_POS.

    Terminations disabled so we can watch it settle. If it stays up and the feet
    land in a rectangle under the body, the pose is good. If it collapses, the
    pose (joint signs / height / gains) is wrong, not the rewards.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 1.5
        self.actions.joint_pos.scale = 0.0
        self.terminations.low_base = None
        self.terminations.bad_orientation = None
        self.terminations.base_contact = None


@configclass
class BingoImprovedFlatPPORunnerCfg(BingoFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "bingo_improved_walking"


# =====================================================================================
# ROUGH TERRAIN (blind, warm-started from the v7 flat policy)
# =====================================================================================
# Terrain scaled for a 0.2 m / 2.5 kg robot. The stock ROUGH_TERRAINS_CFG (steps up to
# 0.23 m, boxes 0.2 m, slopes 0.4 rad) is sized for a 0.5 m ANYmal and would faceplant
# Bingo instantly. Obstacles here are a fraction of leg length; NO stairs. `curriculum=
# True` means row 0 = easy end of each range, top row = hard end; the terrain_levels
# curriculum promotes robots that traverse far enough -> difficulty ramps automatically
# so we train ~once instead of hand-iterating difficulty.
BINGO_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,   # difficulty levels the curriculum climbs
    num_cols=20,   # variety within a level
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.005, 0.03), noise_step=0.005, border_width=0.25
        ),
        # grid_width must NOT evenly divide size (8.0) or the internal border width
        # is 0 and generation errors: int(8/0.3)=26, 26*0.3=7.8 -> border 0.2 m > 0.
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.04), platform_width=1.5
        ),
        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=1.5, border_width=0.25
        ),
        "gentle_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=1.5, border_width=0.25
        ),
    },
)


@configclass
class BingoImprovedRoughEnvCfg(BingoImprovedFlatEnvCfg):
    """Rough-terrain walking: inherits the v7 flat gait design, re-enables scaled
    terrain + the terrain-level curriculum, and relaxes the flat-only penalties that
    fight rough ground. Blind (no height scanner) so the observation vector stays
    identical to the flat policy -> the v7 checkpoint can be warm-started directly.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- re-enable scaled procedural terrain + curriculum (flat cfg stripped it) ---
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BINGO_ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 1  # everyone starts easy, climbs up
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # stays blind: height_scanner / height_scan obs remain None (from flat cfg) so
        # obs dim == flat's 58 and the v7 weights load without a shape mismatch.

        # --- relax the flat-only penalties that punish correct rough behavior ---
        # on slopes/steps the body legitimately tilts; -2.5 would fight climbing.
        self.rewards.flat_orientation_l2.weight = -1.0
        # a fixed WORLD-height floor false-fires in dips / behind bumps and stalls
        # learning; drop it low (real collapses still trip bad_orientation/base_contact).
        self.terminations.low_base.params["minimum_height"] = 0.10
        # terrain forces variable step timing -> loosen the air/contact caps a touch.
        self.rewards.air_time_over.params["max_air_time"] = 0.45
        self.rewards.contact_time_over.params["max_contact_time"] = 0.45

        # --- robustness for varied ground: randomize friction (was fixed 0.8/0.6) ---
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.9)


@configclass
class BingoImprovedRoughEnvCfg_PLAY(BingoImprovedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # a small fixed spread of terrains to watch (no curriculum promotion)
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class BingoImprovedRoughPPORunnerCfg(BingoImprovedFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # same [128,128,128] net as the flat policy -> v7 weights warm-start cleanly.
        self.max_iterations = 1500
        self.experiment_name = "bingo_improved_rough"
