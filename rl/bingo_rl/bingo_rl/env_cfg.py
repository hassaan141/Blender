"""Velocity-tracking locomotion envs for Bingo, adapted from the Unitree Go2 template."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .bingo import BINGO_CFG

BASE_LINK = "origin"  # Bingo's root link name in the USD
FOOT_BODIES = ".*_knee"  # lower-leg link includes the foot; it is the contact body


@configclass
class BingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + BASE_LINK
        # Bingo is much smaller than the terrains are tuned for
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.03)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005

        # actions: legs only (head/tail locked by their drives)
        self.actions.joint_pos.joint_names = ["(fl|fr|bl|br)_.*"]
        self.actions.joint_pos.scale = 0.25

        # commands scaled to a 2.5 kg, short-legged robot
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # events
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_LINK
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_LINK
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_LINK


@configclass
class BingoFlatEnvCfg(BingoRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class BingoFlatEnvCfg_PLAY(BingoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
