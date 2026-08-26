"""Articulation config for the Bingo quadruped (SolidWorks URDF -> Isaac Sim USD)."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Blender/ is the project root; assets live under Blender/URDF/. bingo.py is at
# Blender/rl/bingo_rl/bingo_rl/bingo.py -> parents[3] == Blender.
_BLENDER_ROOT = Path(__file__).resolve().parents[3]
BINGO_USD_PATH = str(_BLENDER_ROOT / "URDF/bingo_urdf_rev_1/urdf/11_isaac_real_values/11_isaac_real_values.usd")

BINGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BINGO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # standing height with straight legs is ~0.16 m; spawn slightly above
        pos=(0.0, 0.0, 0.20),
        # TODO: tune a crouched default pose once knee bend direction is verified in the GUI
        joint_pos={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # placeholder servo specs; update when real torque/speed numbers are known
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["(fl|fr|bl|br)_.*"],
            effort_limit_sim=3.0,
            velocity_limit_sim=10.0,
            # None -> keep the per-joint stiffness/damping authored in the USD by the importer
            stiffness=None,
            damping=None,
        ),
        # head/tail are not part of the action space; drives hold them at 0
        "head_tail": ImplicitActuatorCfg(
            joint_names_expr=["head_.*", "tail_.*"],
            effort_limit_sim=1.5,
            velocity_limit_sim=8.0,
            stiffness=None,
            damping=None,
        ),
    },
)
