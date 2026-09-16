import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app
import torch, os, sys
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
sim = SimulationContext(SimulationCfg(dt=1/120, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(BINGO_V4_CFG.replace(prim_path="/World/Robot"))
sim.reset()
print("V4_NJOINTS", len(robot.data.joint_names))
print("V4_JOINTS", list(robot.data.joint_names))
for _ in range(3):
    robot.set_joint_position_target(robot.data.default_joint_pos); robot.write_data_to_sim(); sim.step(render=False); robot.update(1/120)
print("V4_STEP_OK root_z", float(robot.data.root_pos_w[0,2]))
print("V4_SMOKE_PASSED")
os._exit(0)
