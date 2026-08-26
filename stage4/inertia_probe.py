"""Measure the EFFECTIVE inertia each joint actually presents in the sim:
zero the PD gains, apply a known constant effort, measure angular acceleration.
I_eff = tau / alpha.  Compare against the URDF subtree inertia."""
import argparse, sys, os
import numpy as np
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--tau", type=float, default=0.05)
AppLauncher.add_app_launcher_args(p); args, _ = p.parse_known_args()
app = AppLauncher(args).app
import torch, isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
PROBE = ["l_ear_pitch", "l_ear_roll", "fl_knee", "head_yaw", "tail_pitch"]
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
dt = 1/120
sim = SimulationContext(SimulationCfg(dt=dt, device=dev, gravity=(0.0, 0.0, 0.0)))
cfg = BINGO_V4_CFG.replace(prim_path="/World/Robot")
cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
robot = Articulation(cfg); sim.reset()
names = list(robot.data.joint_names); n = len(names)
# zero the PD so only our effort acts
z = torch.zeros((1, n), device=dev)
robot.write_joint_stiffness_to_sim(z.clone()); robot.write_joint_damping_to_sim(z.clone())
print("[[ gains zeroed:", float(robot.data.joint_stiffness[0].abs().max()),
      float(robot.data.joint_damping[0].abs().max()), flush=True)
for jn in PROBE:
    j = names.index(jn)
    robot.write_joint_state_to_sim(z.clone(), z.clone())
    eff = z.clone(); eff[0, j] = args.tau
    vs = []
    for s in range(30):
        robot.set_joint_effort_target(eff); robot.write_data_to_sim()
        sim.step(render=False); robot.update(dt)
        vs.append(float(robot.data.joint_vel[0, j]))
    vs = np.array(vs)
    alpha = np.polyfit(np.arange(len(vs))*dt, vs, 1)[0]
    I = args.tau/alpha if abs(alpha) > 1e-12 else float('inf')
    print(f"[[ {jn:14s} tau {args.tau:.3f} -> alpha {alpha:10.3f} rad/s^2  "
          f"I_eff {I:.3e} kg m^2   (v after 30 steps {vs[-1]:+.4f})", flush=True)
os._exit(0)
