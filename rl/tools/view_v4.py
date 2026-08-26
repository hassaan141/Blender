"""Open the v4 Bingo USD in the Isaac Sim viewport and hold it there.

Spawns the robot the same way the RL envs do (so this checks the real
articulation, not just geometry), on a ground plane with lighting, and keeps the
window open until you close it.

  --pose zero      all 21 joints at 0 (URDF zero pose)          [default]
  --pose default   the crouch in BINGO_V4_CFG.init_state
  --pose physics   let gravity act (no joint targets held)
  --usd PATH       view a different USD instead of the configured one
  --headless       no window (for CI/smoke); use --steps to bound it
"""
import argparse, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--pose", choices=("zero", "default", "physics"), default="zero")
p.add_argument("--usd", default=None)
p.add_argument("--spawn-z", type=float, default=None, help="spawn height, metres")
p.add_argument("--steps", type=int, default=0, help="0 = run until the window closes")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.bingo_v4 import BINGO_V4_CFG


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # No gravity unless we actually want physics, so the held pose stays put.
    grav = (0.0, 0.0, -9.81) if args.pose == "physics" else (0.0, 0.0, 0.0)
    sim = SimulationContext(SimulationCfg(dt=1 / 120, device=dev, gravity=grav))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    cfg = BINGO_V4_CFG.replace(prim_path="/World/Robot")
    if args.usd:
        cfg.spawn = cfg.spawn.replace(usd_path=args.usd)
    if args.spawn_z is not None:
        st = cfg.init_state
        cfg = cfg.replace(init_state=st.replace(pos=(st.pos[0], st.pos[1], args.spawn_z)))
    robot = Articulation(cfg)
    sim.reset()

    print(f"[[ v4 USD: {cfg.spawn.usd_path}", flush=True)
    print(f"[[ {len(robot.data.joint_names)} DOF: {list(robot.data.joint_names)}", flush=True)
    print(f"[[ {len(robot.data.body_names)} bodies | pose mode: {args.pose}", flush=True)

    hold = None
    if args.pose != "physics":
        hold = (torch.zeros_like(robot.data.joint_pos) if args.pose == "zero"
                else robot.data.default_joint_pos.clone())
        robot.write_joint_state_to_sim(hold, torch.zeros_like(hold))

    sim.set_camera_view(eye=(0.8, -0.8, 0.5), target=(0.0, 0.0, 0.15))
    if not args.headless:
        print("[[ viewport open - close the window to quit", flush=True)

    i = 0
    while app.is_running():
        if hold is not None:      # keep the pose exactly, no settling/collapse
            robot.write_joint_state_to_sim(hold, torch.zeros_like(hold))
        sim.step(render=not args.headless)
        robot.update(1 / 120)
        i += 1
        if args.steps and i >= args.steps:
            break
    print(f"[[ VIEW_OK stepped {i}", flush=True)
    import os
    os._exit(0)


main()
