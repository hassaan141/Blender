"""Prove URDF <-> Isaac v4 USD are the same robot at the zero pose.

Computes each link's origin-relative world position two ways at all-joints-zero:
  (1) forward kinematics straight from the v4 URDF, and
  (2) the Isaac v4 USD (spawned, joints written to 0, body_pos_w read).
Reports the per-body discrepancy. Combined with check_rig.py (URDF<->Blender),
this closes the URDF <-> Blender <-> Isaac loop.
"""
import argparse, numpy as np, xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--urdf", required=True)
p.add_argument("--rev3", action="store_true", help="check rev_3 USD instead of v4")
AppLauncher.add_app_launcher_args(p); a, _ = p.parse_known_args()
app = AppLauncher(a).app
import torch, sys
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
if a.rev3:
    from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG as ROBOT_CFG
else:
    from bingo_rl.bingo_v4 import BINGO_V4_CFG as ROBOT_CFG


def rpy(r, pch, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(pch), np.sin(pch), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def urdf_zero_fk(path):
    root = ET.parse(path).getroot(); J = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        r = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
        J[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                xyz=xyz, R=rpy(*r))
    # per-link inertial (COM) origin
    com = {}
    for ln in root.findall("link"):
        io = ln.find("inertial/origin")
        com[ln.get("name")] = np.array([float(v) for v in (io.get("xyz") or "0 0 0").split()]) if io is not None else np.zeros(3)
    frames = {"origin": (np.eye(3), np.zeros(3))}
    changed = True
    while changed:
        changed = False
        for j in J.values():
            if j["child"] in frames or j["parent"] not in frames: continue
            R, p = frames[j["parent"]]; frames[j["child"]] = (R @ j["R"], p + R @ j["xyz"]); changed = True
    link_p = {k: v[1] for k, v in frames.items()}                       # link frame
    com_p = {k: v[1] + v[0] @ com.get(k, np.zeros(3)) for k, v in frames.items()}  # link COM in world
    return link_p, com_p


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(dt=1/120, device=dev, gravity=(0.0, 0.0, 0.0)))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(ROBOT_CFG.replace(prim_path="/World/Robot")); sim.reset()
    # force zero pose
    jp = torch.zeros_like(robot.data.joint_pos); jv = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(jp, jv); robot.update(1/120)
    for _ in range(2):
        sim.step(render=False); robot.update(1/120)
    names = list(robot.data.body_names)
    bpos = robot.data.body_pos_w[0].cpu().numpy()
    bquat = robot.data.body_quat_w[0].cpu().numpy()  # wxyz
    oi = names.index("origin")
    origin = bpos[oi]
    # express every body in the BASE (origin) LOCAL frame, so a rotated base frame
    # in the USD doesn't masquerade as a geometry error
    w, x, y, z = bquat[oi]
    Rb = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                   [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                   [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    print(f"base quat (wxyz) = {np.round(bquat[oi],5)}  (identity => 1,0,0,0)")
    iso = {n: Rb.T @ (bpos[i] - origin) for i, n in enumerate(names)}   # origin-local

    link_p, com_p = urdf_zero_fk(a.urdf)
    # origin-relative on the URDF side too (subtract origin's own COM/frame consistently)
    def rel(d, ref): return {k: v - d[ref] for k, v in d.items()}
    link_p = rel(link_p, "origin"); com_p = rel(com_p, "origin")
    print("=== Isaac body_pos_w vs URDF link-frame AND URDF COM, zero pose, origin-relative (mm) ===")
    wl = wc = 0.0
    for n in sorted(iso):
        if n in link_p:
            dl = np.linalg.norm(iso[n] - link_p[n]) * 1000
            dc = np.linalg.norm(iso[n] - com_p[n]) * 1000
            wl = max(wl, dl); wc = max(wc, dc)
            print(f"  {n:20s} vs link-frame {dl:6.2f} mm   vs COM {dc:6.2f} mm")
    print(f"WORST_vs_LINKFRAME_MM {wl:.3f}")
    print(f"WORST_vs_COM_MM {wc:.3f}")
    print("CONSISTENT(COM)" if wc < 1.0 else ("CONSISTENT(link)" if wl < 1.0 else "MISMATCH"))
    import os; os._exit(0)


main()
