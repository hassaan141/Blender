"""Stage 4 sanity 1 - isolated ear joint step response, fixed base, no gravity.

For each ear joint: hold every other joint at 0, command that one joint to +0.3 rad,
simulate 3 s, and log target/actual/velocity/effort. Expected: 0 -> 0.3 -> settle.

Also dumps what the USD actually built for those joints (drive type, limits, whether
the revolute is limited or free-spinning), because the URDF declares the two
*_ear_pitch joints as type="continuous".

  ./isaaclab.sh -p stage4/ear_test.py --headless
"""
import argparse, sys, os
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--angle", type=float, default=0.3)
p.add_argument("--seconds", type=float, default=3.0)
p.add_argument("--dt", type=float, default=1.0 / 120.0)
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from pxr import Usd, UsdPhysics, PhysxSchema

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.bingo_v4 import BINGO_V4_CFG

EARS = ["l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]


def dump_usd_joints():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    print("\n[[ ---- what the USD actually built for the ear joints ----", flush=True)
    for prim in stage.Traverse():
        n = prim.GetName()
        if n not in EARS:
            continue
        t = prim.GetTypeName()
        lo = hi = None
        if prim.HasAttribute("physics:lowerLimit"):
            lo = prim.GetAttribute("physics:lowerLimit").Get()
        if prim.HasAttribute("physics:upperLimit"):
            hi = prim.GetAttribute("physics:upperLimit").Get()
        drive = "none"
        for api in prim.GetAppliedSchemas():
            if "DriveAPI" in api:
                drive = api
        stiff = prim.GetAttribute("drive:angular:physics:stiffness").Get() \
            if prim.HasAttribute("drive:angular:physics:stiffness") else None
        damp = prim.GetAttribute("drive:angular:physics:damping").Get() \
            if prim.HasAttribute("drive:angular:physics:damping") else None
        maxf = prim.GetAttribute("drive:angular:physics:maxForce").Get() \
            if prim.HasAttribute("drive:angular:physics:maxForce") else None
        limited = "LIMITED" if (lo is not None and hi is not None
                                and not (lo <= -1e5 or hi >= 1e5)) else "FREE/UNLIMITED"
        print(f"     {n:14s} type={t:22s} limits=[{lo},{hi}] -> {limited}", flush=True)
        print(f"        drive={drive} stiffness={stiff} damping={damp} maxForce={maxf}", flush=True)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(dt=args.dt, device=dev, gravity=(0.0, 0.0, 0.0)))
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    cfg = BINGO_V4_CFG.replace(prim_path="/World/Robot")
    # fixed base: isolate the joint, no floating-base reaction
    try:
        cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
        fixed = True
    except Exception as e:
        print(f"[[ note: could not fix root via cfg ({e}); will pin the root each step", flush=True)
        fixed = False
    robot = Articulation(cfg)
    sim.reset()
    dump_usd_joints()

    names = list(robot.data.joint_names)
    n_dof = len(names)
    # what did IsaacLab ACTUALLY push to the sim (vs what the USD authored)?
    def col(attr):
        try:
            return robot.data.__getattribute__(attr)[0].cpu().numpy()
        except Exception:
            return None
    print("\n[[ ---- gains as applied at runtime ----", flush=True)
    for a_ in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction"):
        v = col(a_)
        if v is not None:
            print(f"     {a_:18s} " + "  ".join(f"{names[names.index(e)]}={v[names.index(e)]:.4g}"
                                                for e in EARS), flush=True)
    steps = int(args.seconds / args.dt)
    root_pose = robot.data.root_state_w[:, :7].clone()

    print(f"\n[[ fixed base={fixed} | gravity OFF | {steps} steps @ {1/args.dt:g} Hz", flush=True)
    results = {}
    for ear in EARS:
        j = names.index(ear)
        zero = torch.zeros((1, n_dof), device=dev)
        robot.write_joint_state_to_sim(zero.clone(), zero.clone())
        if not fixed:
            robot.write_root_link_pose_to_sim(root_pose)
            robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
        # respect the joint's own URDF range: l_ear_roll is [-1.5, 0], so a
        # +0.3 command is out of range and would just pin at the limit.
        ang = -args.angle if ear == "l_ear_roll" else args.angle
        tgt = zero.clone()
        tgt[0, j] = ang
        log = []
        for s in range(steps):
            robot.set_joint_position_target(tgt)
            robot.write_data_to_sim()
            if not fixed:
                robot.write_root_link_pose_to_sim(root_pose)
                robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
            sim.step(render=not args.headless)
            robot.update(args.dt)
            q = float(robot.data.joint_pos[0, j])
            v = float(robot.data.joint_vel[0, j])
            try:
                e = float(robot.data.applied_torque[0, j])
            except Exception:
                e = float("nan")
            log.append((s * args.dt, q, v, e))
        L = np.array(log)
        settle = L[-24:, 1].mean()          # last 0.2 s
        results[ear] = (settle, L, ang)
        turns = (L[:, 1].max() - L[:, 1].min()) / (2 * np.pi)
        verdict = "OK" if abs(settle - ang) < 0.05 else "BAD"
        print(f"\n[[ {ear}: target {ang:+.3f} -> settled {settle:+.4f} rad  [{verdict}]", flush=True)
        print(f"     range {L[:,1].min():+.3f}..{L[:,1].max():+.3f} rad "
              f"({turns:.2f} revolutions) | |vel| max {np.abs(L[:,2]).max():.2f} rad/s "
              f"| |effort| max {np.nanmax(np.abs(L[:,3])):.3f}", flush=True)
        for t in (0.1, 0.5, 1.0, 2.0, 3.0):
            k = min(int(t / args.dt), steps - 1)
            print(f"       t={t:4.1f}s  q={L[k,1]:+8.4f}  v={L[k,2]:+8.3f}  eff={L[k,3]:+7.3f}", flush=True)

    print("\n[[ ===== EAR TEST SUMMARY =====", flush=True)
    bad = [e for e in EARS if abs(results[e][0] - results[e][2]) >= 0.05]
    for e in EARS:
        print(f"     {e:14s} settled {results[e][0]:+.4f} (want {results[e][2]:+.3f})", flush=True)
    print(f"[[ EAR_TEST {'PASS' if not bad else 'FAIL: ' + str(bad)}", flush=True)
    os.makedirs("/home/hassaan/Bingo/Blender/stage4/out", exist_ok=True)
    np.savez("/home/hassaan/Bingo/Blender/stage4/out/ear_test.npz",
             **{e: results[e][1] for e in EARS})
    os._exit(0)


main()
