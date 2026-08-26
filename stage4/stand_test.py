"""Stage 4 sanity 2 - static standing hold, judged against REAL collision geometry.

Gravity ON, all 21 position targets constant, nothing teleported after placement.
Contact/foot-height/penetration all come from stage4/contact_model.py (convex hulls
of the actual collision meshes), not from a fixed paw point.

Spawn height is solved from the hulls so the lowest paw geometry starts just above
the floor for the commanded pose.

  ./isaaclab.sh -p stage4/stand_test.py --headless
"""
import argparse, sys, os
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=5.0)
p.add_argument("--dt", type=float, default=1.0 / 120.0)
p.add_argument("--drop", type=float, default=0.002)
p.add_argument("--pose", default="legacy", choices=("legacy", "solved", "zero"))
p.add_argument("--knee", type=float, default=None, help="override |knee| angle")
p.add_argument("--sp", type=float, default=None, help="override |SP| angle")
p.add_argument("--explicit", action="store_true",
               help="use IdealPDActuator (torque computed in python, effort limit "
                    "actually enforced) instead of the implicit PhysX drive")
p.add_argument("--kp", type=float, default=None,
               help="override LEG stiffness (0 = no position drive at all)")
p.add_argument("--fixbase", action="store_true",
               help="pin the root: isolates leg control from base/contact dynamics")
p.add_argument("--effort", type=float, default=None,
               help="override LEG effort limit (N m) to test whether 3.0 binds")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage4")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
from v4_kinematics import V4Kin, LEGS, axis_rot
from contact_model import ContactModel, quat_to_R

URDF = ("/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/"
        "bingo_urdf_w_ear_joints_physics.urdf")
CONTACT_H = 0.005


# Solved by stage4/solve_stand_pose.py against the REAL collision hulls: each paw's
# lowest hull point sits exactly 0.180 m below its own SY pivot (paw spread 0.00 mm),
# which is the documented 0.19-0.20 m design stance. The inherited rev_3 pose
# (SP +/-0.3, knee +/-0.6) was NOT a valid stance - it drove fr_SP_J into its +1.56
# limit and jammed the leg against the floor.
# Neutral-SY solve (residual 0.00000, paws exactly level at -0.180 m). Holding the
# paw under the SY pivot forced 0.19-0.26 rad of permanent adduction, which needed
# continuous SY torque and toppled the robot sideways; the SP pivot is ~46 mm
# outboard of SY, so SY=0 is the stance the leg geometry is designed for and it
# also gives a wider (y = +/-53 mm vs +/-23 mm) support polygon.
STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}


def stand_pose(sp_mag, knee_mag):
    if args.pose == "zero":
        # The URDF / Blender rig REST pose: every joint at 0, legs straight.
        # check_floor on Bingo_V4_AnimatorRig showed the paws sit +0.8 mm off the
        # floor here, so it is a genuine standing pose, not a crouch.
        return {}
    if args.pose == "legacy":
        return {"fl_SP_J": -sp_mag, "bl_SP_J": -sp_mag, "fr_SP_J": +sp_mag, "br_SP_J": -sp_mag,
                "fl_knee": +knee_mag, "bl_knee": +knee_mag, "fr_knee": -knee_mag, "br_knee": -knee_mag}
    return dict(STAND_SOLVED)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    kin = V4Kin(URDF)
    cm = ContactModel()

    sp_mag = 0.3 if args.sp is None else args.sp
    knee_mag = 0.6 if args.knee is None else args.knee
    STAND = stand_pose(sp_mag, knee_mag)

    # ---- spawn height from the true hulls, not from a fixed paw point ----------
    def leg_chain_pose(leg, q):
        R, p = np.eye(3), np.zeros(3)
        for name, qi in zip(kin.leg_chain(leg), q):
            J = kin.j[name]
            p = p + R @ J["xyz"]
            R = R @ J["R"] @ axis_rot(J["axis"], qi)
        return R, p                       # knee link frame in base coords

    lows = []
    for leg in LEGS:
        q = np.array([0.0, STAND.get(f"{leg}_SP_J", 0.0), STAND.get(f"{leg}_knee", 0.0)])
        R, p = leg_chain_pose(leg, q)
        w = cm.hull[f"{leg}_knee"] @ R.T + p
        lows.append(w[:, 2].min())
    spawn_z = -min(lows) + args.drop
    print(f"[[ pose SP +/-{sp_mag:.2f}  knee +/-{knee_mag:.2f}", flush=True)
    print(f"[[ true paw hull z in base frame: {np.round(lows, 4)}", flush=True)
    print(f"[[ spawn height {spawn_z:.4f} m  (lowest paw {args.drop*1000:.1f} mm above floor)", flush=True)

    sim = SimulationContext(SimulationCfg(dt=args.dt, device=dev, gravity=(0.0, 0.0, -9.81)))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    _cfg = BINGO_V4_CFG.replace(prim_path="/World/Robot")
    if args.explicit:
        # The implicit drive never enforces effort_limit: 3.0 / 8.0 / 15.0 N m all
        # produced byte-identical motion. IdealPDActuator computes tau in python and
        # applies it as a real effort, so both Kp and the ceiling actually bite.
        from isaaclab.actuators import IdealPDActuatorCfg
        import copy as _c
        old = _cfg.actuators["legs"]
        _cfg.actuators["legs"] = IdealPDActuatorCfg(
            joint_names_expr=old.joint_names_expr,
            effort_limit=getattr(old, "effort_limit_sim", None) or 3.0,
            velocity_limit=getattr(old, "velocity_limit_sim", None) or 10.0,
            stiffness=_c.deepcopy(old.stiffness), damping=_c.deepcopy(old.damping),
            armature=_c.deepcopy(getattr(old, "armature", 0.0)))
        print("[[ legs -> IdealPDActuator (explicit torque, enforced effort)", flush=True)
    if args.fixbase:
        _cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
        print("[[ base FIXED (leg-control isolation)", flush=True)
    if args.effort is not None:
        for attr in ("effort_limit_sim", "effort_limit"):
            if hasattr(_cfg.actuators["legs"], attr):
                try:
                    setattr(_cfg.actuators["legs"], attr, args.effort)
                except Exception as e:
                    print(f"[[ set {attr} failed: {e}", flush=True)
        print(f"[[ LEG effort -> {args.effort} N m (cfg pre-build)", flush=True)
    if args.kp is not None:
        _cfg.actuators["legs"].stiffness = args.kp
        _cfg.actuators["legs"].damping = args.kp * 0.02
        print(f"[[ LEG Kp -> {args.kp}  Kd -> {args.kp*0.02} (cfg pre-build)", flush=True)
    robot = Articulation(_cfg)
    sim.reset()

    names = list(robot.data.joint_names)
    bodies = list(robot.data.body_names)
    leg_j = [i for i, n in enumerate(names) if n[:3] in ("fl_", "fr_", "bl_", "br_")]
    knee_j = [i for i, n in enumerate(names) if n.endswith("_knee")]
    lim_lo = np.array([kin.j[n]["lo"] for n in names])
    lim_hi = np.array([kin.j[n]["hi"] for n in names])
    eff_lim = np.full(len(names), np.inf)
    for act in robot.actuators.values():
        ids = act.joint_indices
        ids = np.arange(len(names)) if isinstance(ids, slice) else np.atleast_1d(
            ids.cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)).reshape(-1)
        e = act.effort_limit
        e = e.detach().cpu().numpy().reshape(-1) if isinstance(e, torch.Tensor) else np.atleast_1d(e).reshape(-1)
        for k, j in enumerate(ids):
            eff_lim[int(j)] = e[k] if e.size > 1 else e[0]

    print(f"[[ effort limits as built: legs={eff_lim[names.index('fl_knee')]:.2f} "
          f"head={eff_lim[names.index('head_pitch_joint')]:.2f}", flush=True)
    print(f"[[ gains: Kp " + " ".join(
        f"{n}={float(robot.data.joint_stiffness[0, names.index(n)]):.2f}"
        for n in ("fl_SY_J", "fl_SP_J", "fl_knee")) +
        " | Kd " + " ".join(
        f"{float(robot.data.joint_damping[0, names.index(n)]):.3f}"
        for n in ("fl_SY_J", "fl_SP_J", "fl_knee")), flush=True)

    tgt = torch.zeros((1, len(names)), device=dev)
    for jn, v in STAND.items():
        tgt[0, names.index(jn)] = v
    pose = torch.tensor([[0.0, 0.0, spawn_z, 1.0, 0.0, 0.0, 0.0]], device=dev)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
    robot.write_joint_state_to_sim(tgt.clone(), torch.zeros_like(tgt))
    robot.set_joint_position_target(tgt); robot.write_data_to_sim()

    steps = int(args.seconds / args.dt)
    RP, RQ, Q, PZ, TQ, OTH = [], [], [], [], [], []
    for s in range(steps):
        robot.set_joint_position_target(tgt)
        robot.write_data_to_sim()
        sim.step(render=not args.headless)
        robot.update(args.dt)
        bp = robot.data.body_pos_w[0].cpu().numpy()
        bq = robot.data.body_quat_w[0].cpu().numpy()
        pz, ncon, other = cm.support_summary(bp, bq, bodies, CONTACT_H)
        RP.append(robot.data.root_pos_w[0].cpu().numpy().copy())
        RQ.append(robot.data.root_quat_w[0].cpu().numpy().copy())
        Q.append(robot.data.joint_pos[0].cpu().numpy().copy())
        PZ.append(pz); OTH.append(other)
        try:
            TQ.append(robot.data.applied_torque[0].cpu().numpy().copy())
        except Exception:
            TQ.append(np.zeros(len(names)))

    RP = np.array(RP); RQ = np.array(RQ); Q = np.array(Q); PZ = np.array(PZ); TQ = np.array(TQ)
    want = tgt[0].cpu().numpy()
    qerr = np.abs(Q - want[None, :])
    leg_err = qerr[:, leg_j]
    tilt = np.degrees(np.arccos(np.clip([quat_to_R(q)[2, 2] for q in RQ], -1, 1)))
    ncon = (PZ < CONTACT_H).sum(1)
    at_lim = (Q <= lim_lo[None, :] + 1e-3) | (Q >= lim_hi[None, :] - 1e-3)
    sat = np.abs(TQ) >= (eff_lim[None, :] - 1e-3)
    last = slice(-int(1.0 / args.dt), None)          # final second

    # torso = origin link lowest collision z
    oi = bodies.index("origin")
    torso_z = np.array([cm.lowest_z("origin", RP[i], RQ[i]) for i in range(len(RP))])

    print(f"\n[[ ===== STANDING HOLD ({args.seconds:g} s) =====", flush=True)
    print(f"[[ leg tracking : mean {leg_err.mean():.4f} rad | final-second mean "
          f"{leg_err[last].mean():.4f} | max {leg_err.max():.4f}", flush=True)
    print(f"[[ all-joint    : mean {qerr.mean():.4f} rad  max {qerr.max():.4f}", flush=True)
    print(f"[[ root drift   : xy {np.linalg.norm(RP[-1,:2]-RP[0,:2])*1000:.2f} mm | "
          f"z {RP[0,2]:.4f} -> {RP[-1,2]:.4f} m | last-second z drift "
          f"{(RP[-1,2]-RP[-int(1/args.dt),2])*1000:+.3f} mm", flush=True)
    print(f"[[ body tilt    : final {tilt[-1]:.2f} deg  max {tilt.max():.2f} | "
          f"last-second change {tilt[-1]-tilt[-int(1/args.dt)]:+.3f} deg", flush=True)
    print(f"[[ paw contacts : final {int(ncon[-1])}/4  last-second mean {ncon[last].mean():.2f}/4"
          f" | final paw z {np.round(PZ[-1]*1000,2)} mm", flush=True)
    print(f"[[ torso lowest : {torso_z[-1]*1000:+.1f} mm (must stay > 0)", flush=True)
    nonpaw = OTH[-1]
    print(f"[[ non-paw links touching floor: { {k: round(v*1000,1) for k,v in nonpaw.items()} if nonpaw else 'none'}", flush=True)
    kl = [names[j] for j in knee_j if at_lim[-1, j]]
    print(f"[[ knees at limit: {kl if kl else 'none'}", flush=True)
    st = [names[j] for j in range(len(names)) if sat[last, j].mean() > 0.5]
    print(f"[[ sustained torque saturation (>50% of final second): {st if st else 'none'}", flush=True)
    worst = np.argsort(qerr[-1])[::-1][:6]
    print(f"[[ worst joints (final):", flush=True)
    for j in worst:
        print(f"     {names[j]:18s} err {qerr[-1,j]:.4f} rad  torque {TQ[-1,j]:+.3f}"
              f"/{eff_lim[j]:.1f}", flush=True)

    # ---- exit criteria ---------------------------------------------------------
    C = {}
    C["stable >=5 s"] = args.seconds >= 5.0
    C["4 paw contacts"] = ncon[last].mean() >= 3.99
    C["torso off ground"] = torso_z[last].min() > 0.0
    C["no knee at limit"] = not any(at_lim[last][:, j].any() for j in knee_j)
    C["leg err < 0.05 rad"] = leg_err[last].mean() < 0.05
    C["no sustained saturation"] = len(st) == 0
    C["no root drift"] = abs(RP[-1, 2] - RP[-int(1/args.dt), 2]) < 0.002 and \
                         np.linalg.norm(RP[-1, :2] - RP[-int(1/args.dt), :2]) < 0.002
    C["no progressive tilt"] = abs(tilt[-1] - tilt[-int(1/args.dt)]) < 0.5
    print(f"\n[[ ---- EXIT CRITERIA ----", flush=True)
    for k, v in C.items():
        print(f"     [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    ok = all(C.values())
    print(f"[[ STAND_TEST {'PASS' if ok else 'FAIL'}", flush=True)

    os.makedirs("/home/hassaan/Bingo/Blender/stage4/out", exist_ok=True)
    np.savez("/home/hassaan/Bingo/Blender/stage4/out/stand_test.npz",
             root_pos=RP, root_quat=RQ, q=Q, paw_z=PZ, torque=TQ,
             joint_names=np.array(names), target=want, torso_z=torso_z)
    if not args.headless:
        print("[[ close the window to quit", flush=True)
        while app.is_running():
            robot.set_joint_position_target(tgt); robot.write_data_to_sim()
            sim.step(render=True); robot.update(args.dt)
    os._exit(0 if ok else 1)


main()
