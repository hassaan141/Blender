"""Stage 4 baseline - can the v4 robot actually TRACK the reference under physics?

Unlike Stage 3 (rl/tools/replay_v4.py) nothing is teleported here. The robot is
initialised from reference frame 0 exactly once; after that the ONLY input is a
joint position target per control step, and Isaac's physics decides what happens.

    q_target = reference_q[t]
    robot.set_joint_position_target(q_target)   # name-mapped, validated in Stage 3
    robot.write_data_to_sim()
    sim.step()   x decimation

Gravity, collisions and contacts are all ON. Joint mapping reuses the exact
name-based remap proven in Stage 3 (Isaac orders DOFs breadth-first, the .npz is
per-leg, so positional indexing would scramble the legs).

Contacts: the v4 USD carries no contact-reporter API on its links (same reason
track_v4 skips ContactSensor), so foot contact is measured geometrically from the
rigid-paw point of each knee link - the same PAW_CONTACT_LOCAL the retarget used.

  ./isaaclab.sh -p rl/tools/track_v4_physics.py --motion motions/eccentric_v4.npz --headless
"""
import argparse, sys, os
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--motion", required=True)
p.add_argument("--out", default="/home/hassaan/Bingo/Blender/stage4/out")
p.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
p.add_argument("--loops", type=int, default=1, help="GUI only: replay the clip N times")
p.add_argument("--hold", action="store_true",
               help="zero-order-hold the target across the decimation window "
                    "(old behaviour). Default interpolates to the physics rate: a "
                    "24 Hz ZOH target against Kp=120 is a ~0.33 rad step, i.e. an "
                    "impulsive kick 24x/second.")
p.add_argument("--vel-ff", action="store_true",
               help="feed qdot_ref as the joint velocity target. IdealPDActuator computes "
                    "tau = Kp*(q_des-q) + Kd*(qd_des-qd); without this qd_des is 0, so Kd "
                    "damps toward STANDSTILL rather than toward the reference velocity. "
                    "Theoretically correct, but OFF by default because it MEASURABLY "
                    "REGRESSES on this robot: Kd*qd_ref (Kd 4.0 x up to 5 rad/s = 20 N m) "
                    "is far past the 3.0 N m ceiling, so the feed-forward term alone pins "
                    "the drive at saturation and turns the PD into bang-bang. Measured on "
                    "Timid at Kd 2/4/4: fall frame 39 -> 38, mean joint error 0.0242 -> "
                    "0.0340 rad. Revisit if the effort ceiling ever rises.")
p.add_argument("--kp-scale", type=float, default=None, help="diagnostic: scale every Kp")
p.add_argument("--kd", default=None,
               help="override damping, 'SY=..,SP=..,knee=..,head=..,tail=..'")
p.add_argument("--armature", default=None,
               help="override armature, 'legs=..,head=..,tail=..,ears=..'")
p.add_argument("--solver", default=None, help="pos,vel solver iteration counts")
p.add_argument("--friction", type=float, default=None,
               help="ground static friction (dynamic = 0.85x). IsaacLab defaults to "
                    "0.5/0.5, which for a 2.5 kg robot on rubber paws may be low "
                    "enough to let planted paws slide.")
p.add_argument("--settle", type=int, default=0,
               help="physics steps to settle holding frame 0 before tracking starts")
p.add_argument("--diverge-rad", type=float, default=0.20,
               help="joint error that counts as meaningful divergence")
p.add_argument("--diverge-m", type=float, default=0.05,
               help="root position error that counts as meaningful divergence")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
from v4_kinematics import LEGS
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage4")
from contact_model import ContactModel

CONTACT_H = 0.005          # paw within 5 mm of the floor counts as contact


def quat_angle_deg(qa, qb):
    d = np.abs(np.sum(qa * qb, axis=-1) /
               (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1) + 1e-12))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = np.load(args.motion, allow_pickle=True)
    dof_ref = m["dof_positions"].astype(np.float64)
    if "root_pos" in m.files:
        root_ref = m["root_pos"].astype(np.float64)
        quat_ref = m["root_quat"].astype(np.float64)
    else:                                    # bake_conform schema
        root_ref = m["body_positions"][:, 0].astype(np.float64)
        quat_ref = m["body_rotations"][:, 0].astype(np.float64)
    # qdot_ref: exported by the baker as d(dof_positions)/dt. Verified against a
    # central difference of dof_positions (max discrepancy 1.4e-6 rad/s), so it is
    # the reference velocity, not a stale or independently-authored field.
    vel_ref = (m["dof_velocities"].astype(np.float64) if "dof_velocities" in m.files
               else np.gradient(dof_ref, 1.0 / float(m["fps"]), axis=0))
    npz_names = [str(n) for n in m["dof_names"]]
    fps = float(m["fps"]); T = dof_ref.shape[0]
    control_dt = 1.0 / fps
    decim = max(1, int(round(control_dt / args.physics_dt)))

    # ---- FULL physics: gravity, ground, collisions, contacts -------------------
    _sc = SimulationCfg(dt=args.physics_dt, device=dev, gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(_sc)
    if args.friction is not None:
        _gp = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args.friction, dynamic_friction=args.friction * 0.85))
        print(f"[[ ground friction set to {args.friction}/{args.friction*0.85:.2f}", flush=True)
    else:
        _gp = sim_utils.GroundPlaneCfg()
    _gp.func("/World/ground", _gp)
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    cfg = BINGO_V4_CFG.replace(prim_path="/World/Robot")
    if args.kd:
        kv = dict(kk.split("=") for kk in args.kd.split(","))
        for j, v in kv.items():
            key = {"SY": ".*_SY_J", "SP": ".*_SP_J", "knee": ".*_knee",
                   "head": "head_.*", "tail": "tail_.*"}[j]
            grp = "legs" if j in ("SY", "SP", "knee") else "head_tail"
            cfg.actuators[grp].damping[key] = float(v)
        print(f"[[ Kd override {kv}", flush=True)
    if args.kp_scale:
        for g in cfg.actuators.values():
            if isinstance(g.stiffness, dict):
                g.stiffness = {k: v * args.kp_scale for k, v in g.stiffness.items()}
        print(f"[[ Kp scaled x{args.kp_scale}", flush=True)
    if args.armature:
        av = dict(kk.split("=") for kk in args.armature.split(","))
        for j, v in av.items():
            if j == "legs":
                cfg.actuators["legs"].armature = float(v)
            elif j == "ears":
                cfg.actuators["ears"].armature = float(v)
            else:
                cfg.actuators["head_tail"].armature[f"{j}_.*"] = float(v)
        print(f"[[ armature override {av}", flush=True)
    robot = Articulation(cfg)
    sim.reset()

    isaac_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    missing = [n for n in npz_names if n not in isaac_names]
    if missing:
        print(f"[[ FAIL joints missing in Isaac: {missing}", flush=True); os._exit(1)
    idx = torch.tensor([isaac_names.index(n) for n in npz_names], device=dev, dtype=torch.long)
    knee_i = {l: body_names.index(f"{l}_knee") for l in LEGS}
    cm = ContactModel()   # true collision-hull contact, not a fixed paw point

    # effort / velocity limits, per joint, from the actuator groups
    def to_np(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().reshape(-1)
        return np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)

    eff_lim = np.full(len(isaac_names), np.inf)
    vel_lim = np.full(len(isaac_names), np.inf)
    for act in robot.actuators.values():
        ids = act.joint_indices
        ids = (np.arange(len(isaac_names)) if isinstance(ids, slice)
               else to_np(ids).astype(int))
        e, v = to_np(act.effort_limit), to_np(act.velocity_limit)
        for k, j in enumerate(ids):
            eff_lim[int(j)] = e[k] if e.size > 1 else e[0]
            vel_lim[int(j)] = v[k] if v.size > 1 else v[0]

    print(f"[[ motion {args.motion}", flush=True)
    print(f"[[ {T} frames @ {fps:g} fps | physics {1/args.physics_dt:g} Hz | decimation {decim}", flush=True)
    print(f"[[ gravity ON, ground ON, contacts ON | 21 joints mapped by name", flush=True)

    def ref_target(i):
        jp = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        jp[0, idx] = torch.tensor(dof_ref[i], device=dev, dtype=torch.float32)
        return jp

    def ref_vel(i):
        jv = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        jv[0, idx] = torch.tensor(vel_ref[i], device=dev, dtype=torch.float32)
        return jv

    # ---- initialise ONCE from reference frame 0 --------------------------------
    q0 = ref_target(0)
    pose0 = torch.tensor(np.concatenate([root_ref[0], quat_ref[0]]),
                         device=dev, dtype=torch.float32).unsqueeze(0)
    robot.write_root_link_pose_to_sim(pose0)
    robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    robot.set_joint_position_target(q0); robot.write_data_to_sim()
    for _ in range(args.settle):
        sim.step(render=False); robot.update(args.physics_dt)
    print(f"[[ initialised at frame 0 (root z {float(root_ref[0][2]):.4f} m); "
          f"no teleporting from here on", flush=True)

    # ---- logs -------------------------------------------------------------------
    L = {k: [] for k in ("q_ref", "q_act", "q_err", "q_vel", "torque",
                         "root_pos", "root_pos_ref", "root_quat", "root_quat_ref",
                         "root_lin_vel", "root_ang_vel", "paw_z", "contacts")}

    for i in range(T):
        tgt = ref_target(i)
        nxt = ref_target(min(i + 1, T - 1))
        vt = ref_vel(i)
        vn = ref_vel(min(i + 1, T - 1))
        for k in range(decim):
            if args.hold:
                sub, subv = tgt, vt
            else:                       # first-order hold to the physics rate
                f = (k + 1) / decim
                sub = tgt * (1.0 - f) + nxt * f
                subv = vt * (1.0 - f) + vn * f
            robot.set_joint_position_target(sub)
            robot.set_joint_velocity_target(subv if args.vel_ff
                                            else torch.zeros_like(subv))
            robot.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(args.physics_dt)

        q_act = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        q_tgt = tgt[0].detach().cpu().numpy().copy()
        rp = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        rq = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        # geometric paw contact (no contact reporter on this USD)
        bp = robot.data.body_pos_w[0].detach().cpu().numpy()
        bq = robot.data.body_quat_w[0].detach().cpu().numpy()
        pz = cm.paw_heights(bp, bq, body_names)

        L["q_ref"].append(q_tgt); L["q_act"].append(q_act)
        L["q_err"].append(np.abs(q_act - q_tgt))
        L["q_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        try:
            L["torque"].append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        except Exception:
            L["torque"].append(np.zeros(len(isaac_names)))
        L["root_pos"].append(rp); L["root_pos_ref"].append(root_ref[i])
        L["root_quat"].append(rq); L["root_quat_ref"].append(quat_ref[i])
        L["root_lin_vel"].append(robot.data.root_lin_vel_w[0].detach().cpu().numpy().copy())
        L["root_ang_vel"].append(robot.data.root_ang_vel_w[0].detach().cpu().numpy().copy())
        pxy = np.array([(bp[knee_i[l]])[:2] for l in LEGS])
        L.setdefault("paw_xy", []).append(pxy)
        L["paw_z"].append(pz); L["contacts"].append(pz < CONTACT_H)

        if args.headless and (i % 60 == 0 or i == T - 1):
            print(f"[[ t={i:4d} qerr max {np.abs(q_act-q_tgt).max():.3f} rad | "
                  f"root z {rp[2]:+.3f} (ref {root_ref[i][2]:+.3f}) | "
                  f"contacts {int((pz < CONTACT_H).sum())}", flush=True)

    D = {k: np.asarray(v) for k, v in L.items()}

    # ---- analysis ---------------------------------------------------------------
    qerr = D["q_err"]
    rpe = np.linalg.norm(D["root_pos"] - D["root_pos_ref"], axis=1)
    rqe = quat_angle_deg(D["root_quat"], D["root_quat_ref"])
    up_z = np.array([quat_to_R(q)[2, 2] for q in D["root_quat"]])
    tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))

    div_j = np.where(qerr.max(1) > args.diverge_rad)[0]
    div_r = np.where(rpe > args.diverge_m)[0]
    first_j = int(div_j[0]) if len(div_j) else -1
    first_r = int(div_r[0]) if len(div_r) else -1

    vel_sat = np.abs(D["q_vel"]) >= (vel_lim[None, :] - 1e-3)
    tq_sat = np.abs(D["torque"]) >= (eff_lim[None, :] - 1e-4)

    fallen = (D["root_pos"][:, 2] < 0.5 * float(root_ref[0][2])) | (tilt > 70.0)
    first_fall = int(np.where(fallen)[0][0]) if fallen.any() else -1

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, os.path.splitext(os.path.basename(args.motion))[0] + "_stage4")
    np.savez(stem + ".npz", joint_names=np.array(isaac_names), fps=np.array(fps), **D)
    with open(stem + ".csv", "w") as f:
        f.write("frame,qerr_max,qerr_mean,root_pos_err_m,root_ori_err_deg,root_z,"
                "tilt_deg,n_contacts,max_jvel,max_torque\n")
        for i in range(T):
            f.write(f"{i},{qerr[i].max():.6f},{qerr[i].mean():.6f},{rpe[i]:.6f},{rqe[i]:.4f},"
                    f"{D['root_pos'][i,2]:.6f},{tilt[i]:.3f},{int(D['contacts'][i].sum())},"
                    f"{np.abs(D['q_vel'][i]).max():.3f},{np.abs(D['torque'][i]).max():.4f}\n")

    print(f"\n[[ ===== STAGE 4 BASELINE ({T} frames) =====", flush=True)
    print(f"[[ joint tracking error : mean {qerr.mean():.4f} rad  max {qerr.max():.4f} rad", flush=True)
    print(f"[[ root position error  : mean {rpe.mean()*1000:.1f} mm  max {rpe.max()*1000:.1f} mm", flush=True)
    print(f"[[ root orientation err : mean {rqe.mean():.2f} deg  max {rqe.max():.2f} deg", flush=True)
    print(f"[[ first joint divergence (> {args.diverge_rad} rad): "
          f"{'frame ' + str(first_j) if first_j >= 0 else 'none'}", flush=True)
    print(f"[[ first root divergence  (> {args.diverge_m*1000:.0f} mm): "
          f"{'frame ' + str(first_r) if first_r >= 0 else 'none'}", flush=True)
    print(f"[[ fallen/collapsed      : "
          f"{'frame ' + str(first_fall) if first_fall >= 0 else 'no'}"
          f" | root z {D['root_pos'][:,2].min():.3f}..{D['root_pos'][:,2].max():.3f} m"
          f" | max tilt {tilt.max():.1f} deg", flush=True)
    print(f"[[ contacts per frame    : mean {D['contacts'].sum(1).mean():.2f} of 4"
          f" | frames with none {int((D['contacts'].sum(1) == 0).sum())}", flush=True)
    if "paw_xy" in D:
        _sl = []
        for i in range(1, T):
            p_ = D["contacts"][i] & D["contacts"][i-1]
            if p_.any():
                _sl.append(np.linalg.norm(D["paw_xy"][i][p_] - D["paw_xy"][i-1][p_], axis=1).max())
        if _sl:
            print(f"[[ planted-paw SLIP     : mean {np.mean(_sl)*1000:.2f} mm/frame  "
                  f"max {np.max(_sl)*1000:.2f} mm  total {np.sum(_sl)*1000:.1f} mm", flush=True)
    print(f"[[ min paw z             : {D['paw_z'].min()*1000:+.1f} mm"
          f" (negative = through the floor)", flush=True)

    worst = np.argsort(qerr.mean(0))[::-1][:6]
    print(f"[[ worst-tracked joints (mean err):", flush=True)
    for j in worst:
        print(f"     {isaac_names[j]:18s} mean {qerr[:,j].mean():.4f} max {qerr[:,j].max():.4f} rad"
              f" | vel-sat {100*vel_sat[:,j].mean():5.1f}%  torque-sat {100*tq_sat[:,j].mean():5.1f}%"
              f" (eff lim {eff_lim[j]:.2f})", flush=True)
    sat_any = [isaac_names[j] for j in range(len(isaac_names)) if tq_sat[:, j].mean() > 0.05]
    print(f"[[ joints torque-saturated >5% of frames: {sat_any if sat_any else 'none'}", flush=True)
    print(f"[[ wrote {stem}.npz and {stem}.csv", flush=True)

    if not args.headless:
        import time
        for _ in range(max(0, args.loops - 1)):
            pass
        print("[[ done - close the window to quit", flush=True)
        while app.is_running():
            sim.step(render=True)
    os._exit(0)


main()
