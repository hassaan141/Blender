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
p.add_argument("--torque-ff", action="store_true",
               help="apply stage4_torque_residual from dynamic_retarget.py as "
                    "actuator feed-forward; configured effort limits still apply")
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
p.add_argument("--base-balance", type=float, default=0.0, metavar="GAIN",
               help="contact-safe roll/pitch feedback through the physical legs. "
                    "0 disables it; 1 applies the geometric correction needed to "
                    "keep the reference paw plane level. This is joint feedback, "
                    "not a force on the floating base.")
p.add_argument("--balance-cap-m", type=float, default=0.025,
               help="maximum per-leg vertical balance correction in metres")
p.add_argument("--balance-cap-rad", type=float, default=0.18,
               help="maximum balance correction per leg joint in radians")
p.add_argument("--balance-start", type=int, default=0,
               help="first reference frame using base-balance feedback")
p.add_argument("--balance-end", type=int, default=-1,
               help="last feedback frame; -1 means the end of the clip")
p.add_argument("--balance-windows", default=None,
               help="optional comma-separated feedback ranges, e.g. "
                    "'115-140,168-193'; overrides --balance-start/end")
p.add_argument("--whole-body", action="store_true",
               help="contact-aware whole-body root tracking through physical leg IK; "
                    "never applies forces or poses directly to the floating root")
p.add_argument("--contact-source", default=None,
               help="npz carrying the intended stance schedule (defaults to --motion)")
p.add_argument("--contact-field", default="source_contacts",
               help="intended stance field; falls back to 'contacts' when absent")
p.add_argument("--wb-pos-gain", type=float, default=0.20,
               help="fraction of root position error converted to stance-paw displacement")
p.add_argument("--wb-vel-horizon", type=float, default=0.08,
               help="seconds of root linear-velocity error added to the position correction")
p.add_argument("--wb-rot-gain", type=float, default=0.35,
               help="fraction of root rotation-vector error converted to stance-paw displacement")
p.add_argument("--wb-ang-horizon", type=float, default=0.06,
               help="seconds of root angular-velocity error added to the rotation correction")
p.add_argument("--wb-max-pos", type=float, default=0.045,
               help="per-axis stance-paw translation correction cap, metres")
p.add_argument("--wb-max-rot", type=float, default=0.30,
               help="per-axis stance-paw rotation correction cap, radians")
p.add_argument("--wb-max-dq", type=float, default=0.30,
               help="maximum residual applied to any leg joint, radians")
p.add_argument("--wb-pos-axes", default="1,1,1",
               help="comma-separated body-x,y,z tracking multipliers")
p.add_argument("--wb-rot-axes", default="1,1,1",
               help="comma-separated body-roll,pitch,yaw tracking multipliers")
p.add_argument("--wb-support", choices=("planned", "actual", "both"), default="both",
               help="which stance set receives whole-body corrections")
p.add_argument("--wb-windows", default=None,
               help="comma-separated whole-body feedback frame ranges; default is full clip")
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
from v4_kinematics import V4Kin, LEGS
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


def rotvec_from_R(R):
    """Stable rotation vector for the modest per-step attitude errors used here."""
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    a = float(np.arccos(c))
    v = 0.5 * np.array([R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]])
    s = float(np.sin(a))
    return v if a < 1e-7 or abs(s) < 1e-7 else v * (a / s)


def nlerp_quat(q0, q1, a):
    """Shortest-path normalized interpolation; sufficient at 120 Hz."""
    q1 = q1 if np.dot(q0, q1) >= 0.0 else -q1
    q = (1.0 - a) * q0 + a * q1
    return q / (np.linalg.norm(q) + 1e-12)


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
    root_lv_ref = (m["body_linear_velocities"][:, 0].astype(np.float64)
                   if "body_linear_velocities" in m.files else
                   np.gradient(root_ref, control_dt, axis=0))
    root_av_ref = (m["body_angular_velocities"][:, 0].astype(np.float64)
                   if "body_angular_velocities" in m.files else np.zeros((T, 3)))
    torque_ff_npz = np.zeros_like(dof_ref)
    if args.torque_ff:
        if "stage4_torque_residual" not in m.files:
            raise ValueError("--torque-ff requires stage4_torque_residual in the motion")
        raw = np.asarray(m["stage4_torque_residual"], float)
        if raw.shape != (T, 4, 3):
            raise ValueError(f"stage4_torque_residual shape {raw.shape}, expected {(T, 4, 3)}")
        for lk, leg in enumerate(LEGS):
            for jj, suffix in enumerate(("SY_J", "SP_J", "knee")):
                torque_ff_npz[:, npz_names.index(f"{leg}_{suffix}")] = raw[:, lk, jj]

    # Linearized physical-leg IK used by the optional landing stabilizer.  A
    # floating base cannot be orientation-controlled directly; instead, a lifted
    # side's paws are extended through SP/knee joints.  Precompute each frame's
    # local paw position and least-norm dq/dz so runtime feedback stays cheap.
    balance_paw = balance_dq_dz = q_lo = q_hi = None
    if args.base_balance:
        urdf = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "URDF",
            "bingo_urdf v4_w_ear_joints", "urdf",
            "bingo_urdf_w_ear_joints_physics.urdf"))
        kin_balance = V4Kin(urdf)
        leg_npz = [[npz_names.index(f"{leg}_SY_J"),
                    npz_names.index(f"{leg}_SP_J"),
                    npz_names.index(f"{leg}_knee")] for leg in LEGS]
        balance_paw = np.zeros((T, 4, 3))
        balance_dq_dz = np.zeros((T, 4, 3))
        eps = 1e-4
        for i in range(T):
            for k, leg in enumerate(LEGS):
                qleg = dof_ref[i, leg_npz[k]].copy()
                ankle = kin_balance.leg_points(leg, qleg)[2]
                balance_paw[i, k] = ankle
                jz = np.zeros(3)
                # SY is intentionally excluded: it is already the most limited
                # axis and changing it would turn landing balance into steering.
                for j in (1, 2):
                    qq = qleg.copy(); qq[j] += eps
                    jz[j] = (kin_balance.leg_points(leg, qq)[2][2] - ankle[2]) / eps
                balance_dq_dz[i, k] = jz / (np.dot(jz, jz) + 1e-4)
        q_lo = np.array([kin_balance.j[n]["lo"] for n in npz_names])
        q_hi = np.array([kin_balance.j[n]["hi"] for n in npz_names])
        if args.balance_windows:
            balance_windows = [tuple(int(x) for x in part.split("-", 1))
                               for part in args.balance_windows.split(",")]
        else:
            balance_windows = [(args.balance_start,
                                T - 1 if args.balance_end < 0 else args.balance_end)]

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

    # Full-pose feedback is converted into support-point motion through the exact
    # v4 leg FK.  This is the model-based inner controller in the TMR loop: the
    # simulator remains the full dynamics model and the floating root is untouched.
    wb_kin = wb_leg_npz = wb_lo = wb_hi = None
    # The intended contact schedule is part of the Stage-4 objective even when
    # online whole-body feedback is disabled.  Prefer the source-authored schedule;
    # the derived `contacts` field in older files can be stale.
    cp = args.contact_source or args.motion
    cr = np.load(cp, allow_pickle=True)
    cf = args.contact_field if args.contact_field in cr.files else "contacts"
    wb_contacts = (np.asarray(cr[cf], bool) if cf in cr.files and len(cr[cf]) == T
                   else np.zeros((T, 4), bool))
    if args.whole_body:
        urdf = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "URDF",
            "bingo_urdf v4_w_ear_joints", "urdf",
            "bingo_urdf_w_ear_joints_physics.urdf"))
        wb_kin = V4Kin(urdf)
        wb_leg_npz = [[npz_names.index(f"{leg}_SY_J"),
                       npz_names.index(f"{leg}_SP_J"),
                       npz_names.index(f"{leg}_knee")] for leg in LEGS]
        wb_lo = np.array([wb_kin.j[n]["lo"] for n in npz_names])
        wb_hi = np.array([wb_kin.j[n]["hi"] for n in npz_names])
        if cf not in cr.files or len(cr[cf]) != T:
            raise ValueError(f"whole-body contact schedule must have {T} frames: {cp}")
        wb_contacts = np.asarray(cr[cf], bool)
        wb_pos_axes = np.asarray([float(x) for x in args.wb_pos_axes.split(",")])
        wb_rot_axes = np.asarray([float(x) for x in args.wb_rot_axes.split(",")])
        if wb_pos_axes.shape != (3,) or wb_rot_axes.shape != (3,):
            raise ValueError("--wb-pos-axes and --wb-rot-axes require three values")
        wb_windows = ([tuple(int(x) for x in p.split("-", 1))
                       for p in args.wb_windows.split(",")]
                      if args.wb_windows else [(0, T - 1)])

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
    if args.base_balance:
        print(f"[[ physical base-balance feedback gain {args.base_balance:g} | "
              f"caps {args.balance_cap_m*1000:g} mm / {args.balance_cap_rad:g} rad | "
              f"windows {balance_windows}", flush=True)
    if args.whole_body:
        print(f"[[ whole-body contact IK | pos {args.wb_pos_gain:g} + "
              f"{args.wb_vel_horizon:g}s*vel | rot {args.wb_rot_gain:g} + "
              f"{args.wb_ang_horizon:g}s*ang | caps {args.wb_max_pos:g}m/"
              f"{args.wb_max_rot:g}rad/{args.wb_max_dq:g}rad | "
              f"planned contacts {int(wb_contacts.sum())}", flush=True)
    if args.torque_ff:
        print(f"[[ contact-wrench torque feed-forward ON | max requested "
              f"{np.abs(torque_ff_npz).max():.3f} Nm", flush=True)

    def ref_target(i):
        jp = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        jp[0, idx] = torch.tensor(dof_ref[i], device=dev, dtype=torch.float32)
        return jp

    def ref_vel(i):
        jv = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        jv[0, idx] = torch.tensor(vel_ref[i], device=dev, dtype=torch.float32)
        return jv

    def ref_effort(i):
        je = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        je[0, idx] = torch.tensor(torque_ff_npz[i], device=dev, dtype=torch.float32)
        return je

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
    L = {k: [] for k in ("q_ref", "q_cmd", "q_act", "q_err", "q_vel", "torque",
                         "torque_ff",
                         "root_pos", "root_pos_ref", "root_quat", "root_quat_ref",
                         "root_lin_vel", "root_ang_vel", "paw_z", "contacts",
                         "planned_contacts", "wb_corr")}

    for i in range(T):
        tgt = ref_target(i)
        nxt = ref_target(min(i + 1, T - 1))
        vt = ref_vel(i)
        vn = ref_vel(min(i + 1, T - 1))
        et = ref_effort(i)
        en = ref_effort(min(i + 1, T - 1))
        for k in range(decim):
            if args.hold:
                sub, subv = tgt, vt
                sube = et
            else:                       # first-order hold to the physics rate
                f = (k + 1) / decim
                sub = tgt * (1.0 - f) + nxt * f
                subv = vt * (1.0 - f) + vn * f
                sube = et * (1.0 - f) + en * f
            wb_corr = np.zeros(len(npz_names))
            wb_active = args.whole_body and any(a <= i <= b for a, b in wb_windows)
            if wb_active:
                j1 = min(i + 1, T - 1)
                rp_act = robot.data.root_pos_w[0].detach().cpu().numpy()
                rq_act = robot.data.root_quat_w[0].detach().cpu().numpy()
                rv_act = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
                rw_act = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
                rp_des = (1.0 - f) * root_ref[i] + f * root_ref[j1]
                rq_des = nlerp_quat(quat_ref[i], quat_ref[j1], f)
                rv_des = (1.0 - f) * root_lv_ref[i] + f * root_lv_ref[j1]
                rw_des = (1.0 - f) * root_av_ref[i] + f * root_av_ref[j1]
                Ra, Rd = quat_to_R(rq_act), quat_to_R(rq_des)
                # Desired body displacement/rotation expressed in the current body.
                dp_body = (args.wb_pos_gain * (Ra.T @ (rp_des - rp_act))
                           + args.wb_vel_horizon * (Ra.T @ (rv_des - rv_act)))
                dr_body = (args.wb_rot_gain * rotvec_from_R(Ra.T @ Rd)
                           + args.wb_ang_horizon * (Ra.T @ (rw_des - rw_act)))
                dp_body = wb_pos_axes * np.clip(
                    dp_body, -args.wb_max_pos, args.wb_max_pos)
                dr_body = wb_rot_axes * np.clip(
                    dr_body, -args.wb_max_rot, args.wb_max_rot)
                base_npz = sub[0, idx].detach().cpu().numpy().copy()
                bp_now = robot.data.body_pos_w[0].detach().cpu().numpy()
                bq_now = robot.data.body_quat_w[0].detach().cpu().numpy()
                actual_support = cm.paw_heights(bp_now, bq_now, body_names) < CONTACT_H
                # Planned stance preserves authored timing; actual stance avoids
                # applying a ground-reaction correction to a foot still in swing.
                planned = ({"planned": wb_contacts[i], "actual": actual_support,
                            "both": wb_contacts[i] | actual_support}[args.wb_support])
                eps = 1e-4
                for lk, leg in enumerate(LEGS):
                    if not planned[lk]:
                        continue
                    ids_npz = wb_leg_npz[lk]
                    qleg = base_npz[ids_npz].copy()
                    hull = cm.hull[f"{leg}_knee"]
                    paw = wb_kin.leg_points(
                        leg, qleg, support_hull=hull, world_R=Ra,
                        support_softness=0.001)[3]
                    J = np.zeros((3, 3))
                    for jj in range(3):
                        qq = qleg.copy(); qq[jj] += eps
                        pp = wb_kin.leg_points(
                            leg, qq, support_hull=hull, world_R=Ra,
                            support_softness=0.001)[3]
                        J[:, jj] = (pp - paw) / eps
                    # Fixed stance paw: desired body translation/rotation is made
                    # by moving the paw oppositely in body coordinates.
                    paw_delta = -dp_body - np.cross(dr_body, paw)
                    dq = J.T @ np.linalg.solve(J @ J.T + 2e-4 * np.eye(3), paw_delta)
                    dq = np.clip(dq, -args.wb_max_dq, args.wb_max_dq)
                    wb_corr[ids_npz] = dq
                    base_npz[ids_npz] = qleg + dq
                base_npz = np.clip(base_npz, wb_lo, wb_hi)
                sub[0, idx] = torch.tensor(base_npz, device=dev, dtype=torch.float32)
            balance_active = (args.base_balance and any(
                a <= i <= b for a, b in balance_windows))
            if balance_active:
                rq_act = robot.data.root_quat_w[0].detach().cpu().numpy()
                Rerr = quat_to_R(quat_ref[i]).T @ quat_to_R(rq_act)
                theta = 0.5 * np.array([
                    Rerr[2, 1] - Rerr[1, 2],
                    Rerr[0, 2] - Rerr[2, 0],
                    Rerr[1, 0] - Rerr[0, 1],
                ])
                corr = np.zeros(len(npz_names))
                for lk, ids_npz in enumerate(leg_npz):
                    x, y = balance_paw[i, lk, :2]
                    dz = args.base_balance * (-theta[0] * y + theta[1] * x)
                    dz = float(np.clip(dz, -args.balance_cap_m, args.balance_cap_m))
                    dq = np.clip(balance_dq_dz[i, lk] * dz,
                                 -args.balance_cap_rad, args.balance_cap_rad)
                    corr[ids_npz] = dq
                base_npz = sub[0, idx].detach().cpu().numpy() + corr
                base_npz = np.clip(base_npz, q_lo, q_hi)
                sub[0, idx] = torch.tensor(base_npz, device=dev, dtype=torch.float32)
            robot.set_joint_position_target(sub)
            robot.set_joint_velocity_target(subv if args.vel_ff
                                            else torch.zeros_like(subv))
            robot.set_joint_effort_target(sube if args.torque_ff
                                          else torch.zeros_like(sube))
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

        q_cmd = sub[0].detach().cpu().numpy().copy()
        L["q_ref"].append(q_tgt); L["q_cmd"].append(q_cmd); L["q_act"].append(q_act)
        L["q_err"].append(np.abs(q_act - q_tgt))
        L["torque_ff"].append(sube[0].detach().cpu().numpy().copy())
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
        L["planned_contacts"].append(wb_contacts[i])
        L["wb_corr"].append(wb_corr)

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

    # "Fallen" has to be measured against the REFERENCE, not against absolutes.
    # Laidback lies down to 54 mm and Eccentric is an authored SIT held at 47-78 deg
    # of tilt: a fixed 70 deg / half-of-frame-0-height rule scores both as falls on
    # frames where the robot is doing exactly what was authored. Compare instead the
    # angle between the actual and the reference up-axis, and the height against the
    # reference height at THAT frame.
    up_ref = np.array([quat_to_R(q)[:, 2] for q in D["root_quat_ref"]])
    up_act = np.array([quat_to_R(q)[:, 2] for q in D["root_quat"]])
    tilt_vs_ref = np.degrees(np.arccos(np.clip((up_ref * up_act).sum(1), -1, 1)))
    ref_z = np.maximum(D["root_pos_ref"][:, 2], 1e-3)
    fallen = (D["root_pos"][:, 2] < 0.5 * ref_z) | (tilt_vs_ref > 45.0)
    first_fall = int(np.where(fallen)[0][0]) if fallen.any() else -1

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, os.path.splitext(os.path.basename(args.motion))[0] + "_stage4")
    np.savez(stem + ".npz", joint_names=np.array(isaac_names), fps=np.array(fps),
             effort_limits=eff_lim, velocity_limits=vel_lim, **D)
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
          f" | max tilt {tilt.max():.1f} deg (ref {np.degrees(np.arccos(np.clip(up_ref[:,2],-1,1))).max():.1f})"
          f" | max tilt vs REF {tilt_vs_ref.max():.1f} deg", flush=True)
    print(f"[[ contacts per frame    : mean {D['contacts'].sum(1).mean():.2f} of 4"
          f" | frames with none {int((D['contacts'].sum(1) == 0).sum())}", flush=True)
    if wb_contacts.shape == D["contacts"].shape and wb_contacts.any():
        ref_ct = wb_contacts
        agree = 100.0 * float((ref_ct == D["contacts"]).mean())
        print(f"[[ contact agreement     : {agree:.1f}% vs the reference schedule "
              f"(reference mean {ref_ct.sum(1).mean():.2f} of 4)", flush=True)
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
