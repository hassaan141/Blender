"""Build a dynamics-aware Stage-4 joint reference from a kinematic motion.

The floating base is never prescribed.  Instead, the required centroidal wrench
of the authored motion is distributed over the intended stance paws, subject to
unilateral contact and the real ground friction coefficient.  The resulting
stance-leg torques are converted to bounded IdealPD position residuals using the
actual v4 gains and effort limits.  Isaac remains the authority: this file only
produces joint targets which must subsequently pass track_v4_physics.py.

This is the inner, model-based control part of TMR.  Temporal deformation stays a
separate outer operation (retime_segment.py), so timing and controls can be
iterated without changing the trusted Stage-3 source.
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, quat_to_mat
from contact_model import ContactModel
from wrench_refine import centroidal, evaluate

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints",
                    "urdf", "bingo_urdf_w_ear_joints_physics.urdf")
G = 9.81

# Exact values from bingo_v4.py, not tuning parameters invented for this pass.
LEG_KP = np.array([40.0, 120.0, 120.0])
LEG_EFFORT = np.array([3.0, 3.0, 3.0])


def cross_matrix(p):
    return np.array([[0.0, -p[2], p[1]],
                     [p[2], 0.0, -p[0]],
                     [-p[1], p[0], 0.0]])


def allocate_wrench(points, active, F, M, mu, previous):
    """Friction-constrained force allocation with soft wrench tracking.

    A source pose can demand an impossible wrench.  In that case SLSQP returns
    the closest legal wrench instead of hiding the defect in an unbounded joint
    command.  Moment rows are expressed in metre-equivalent units so force and
    moment residuals have comparable conditioning on Bingo's scale.
    """
    ids = np.where(active)[0]
    out = np.zeros((4, 3))
    if not len(ids):
        return out, np.r_[F, M]
    A = np.zeros((6, 3 * len(ids)))
    for k, leg_i in enumerate(ids):
        A[:3, 3*k:3*k+3] = np.eye(3)
        A[3:, 3*k:3*k+3] = cross_matrix(points[leg_i])
    # Scale moments by a characteristic body length (about the support radius).
    radius = max(0.12, float(np.max(np.linalg.norm(points[ids, :2], axis=1))))
    W = np.diag([1.0, 1.0, 1.0, 1.0/radius, 1.0/radius, 1.0/radius])
    b = np.r_[F, M]
    x0 = previous[ids].reshape(-1).copy()
    if not np.isfinite(x0).all() or x0[:, None].size == 0:
        x0 = np.zeros(3 * len(ids))
    # A physically neutral start is preferable after a contact transition.
    if np.linalg.norm(x0) < 1e-9:
        x0[2::3] = max(0.0, F[2]) / len(ids)

    def objective(x):
        r = W @ (A @ x - b)
        return float(r @ r + 2e-4 * np.sum((x - x0) ** 2))

    cons = []
    for k in range(len(ids)):
        j = 3 * k
        cons.extend([
            {"type": "ineq", "fun": lambda x, j=j: x[j+2]},
            {"type": "ineq", "fun": lambda x, j=j: mu*x[j+2] - x[j]},
            {"type": "ineq", "fun": lambda x, j=j: mu*x[j+2] + x[j]},
            {"type": "ineq", "fun": lambda x, j=j: mu*x[j+2] - x[j+1]},
            {"type": "ineq", "fun": lambda x, j=j: mu*x[j+2] + x[j+1]},
        ])
    sol = minimize(objective, x0, method="SLSQP", constraints=cons,
                   options={"maxiter": 120, "ftol": 1e-10, "disp": False})
    x = sol.x if sol.success and np.isfinite(sol.x).all() else x0
    out[ids] = x.reshape(-1, 3)
    return out, A @ x - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="temporally adjusted kinematic reference")
    ap.add_argument("--contacts", required=True, help="npz with intended stance schedule")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact-field", default="source_contacts")
    ap.add_argument("--mu", type=float, default=0.5,
                    help="must match Stage-4 ground material")
    ap.add_argument("--force-scale", type=float, default=1.0,
                    help="global scale for dynamic (above static support) force residual")
    ap.add_argument("--horizontal-scale", type=float, default=1.0)
    ap.add_argument("--vertical-scale", type=float, default=1.0)
    ap.add_argument("--roll-pitch-scale", type=float, default=1.0)
    ap.add_argument("--yaw-scale", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=1.5,
                    help="Gaussian smoothing of torque residual in animation frames")
    ap.add_argument("--windows", default=None,
                    help="optional comma-separated control windows, e.g. '90-145,158-180'; "
                         "outside them the trusted kinematic target is bit-for-bit unchanged")
    ap.add_argument("--window-ramp", type=int, default=5,
                    help="smooth activation/deactivation ramp in animation frames")
    ap.add_argument("--max-effort-fraction", type=float, default=0.95,
                    help="keep feed-forward residual below the configured 3 Nm limit")
    ap.add_argument("--torque-only", action="store_true",
                    help="preserve q bit-for-bit and export the optimized contact torque "
                         "for track_v4_physics.py --torque-ff")
    a = ap.parse_args()

    m = np.load(a.motion, allow_pickle=True)
    c = np.load(a.contacts, allow_pickle=True)
    q_source = np.asarray(m["dof_positions"])
    q = q_source.astype(float)
    names = [str(x) for x in m["dof_names"]]
    rp = np.asarray(m["root_pos"] if "root_pos" in m.files else
                    m["body_positions"][:, 0], float)
    rq = np.asarray(m["root_quat"] if "root_quat" in m.files else
                    m["body_rotations"][:, 0], float)
    field = a.contact_field if a.contact_field in c.files else "contacts"
    contacts = np.asarray(c[field], bool)
    if len(contacts) != len(q):
        raise SystemExit(f"contact schedule {len(contacts)} != motion {len(q)} frames")

    dyn = centroidal(a.motion, 0.006)
    hdot = np.gradient(dyn["h"], dyn["dt"], axis=0)
    ev = evaluate(dyn["C"], hdot, dyn["M"], dyn["sup"], dyn["dt"], a.mu)
    required_F = ev["F"].copy()
    required_M = ev["Mo"].copy()
    required_F[:, :2] *= a.horizontal_scale
    required_F[:, 2] = dyn["M"] * G + a.vertical_scale * (
        required_F[:, 2] - dyn["M"] * G)
    required_M[:, :2] *= a.roll_pitch_scale
    required_M[:, 2] *= a.yaw_scale

    kin = V4Kin(URDF)
    cm = ContactModel()
    leg_ids = [[names.index(f"{leg}_SY_J"), names.index(f"{leg}_SP_J"),
                names.index(f"{leg}_knee")] for leg in LEGS]
    T = len(q)
    points = np.zeros((T, 4, 3))
    jac = np.zeros((T, 4, 3, 3))
    eps = 1e-5
    for i in range(T):
        R = quat_to_mat(rq[i])
        for lk, leg in enumerate(LEGS):
            ql = q[i, leg_ids[lk]]
            hull = cm.hull[f"{leg}_knee"]
            p0 = kin.leg_points(leg, ql, support_hull=hull,
                                world_R=R, support_softness=0.001)[3]
            points[i, lk] = rp[i] + R @ p0
            for j in range(3):
                qq = ql.copy(); qq[j] += eps
                p1 = kin.leg_points(leg, qq, support_hull=hull,
                                    world_R=R, support_softness=0.001)[3]
                jac[i, lk, :, j] = R @ ((p1 - p0) / eps)

    forces = np.zeros((T, 4, 3)); residual = np.zeros((T, 6))
    prev = np.zeros((4, 3))
    for i in range(T):
        forces[i], residual[i] = allocate_wrench(
            points[i], contacts[i], required_F[i], required_M[i], a.mu, prev)
        prev = forces[i]

    # The open-loop reference already creates static ground support through normal
    # tracking error. Add only the dynamic part of the allocated force; otherwise
    # gravity is counted twice and the body is unnecessarily lifted.
    static = np.zeros_like(forces)
    for i in range(T):
        ids = np.where(contacts[i])[0]
        if len(ids):
            static[i, ids, 2] = dyn["M"] * G / len(ids)
    dynamic_force = forces - static
    dynamic_force[:, :, :2] *= a.force_scale
    dynamic_force[:, :, 2] *= a.force_scale

    tau = np.zeros((T, 4, 3))
    for i in range(T):
        for lk in range(4):
            if contacts[i, lk]:
                # Equilibrium: tau_motor + J^T f_contact = 0.
                tau[i, lk] = -jac[i, lk].T @ dynamic_force[i, lk]
    if a.smooth > 0:
        tau = gaussian_filter1d(tau, a.smooth, axis=0, mode="nearest")
    control_weight = np.ones(T)
    if a.windows:
        control_weight[:] = 0.0
        ramp = max(1, a.window_ramp)
        for part in a.windows.split(","):
            s, e = (int(x) for x in part.split("-", 1))
            s, e = max(0, s), min(T - 1, e)
            if s > e:
                continue
            control_weight[s:e+1] = 1.0
            nr = min(ramp, e - s + 1)
            u = np.linspace(0.0, 1.0, nr)
            ss = u*u*(3.0-2.0*u)
            control_weight[s:s+nr] = np.minimum(control_weight[s:s+nr], ss)
            control_weight[e-nr+1:e+1] = np.minimum(
                control_weight[e-nr+1:e+1], ss[::-1])
        tau *= control_weight[:, None, None]
    cap = a.max_effort_fraction * LEG_EFFORT
    tau = np.clip(tau, -cap, cap)

    dq = np.zeros_like(q)
    for lk, ids in enumerate(leg_ids):
        dq[:, ids] = tau[:, lk] / LEG_KP
    q_new = q + dq
    lo = np.array([kin.j[n]["lo"] for n in names])
    hi = np.array([kin.j[n]["hi"] for n in names])
    q_new = np.clip(q_new, lo, hi)
    # Cheeky sits on a narrow hybrid-contact basin: a one-ULP rewrite of an
    # otherwise untouched target has measurably selected a different collision
    # branch in GPU PhysX. Preserve inactive frames bit-for-bit.
    inactive = control_weight == 0.0
    q_new[inactive] = q_source[inactive]
    if a.torque_only:
        q_new = q_source.copy()
    dq = q_new - q

    out = {k: m[k] for k in m.files}
    out["dof_positions"] = q_new.astype(np.float32)
    out["dof_velocities"] = np.gradient(q_new, dyn["dt"], axis=0).astype(np.float32)
    out["source_contacts"] = contacts
    out["stage4_contact_forces"] = forces.astype(np.float32)
    out["stage4_dynamic_forces"] = dynamic_force.astype(np.float32)
    out["stage4_torque_residual"] = tau.astype(np.float32)
    out["stage4_joint_residual"] = dq.astype(np.float32)
    out["stage4_wrench_residual"] = residual.astype(np.float32)
    out["stage4_force_scales"] = np.array(
        [a.force_scale, a.horizontal_scale, a.vertical_scale,
         a.roll_pitch_scale, a.yaw_scale])
    out["stage4_friction"] = np.array(a.mu)
    out["stage4_control_weight"] = control_weight.astype(np.float32)
    np.savez(a.out, **out)

    lim = np.abs(tau) / LEG_EFFORT
    print(f"[[ dynamics-aware reference: {T} frames, mass {dyn['M']:.4f} kg, mu {a.mu:g}")
    print(f"[[ intended contacts: {int(contacts.sum())} paw-frames; flight frames "
          f"{int((contacts.sum(1)==0).sum())}")
    print(f"[[ legal wrench residual: force mean/max "
          f"{np.linalg.norm(residual[:,:3],axis=1).mean():.3f}/"
          f"{np.linalg.norm(residual[:,:3],axis=1).max():.3f} N | moment mean/max "
          f"{np.linalg.norm(residual[:,3:],axis=1).mean():.4f}/"
          f"{np.linalg.norm(residual[:,3:],axis=1).max():.4f} Nm")
    print(f"[[ added leg target residual mean/max "
          f"{np.degrees(np.abs(dq[:,:12])).mean():.3f}/"
          f"{np.degrees(np.abs(dq[:,:12])).max():.3f} deg")
    print(f"[[ feed-forward effort fraction mean/max {lim.mean():.3f}/{lim.max():.3f}; "
          f"at cap {100*np.mean(lim >= a.max_effort_fraction-1e-6):.1f}%")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
