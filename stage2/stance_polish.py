"""Kill the residual skate of a planted paw, by penalising the MATERIAL velocity
of the contacting point rather than the position of "the lowest vertex".

Why a separate pass. solve_spatial_retarget locks each stance to a world anchor and
scores itself on a softmin of the contact patch, which reads 0.16-0.20 mm per
stance-frame. Measured properly (stage2/slip_audit.py: the world velocity of the
hull vertex that is actually touching) the same clips still skate 0.9-2.2 mm/frame,
against Ashley's 0.03-0.11. That residue is what pushes the physics robot around:
Laidback stands still for 280 frames and accumulates 18 deg of unauthored yaw
before its turn, and lands the turn's swing feet in the wrong place.

The distinction matters because a convex paw is allowed to ROLL: the contact point
travels along the floor while the material point in contact is instantaneously at
rest, and that costs no friction. A position-based penalty forbids legal rolling
and permits illegal skating; a material-velocity penalty does the opposite.

Per frame, for each planted leg, with the neighbour frames held fixed:

    minimise  || X_i(c) - X_{i-1}(c) ||_xy  +  || X_{i+1}(c) - X_i(c) ||_xy
              + floor terms + a regulariser toward the incoming pose

where c is the hull vertex touching at frame i, evaluated in BOTH frames' poses -
i.e. a finite difference of one material point, not of the argmin. Swept forward,
backward, forward (Gauss-Seidel), so the correction propagates along the stance.

Root pose, expressive joints and the contact schedule are untouched.

  python3 stage2/stance_polish.py --motion stage2/out/laidback_grounded.npz \
      --out stage2/out/laidback_polished.npz
"""
import argparse, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "stage4"))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--w-slip", type=float, default=300.0)
    ap.add_argument("--w-floor", type=float, default=400.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--w-smooth", type=float, default=0.0,
                    help="penalise q_i deviating from the average of its neighbours. "
                         "The slip objective on its own buys a lower number with "
                         "high-frequency joint targets the physics tracker cannot "
                         "follow - measured on DeadPan, reference slip 1.19 -> 0.92 "
                         "mm/frame while the fall moved 227 -> 115.")
    ap.add_argument("--contact-tol", type=float, default=0.005)
    ap.add_argument("--start", type=int, default=None,
                    help="restrict the pass to a frame window. The pass helps where "
                         "stance runs are long and never break, and hurts where the "
                         "clip steps, so it is sometimes right for part of a clip.")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--root", type=float, default=0.03,
                    help="metres. Also solve a bounded horizontal BODY offset that "
                         "removes the skate the legs cannot. With the root pose held "
                         "fixed a 3-DOF leg simply cannot keep a foot still while the "
                         "body moves where Ashley put it, so part of the residue is not "
                         "the legs' to fix. 0 disables.")
    ap.add_argument("--w-root-prior", type=float, default=6.0)
    ap.add_argument("--w-root-acc", type=float, default=25.0)
    a = ap.parse_args()

    kin = V4Kin(URDF); cm = ContactModel()
    m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
    q = m["dof_positions"].astype(float).copy()
    q0_in = q.copy()
    names = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q)
    ji = {f"{l}_{s}": names.index(f"{l}_{s}") for l in LEGS for s in ("SY_J", "SP_J", "knee")}
    lim = {l: kin.leg_limits(l) for l in LEGS}
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}
    R0 = [quat_to_mat(rq[i]) for i in range(T)]

    if "stage4_planted" in m.files:
        planted = np.asarray(m["stage4_planted"], bool)
    elif "source_contacts" in m.files:
        planted = np.asarray(m["source_contacts"], bool)
    else:
        raise SystemExit("motion carries no stance schedule")

    def chain(leg, qleg, i):
        R, p = R0[i], rp[i].copy()
        for nm, qi in zip(kin.leg_chain(leg), qleg):
            J = kin.j[nm]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
        return R, p

    def vert_world(leg, qleg, i, c):
        R, p = chain(leg, qleg, i)
        return R @ c + p

    def low(leg, qleg, i):
        R, p = chain(leg, qleg, i)
        w = hull[leg] @ R.T + p
        j = int(np.argmin(w[:, 2]))
        return hull[leg][j], float(w[j, 2])

    def getq(i, k):
        l = LEGS[k]
        return np.array([q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]], q[i, ji[f"{l}_knee"]]])

    root_off = np.zeros((T, 3))

    def solve_root():
        """Least-squares horizontal body offset that cancels the remaining skate.
        The material-point difference is affine in the offset:
            (X_i - X_{i-1}) + (o_i - o_{i-1})
        so this is a linear problem, tridiagonal after the smoothness terms."""
        rows = []; rhs = []
        for i in range(max(1, f0), f1):
            for k, l in enumerate(LEGS):
                if not (planted[i, k] and planted[i - 1, k]):
                    continue
                c, _ = low(l, getq(i, k), i)
                dxy = (vert_world(l, getq(i, k), i, c)
                       - vert_world(l, getq(i - 1, k), i - 1, c))[:2]
                r = np.zeros(T); r[i] = 1.0; r[i - 1] = -1.0
                rows.append(r); rhs.append(-dxy)
        for i in range(T):
            r = np.zeros(T); r[i] = a.w_root_prior / a.w_slip * 10.0
            rows.append(r); rhs.append(np.zeros(2))
        for i in range(1, T - 1):
            r = np.zeros(T); r[i - 1] = 1.0; r[i] = -2.0; r[i + 1] = 1.0
            rows.append(r * a.w_root_acc / a.w_slip * 10.0); rhs.append(np.zeros(2))
        A = np.vstack(rows); b = np.vstack(rhs)
        o = np.linalg.lstsq(A, b, rcond=None)[0]
        n = np.linalg.norm(o, axis=1, keepdims=True)
        return o * np.minimum(1.0, a.root / np.maximum(n, 1e-12))

    f0 = 0 if a.start is None else max(0, a.start)
    f1 = T if a.end is None else min(T, a.end + 1)
    order = list(range(f0, f1))
    for sw in range(a.sweeps):
        seq = order if sw % 2 == 0 else order[::-1]
        for i in seq:
            for k, l in enumerate(LEGS):
                if not planted[i, k]:
                    continue
                qc = getq(i, k)
                c, _ = low(l, qc, i)                     # the touching material point
                nb = []
                if i > 0 and planted[i-1, k]:
                    nb.append(vert_world(l, getq(i-1, k), i-1, c))
                if i < T-1 and planted[i+1, k]:
                    nb.append(vert_world(l, getq(i+1, k), i+1, c))
                if not nb:
                    continue
                lo, hi = lim[l][:, 0], lim[l][:, 1]
                qc = np.clip(qc, lo + 1e-9, hi - 1e-9)

                qmid = None
                if a.w_smooth > 0 and 0 < i < T - 1:
                    qmid = 0.5 * (getq(i - 1, k) + getq(i + 1, k))

                def res(x, l=l, i=i, c=c, nb=nb, qc=qc, qmid=qmid):
                    w = vert_world(l, x, i, c)
                    r = []
                    for wn in nb:
                        r.extend(a.w_slip * (w[:2] - wn[:2]))
                    _, z = low(l, x, i)
                    r.append(a.w_floor * (z - 0.0))
                    r.extend(a.w_reg * (x - qc))
                    if qmid is not None:
                        r.extend(a.w_smooth * (x - qmid))
                    return np.array(r)

                rr = least_squares(res, qc, bounds=(lo, hi), xtol=1e-11, ftol=1e-11,
                                   max_nfev=80)
                q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]], q[i, ji[f"{l}_knee"]] = rr.x

        if a.root > 0 and sw < a.sweeps - 1:
            o = solve_root()
            rp[:, :2] += o
            root_off[:, :2] += o
            print(f"[[   sweep {sw}: body offset mean "
                  f"{np.linalg.norm(root_off[:, :2], axis=1).mean()*1000:.1f} mm "
                  f"max {np.linalg.norm(root_off[:, :2], axis=1).max()*1000:.1f} mm")

    dq = np.abs(q[:, :12] - q0_in[:, :12])
    print(f"[[ stance polish: {a.sweeps} sweeps over {int(planted.sum())} planted "
          f"foot-frames | leg joint change mean {np.degrees(dq.mean()):.2f} deg "
          f"max {np.degrees(dq.max()):.2f} deg")
    if a.root > 0:
        d["root_pos"] = rp.astype(np.float32)
        if "body_positions" in d:
            bp = np.asarray(d["body_positions"], float).copy(); bp[:, 0] = rp
            d["body_positions"] = bp.astype(np.float32)
        d["stance_polish_offset"] = root_off.astype(np.float32)
    d["dof_positions"] = q.astype(np.float32)
    d["dof_velocities"] = np.gradient(q, 1.0 / float(m["fps"]), axis=0).astype(np.float32)
    np.savez(a.out, **d)
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
