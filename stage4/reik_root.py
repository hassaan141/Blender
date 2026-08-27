"""Consume a per-frame body offset by re-solving the legs, so the paws do not move.

stage4/wrench_refine.py decides WHERE the body has to be for the motion to be
contact-wrench feasible, but writes only the offset: translating the root alone
would drag every paw along with it and change nothing physically. This pass moves
the root and then re-solves the 12 leg joints so that

    - every paw that was ON the floor keeps its exact world contact point (hard),
    - every other paw keeps its world ankle position (soft, so swing shape is kept),
    - no hull vertex ends up below the floor,
    - joints stay inside the v4 limits and close to the incoming solution.

The head, tail, ears and the body ORIENTATION are never touched: the offset is a
translation, and expression is what the eye reads.

  python3 stage4/reik_root.py --motion clip_dyn.npz --out clip_dyn_ik.npz
"""
import argparse, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2")); sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset-key", default="wrench_offset")
    ap.add_argument("--contact-tol", type=float, default=0.006)
    ap.add_argument("--w-contact", type=float, default=400.0)
    ap.add_argument("--w-ankle", type=float, default=8.0)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--w-cont", type=float, default=40.0,
                    help="temporal continuity: pull each frame toward its already-"
                         "solved neighbour. Without it this pass solves every frame "
                         "independently and hops between IK solutions wherever the "
                         "body offset changes - measured on DeadPan, that took the "
                         "hind knees' jerk p99 from 64-72 to 2353-3528 rad/s^3 over "
                         "frames 0-80 (a visible rear-leg shake) and pushed peak leg "
                         "velocity to 11.4 rad/s, past the URDF's 10 rad/s limit. "
                         "Swept on DeadPan (hind-knee jerk p99 over frames 0-80, "
                         "rad/s^3, against 64-72 before this pass ran): w-cont 0 -> "
                         "3528/2353, 12 -> 1346/868, 40 -> 197/190, 120 -> 16/21. "
                         "40 is the knee of the curve; 120 holds the planted paw only "
                         "to 4.1 mm instead of 2.3 mm and buries 3 frames.")
    ap.add_argument("--sweeps", type=int, default=3,
                    help="forward/backward/forward passes, so continuity propagates "
                         "both ways instead of only from the past")
    ap.add_argument("--vel-limit", type=float, default=10.0,
                    help="rad/s. Hard bound between consecutive frames (the v4 leg "
                         "joint velocity limit).")
    ap.add_argument("--w-floor", type=float, default=4000.0,
                    help="no-penetration weight. 800 lost to the contact term "
                         "and left 15 DeadPan frames up to 3.5 mm underground.")
    a = ap.parse_args()

    kin = V4Kin(URDF); cm = ContactModel()
    m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
    if a.offset_key not in m.files:
        raise SystemExit(f"{a.motion} carries no '{a.offset_key}' - nothing to consume")
    DP = m[a.offset_key].astype(float)
    q_orig = m["dof_positions"].astype(float)
    q = q_orig.copy()
    names = [str(x) for x in m["dof_names"]]
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    rp0 = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    # root_pos in the file already includes the offset (wrench_refine wrote it out);
    # recover the pre-offset root so the paw targets are the ORIGINAL world points.
    rp_old = rp0 - DP
    T = len(q)
    ji = {f"{l}_{s}": names.index(f"{l}_{s}") for l in LEGS for s in ("SY_J", "SP_J", "knee")}
    lim = {l: kin.leg_limits(l) for l in LEGS}
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}

    def leg_pts(leg, qleg, R0, p0):
        R, p = R0, p0.copy()
        for nm, qi in zip(kin.leg_chain(leg), qleg):
            J = kin.j[nm]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
        w = hull[leg] @ R.T + p
        j = int(np.argmin(w[:, 2]))
        return w[j], w[:, 2].min(), p          # contact vertex, lowest z, shank-tip frame origin

    lo = np.concatenate([lim[l][:, 0] for l in LEGS])
    hi = np.concatenate([lim[l][:, 1] for l in LEGS])
    live = [i for i in range(T) if np.linalg.norm(DP[i]) >= 1e-9]
    moved = len(live)
    step = a.vel_limit / float(m["fps"])

    def getq(i):
        return np.array([q[i, ji[f"{l}_{s}"]] for l in LEGS
                         for s in ("SY_J", "SP_J", "knee")])

    # Targets are fixed by the ORIGINAL pose, so they are computed once and reused
    # by every sweep; only the solution moves.
    TGT = {}
    for i in live:
        R0 = quat_to_mat(rq[i])
        qo = np.clip(np.concatenate([[q_orig[i, ji[f"{l}_SY_J"]], q_orig[i, ji[f"{l}_SP_J"]],
                                      q_orig[i, ji[f"{l}_knee"]]] for l in LEGS]),
                     lo + 1e-9, hi - 1e-9)
        tc = []; ta = []; on = []
        for k, l in enumerate(LEGS):
            c, z, ap_ = leg_pts(l, qo[3*k:3*k+3], R0, rp_old[i])
            tc.append(c); ta.append(ap_); on.append(z < a.contact_tol)
        TGT[i] = (R0, qo, tc, ta, on)

    for sw in range(max(1, a.sweeps)):
        seq = live if sw % 2 == 0 else live[::-1]
        for n_, i in enumerate(seq):
            R0, qo, tc, ta, on = TGT[i]
            nb = []
            if n_ > 0:
                nb.append(getq(seq[n_ - 1]))
            if n_ + 1 < len(seq):
                nb.append(getq(seq[n_ + 1]))
            x0 = np.clip(getq(i), lo + 1e-9, hi - 1e-9)
            blo, bhi = lo.copy(), hi.copy()
            if nb:                                   # hard velocity bound
                blo = np.maximum(blo, nb[0] - step)
                bhi = np.minimum(bhi, nb[0] + step)
                blo = np.minimum(blo, bhi - 1e-9)
            x0 = np.clip(x0, blo + 1e-10, bhi - 1e-10)

            def resid(x, R0=R0, qo=qo, tc=tc, ta=ta, on=on, nb=nb, i=i):
                r = []
                for k, l in enumerate(LEGS):
                    c, z, ap_ = leg_pts(l, x[3*k:3*k+3], R0, rp0[i])
                    if on[k]:
                        r.extend(a.w_contact * (c - tc[k]))
                    else:
                        r.extend(a.w_ankle * (ap_ - ta[k]))
                    r.append(a.w_floor * min(0.0, z))
                r.extend(a.w_reg * (x - qo))
                for qn in nb:
                    r.extend(a.w_cont * (x - qn))
                return np.array(r)

            res = least_squares(resid, x0, bounds=(blo, bhi),
                                xtol=1e-10, ftol=1e-10, max_nfev=140)
            for k, l in enumerate(LEGS):
                (q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                 q[i, ji[f"{l}_knee"]]) = res.x[3*k:3*k+3]

    # honest after-measurement, against the ORIGINAL world contact points
    err = []; low = np.zeros(T)
    for i in range(T):
        R0 = quat_to_mat(rq[i]); best = 1e9
        for k, l in enumerate(LEGS):
            c0, z0, _ = leg_pts(l, [q_orig[i, ji[f"{l}_SY_J"]], q_orig[i, ji[f"{l}_SP_J"]],
                                    q_orig[i, ji[f"{l}_knee"]]], R0, rp_old[i])
            c1, z1, _ = leg_pts(l, [q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                                    q[i, ji[f"{l}_knee"]]], R0, rp0[i])
            best = min(best, z1)
            if z0 < a.contact_tol:
                err.append(np.linalg.norm(c1 - c0))
        low[i] = best
    dq = np.abs(q[:, :12] - q_orig[:, :12])
    _v = np.abs(np.gradient(q[:, :12], 1.0 / float(m["fps"]), axis=0))
    _j = np.gradient(np.gradient(_v, 1.0 / float(m["fps"]), axis=0),
                     1.0 / float(m["fps"]), axis=0)
    print(f"[[ frames re-solved: {moved}/{T} in {a.sweeps} sweep(s) | "
          f"peak leg velocity {_v.max():.2f} rad/s (limit {a.vel_limit:g}) | "
          f"leg jerk p99 {np.percentile(np.abs(_j), 99):.0f} rad/s^3")
    if err:
        print(f"[[ planted paw kept at its world point: mean {np.mean(err)*1000:.2f} mm "
              f"p95 {np.percentile(err,95)*1000:.2f} max {np.max(err)*1000:.2f} mm")
    print(f"[[ lowest hull z after: min {low.min()*1000:+.2f} mm | "
          f"frames penetrating >1 mm {int((low<-0.001).sum())}/{T}")
    print(f"[[ leg joint change: mean {np.degrees(dq.mean()):.2f} deg max {np.degrees(dq.max()):.2f} deg")
    d["root_quat"] = rq.astype(np.float32)
    d["dof_positions"] = q.astype(np.float32)
    d["dof_velocities"] = np.gradient(q, 1.0 / float(m["fps"]), axis=0).astype(np.float32)
    np.savez(a.out, **d)
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
