"""Stage 4 - ground-lock a conform-rig-authored motion that has NO explicit stance
schedule (Path B: the Blender rig IS the robot, baked straight by bake_conform.py).

`ground_fix.py` assumes a Stage-2 (Path A / Ashley's rig) motion where Ashley's own
foot controls already tell us WHEN a paw is meant to be planted (`source_contacts`,
or a toe-height heuristic against her raw bake). A clip authored directly on the
physical conform rig has no such source, and - measured on
`Bingo_Walk_V4_Upright.blend` - can have NO frame where any paw's world speed drops
into a real stance band: every leg stays near-continuously in motion, so the usual
"low AND slow" contact test used by detect_contacts.py / bake_conform.py finds
essentially nothing (duty factor ~0-2%) even though the joint articulation is a
real, forward-travelling gait.

This script infers a stance window per leg from HEIGHT ALONE (each leg's own
bottom percentile of its collision-hull height over the clip - self-calibrating,
no absolute floor assumption, no speed requirement), then runs the same
root-z + leg-IK least-squares ground solve as ground_fix.py so paws land on the
floor during their inferred stance and never penetrate during swing.

The inferred stance timing is NOT the animator's authored intent (there wasn't
one to read) - it is a best-effort reconstruction from the geometry that exists.
Report the before/after numbers honestly; don't claim this recovers "the" gait.

    python3 stage4/ground_fix_conform.py --motion motions/bingo_walk_v4_upright.npz \
        --out motions/bingo_walk_v4_upright_grounded.npz
"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "stage2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat  # noqa: E402
from contact_model import ContactModel  # noqa: E402

URDF = str(REPO / "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf")
HULLS = str(REPO / "stage4/out/collision_hulls.npz")


def hysteresis(low_ok, strong, min_plant, min_swing):
    T = len(low_ok)
    st = np.zeros(T, bool)
    cur = strong[0]
    for i in range(T):
        cur = low_ok[i] if cur else strong[i]
        st[i] = cur
    for want, mn in ((True, min_plant), (False, min_swing)):
        i = 0
        while i < T:
            j = i
            while j < T and st[j] == st[i]:
                j += 1
            if st[i] == want and (j - i) < mn and i > 0 and j < T:
                st[i:j] = not want
            i = j
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low-frac", type=float, default=0.5,
                     help="permissive (hold) band: bottom fraction of this leg's own "
                          "height range over the clip")
    ap.add_argument("--strong-frac", type=float, default=0.25,
                     help="strict (enter) band: bottom fraction of this leg's own "
                          "height range over the clip")
    ap.add_argument("--w-ground", type=float, default=200.0)
    ap.add_argument("--pen-mult", type=float, default=8.0)
    ap.add_argument("--w-reg", type=float, default=0.5)
    ap.add_argument("--w-rootz", type=float, default=4.0)
    ap.add_argument("--clearance", type=float, default=0.0)
    ap.add_argument("--smooth", type=float, default=1.0)
    a = ap.parse_args()

    kin = V4Kin(URDF)
    cm = ContactModel(path=HULLS)
    m = np.load(a.motion, allow_pickle=True)
    d = {k: m[k] for k in m.files}
    q = m["dof_positions"].astype(float).copy()
    names = [str(x) for x in m["dof_names"]]
    rp = m["body_positions"][:, 0].astype(float).copy()
    rq = m["body_rotations"][:, 0].astype(float)
    T = len(q)
    ji = {f"{l}_{s}": names.index(f"{l}_{s}") for l in LEGS for s in ("SY_J", "SP_J", "knee")}
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}

    def paw_low(leg, qleg, R0, p0):
        R, p = R0, p0.copy()
        for nm, qi in zip(kin.leg_chain(leg), qleg):
            J = kin.j[nm]
            p = p + R @ J["xyz"]
            R = R @ J["R"] @ axis_rot(J["axis"], qi)
        w = hull[leg] @ R.T + p
        j = int(np.argmin(w[:, 2]))
        return w[j, 2], w[j, :2]

    def paw_low_z(leg, qleg, R0, p0):
        return paw_low(leg, qleg, R0, p0)[0]

    # --- self-calibrated stance schedule: height alone, no speed requirement ----
    height = np.zeros((T, 4))
    for i in range(T):
        R0 = quat_to_mat(rq[i])
        for k, l in enumerate(LEGS):
            qleg = [q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]], q[i, ji[f"{l}_knee"]]]
            height[i, k] = paw_low_z(l, qleg, R0, rp[i])

    planted = np.zeros((T, 4), bool)
    for k, l in enumerate(LEGS):
        z = height[:, k]
        lo_z, hi_z = z.min(), z.max()
        low_thr = lo_z + a.low_frac * (hi_z - lo_z)
        strong_thr = lo_z + a.strong_frac * (hi_z - lo_z)
        low_ok = z < low_thr
        strong = z < strong_thr
        fps = float(m["fps"])
        st = hysteresis(low_ok, strong, min_plant=max(2, int(0.10 * fps)),
                         min_swing=max(2, int(0.08 * fps)))
        planted[:, k] = st
        print(f"[[ {l}: height range [{lo_z*1000:.1f},{hi_z*1000:.1f}] mm  "
              f"duty {st.mean()*100:.0f}%")
    print(f"[[ inferred stance (height-only, NOT authored): mean {planted.sum(1).mean():.2f}/4 "
          f"planted per frame")

    lim = {l: kin.leg_limits(l) for l in LEGS}
    lo = np.concatenate([[-0.15]] + [lim[l][:, 0] for l in LEGS])
    hi = np.concatenate([[0.15]] + [lim[l][:, 1] for l in LEGS])

    ch_tree = {}
    for _n, _j in kin.j.items():
        ch_tree.setdefault(_j["parent"], []).append(_n)
    NONLEG = [k for k in cm.hull if not any(k.startswith(l + "_") for l in LEGS)]

    def all_low_z_nonleg(i, qq, rqa, rpa, rootz):
        dd = {n: qq[i, names.index(n)] for n in names}
        fr = {"origin": (quat_to_mat(rqa[i]), np.array([rpa[i, 0], rpa[i, 1], rootz]))}
        st = ["origin"]
        best = 1e9
        while st:
            par = st.pop()
            Rp, pp = fr[par]
            for jn in ch_tree.get(par, []):
                J = kin.j[jn]
                pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], dd.get(jn, 0.0)), pj)
                st.append(J["child"])
        for ln, (R, p) in fr.items():
            if ln in cm.hull and ln in NONLEG:
                best = min(best, float((cm.hull[ln] @ R.T + p)[:, 2].min()))
        return best

    def all_low_z(i, qq, rqa, rpa, rootz):
        dd = {n: qq[i, names.index(n)] for n in names}
        fr = {"origin": (quat_to_mat(rqa[i]), np.array([rpa[i, 0], rpa[i, 1], rootz]))}
        st = ["origin"]
        while st:
            par = st.pop()
            Rp, pp = fr[par]
            for jn in ch_tree.get(par, []):
                J = kin.j[jn]
                pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], dd.get(jn, 0.0)), pj)
                st.append(J["child"])
        return min(float((cm.hull[ln] @ R.T + p)[:, 2].min())
                   for ln, (R, p) in fr.items() if ln in cm.hull)

    q_new = q.copy()
    rz_new = rp[:, 2].copy()
    n_before = 0
    n_after = 0
    tot = int(planted.sum())
    for i in range(T):
        R0 = quat_to_mat(rq[i])
        base = rp[i].copy()
        q0 = np.concatenate([[0.0]] + [[q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                                        q[i, ji[f"{l}_knee"]]] for l in LEGS])
        q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
        nonleg0 = all_low_z_nonleg(i, q, rq, rp, base[2])

        def resid(x, i=i, R0=R0, base=base, q0=q0, nonleg0=nonleg0):
            dz = x[0]
            p0 = base + np.array([0, 0, dz])
            r = []
            for k, l in enumerate(LEGS):
                z, _xy = paw_low(l, x[1 + 3*k:4 + 3*k], R0, p0)
                e = z - a.clearance
                if planted[i, k]:
                    r.append(a.w_ground * max(0.0, e))
                    r.append(a.w_ground * a.pen_mult * min(0.0, e))
                else:
                    r.append(a.w_ground * a.pen_mult * min(0.0, e))
            r.append(a.w_ground * a.pen_mult * min(0.0, nonleg0 + dz - a.clearance))
            r.append(a.w_rootz * dz)
            r.extend(a.w_reg * (x[1:] - q0[1:]))
            return np.array(r)

        z0 = np.array([paw_low_z(l, q0[1+3*k:4+3*k], R0, base) for k, l in enumerate(LEGS)])
        n_before += int(((z0 < 0.005) & planted[i]).sum())
        r = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=120)
        x = r.x
        rz_new[i] = base[2] + x[0]
        for k, l in enumerate(LEGS):
            (q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
             q_new[i, ji[f"{l}_knee"]]) = x[1+3*k:4+3*k]

    if a.smooth > 0:
        rr = int(np.ceil(3 * a.smooth))
        w = np.exp(-0.5 * (np.arange(-rr, rr + 1) / a.smooth) ** 2)
        w /= w.sum()
        for c in range(12):
            q_new[:, c] = np.convolve(np.pad(q_new[:, c], (rr, rr), mode="edge"), w, "valid")
        rz_new = np.convolve(np.pad(rz_new, (rr, rr), mode="edge"), w, "valid")

    # restore contact after smoothing (root-z-only, so it barely perturbs the pose)
    for _pass in range(2):
        dzc = np.zeros(T)
        for i in range(T):
            base = np.array([rp[i, 0], rp[i, 1], rz_new[i]])
            dzc[i] = a.clearance - all_low_z(i, q_new, rq, rp, base[2])
        if a.smooth > 0:
            rr2 = max(1, int(np.ceil(3 * 0.6)))
            w2 = np.exp(-0.5 * (np.arange(-rr2, rr2 + 1) / 0.6) ** 2)
            w2 /= w2.sum()
            dzc = np.convolve(np.pad(dzc, (rr2, rr2), mode="edge"), w2, "valid")
        rz_new = rz_new + dzc

    low_final = np.zeros(T)
    n_final = 0
    for i in range(T):
        R0 = quat_to_mat(rq[i])
        base = np.array([rp[i, 0], rp[i, 1], rz_new[i]])
        z = np.array([paw_low_z(l, [q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                                    q_new[i, ji[f"{l}_knee"]]], R0, base) for l in LEGS])
        n_final += int(((z < 0.005) & planted[i]).sum())
        low_final[i] = all_low_z(i, q_new, rq, rp, rz_new[i])

    dt = 1.0 / float(m["fps"])
    bp = d["body_positions"].astype(float).copy()
    bp[:, 0, 2] = rz_new
    d["body_positions"] = bp.astype(np.float32)
    d["body_linear_velocities"] = np.gradient(bp, dt, axis=0).astype(np.float32)
    d["dof_positions"] = q_new.astype(np.float32)
    d["dof_velocities"] = np.gradient(q_new, dt, axis=0).astype(np.float32)
    d["contacts"] = planted
    d["ground_fix_inferred_stance"] = np.array(True)
    np.savez(a.out, **d)

    dq = np.abs(q_new[:, :12] - q[:, :12])
    print(f"[[ planted paws on floor (of the INFERRED schedule): {n_before}/{tot} -> "
          f"{n_final}/{tot} ({100*n_before/max(tot,1):.0f}% -> {100*n_final/max(tot,1):.0f}%)")
    print(f"[[ lowest point of whole robot per frame: mean {low_final.mean()*1000:+.2f} mm | "
          f"worst float {low_final.max()*1000:+.2f} mm | worst penetration {low_final.min()*1000:+.2f} mm")
    print(f"[[ frames penetrating >1mm: {int((low_final < -0.001).sum())}/{T} | "
          f"floating >2mm: {int((low_final > 0.002).sum())}/{T}")
    print(f"[[ leg joint change: mean {np.degrees(dq.mean()):.2f} deg  max {np.degrees(dq.max()):.2f} deg")
    print(f"[[ root z: before mean {rp[:,2].mean():.4f}  after mean {rz_new.mean():.4f}  "
          f"change mean {np.abs(rz_new-rp[:,2]).mean()*1000:.1f} mm")
    print(f"[[ wrote {a.out}")


main()
