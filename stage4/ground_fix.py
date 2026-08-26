"""Stage 4 - make a retargeted clip GROUND-AWARE, the way Ashley's foot controls are.

Ashley animates with square foot controls that sit ON the ground plane, so her
performance is ground-coherent by construction (Timid: 3.83 of 4 paws down, 12.4 mm
spread). The Stage 2 retarget optimises ANKLE position and treats the paw contact
only as a soft 3D target using the biased PAW_CONTACT_LOCAL point, so that ground
coherence is lost (Timid retarget: 1.51 of 4, 57.8 mm spread) and the robot ends up
physically impossible - floating, or one paw through the floor.

This pass restores it WITHOUT re-authoring the motion. Per frame it solves

    variables : root z, 12 leg joint angles
    strong    : every paw the SOURCE says is planted -> true collision-hull lowest
                point exactly on z = 0
    hard      : no paw below z = 0 (no penetration)
    weak      : stay close to the Stage 3 solution (keeps the performance)

The contact point is the real support function of the convex hull - argmin over hull
vertices of (R v).z - so it stays correct as the shank rotates.
"""
import argparse, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

U = ("/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/"
     "bingo_urdf_w_ear_joints_physics.urdf")
ALEGS = ["aFL", "aFR", "aBL", "aBR"]
MAP = {"fl": "aFL", "fr": "aFR", "bl": "aBL", "br": "aBR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-ground", type=float, default=40.0)
    ap.add_argument("--pen-mult", type=float, default=8.0,
                    help="how much harder penetration is penalised than float. Pulling a "
                         "planted paw toward 0 symmetrically makes least-squares split the "
                         "error above AND below the floor; physics can tolerate a small gap "
                         "but not a buried foot, so the two must not cost the same.")
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--w-rootz", type=float, default=4.0)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--force-planted", default=None,
                    help="a,b : force all four paws planted over frames [a,b]. Stage 4 fix #3 "
                         "(contact-timing). Use for a short flight phase the open-loop "
                         "controller cannot land - it removes the hop, so check the visual cost.")
    ap.add_argument("--clearance", type=float, default=0.0,
                    help="target planted paws this far ABOVE the floor. Starting exactly "
                         "at 0 (or below) makes PhysX resolve an initial penetration and "
                         "punt the robot upward, which is what launched it airborne.")
    a = ap.parse_args()

    kin = V4Kin(U); cm = ContactModel()
    m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
    q = m["dof_positions"].astype(float).copy()
    names = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float).copy()
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q)
    ji = {f"{l}_{s}": names.index(f"{l}_{s}") for l in LEGS for s in ("SY_J", "SP_J", "knee")}

    # --- Ashley's own ground truth: which paws are planted, per frame -----------
    s = np.load(a.source, allow_pickle=True)
    toe = np.stack([s[f"toe_{al}"].astype(float)[:, 2] for al in ALEGS], 1)
    leglen = float(s["rest_lengths"].sum(1).mean())
    g = np.percentile(toe, 3.0)
    planted = (toe - g) < 0.18 * leglen
    if len(planted) != T:
        idx = np.clip(np.round(np.linspace(0, len(planted) - 1, T)).astype(int), 0, len(planted) - 1)
        planted = planted[idx]
    if a.force_planted:
        fa, fb = (int(x) for x in a.force_planted.split(","))
        planted[fa:fb + 1, :] = True
        print(f"[[ contact-timing: forcing all 4 paws planted over frames {fa}-{fb}")
    print(f"[[ source planted paws: mean {planted.sum(1).mean():.2f}/4  "
          f"all-four frames {int((planted.sum(1) == 4).sum())}/{T}")

    lim = {l: kin.leg_limits(l) for l in LEGS}
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}

    def paw_low_z(leg, qleg, R0, p0):
        R, p = R0, p0.copy()
        for nm, qi in zip(kin.leg_chain(leg), qleg):
            J = kin.j[nm]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
        return (hull[leg] @ R.T + p)[:, 2].min()

    ch_tree = {}
    for _n, _j in kin.j.items():
        ch_tree.setdefault(_j["parent"], []).append(_n)

    NONLEG = [k for k in cm.hull if not any(k.startswith(l + "_") for l in LEGS)]

    def all_low_z_nonleg(i, qq, rqa, rpa, rootz):
        d = {n: qq[i, names.index(n)] for n in names}
        fr = {"origin": (quat_to_mat(rqa[i]), np.array([rpa[i, 0], rpa[i, 1], rootz]))}
        st = ["origin"]; best = 1e9
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch_tree.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
                st.append(J["child"])
        for ln, (R, p) in fr.items():
            if ln in cm.hull and ln in NONLEG:
                best = min(best, float((cm.hull[ln] @ R.T + p)[:, 2].min()))
        return best

    def all_low_z(i, qq, rqa, rpa, rootz):
        """Lowest collision-hull z over every link, at frame i."""
        d = {n: qq[i, names.index(n)] for n in names}
        fr = {"origin": (quat_to_mat(rqa[i]), np.array([rpa[i, 0], rpa[i, 1], rootz]))}
        st = ["origin"]
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch_tree.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
                st.append(J["child"])
        return min(float((cm.hull[ln] @ R.T + p)[:, 2].min())
                   for ln, (R, p) in fr.items() if ln in cm.hull)

    lo = np.concatenate([[-0.15]] + [lim[l][:, 0] for l in LEGS])
    hi = np.concatenate([[0.15]] + [lim[l][:, 1] for l in LEGS])

    q_new = q.copy(); rz_new = rp[:, 2].copy()
    n_before = 0; n_after = 0
    for i in range(T):
        R0 = quat_to_mat(rq[i]); base = rp[i].copy()
        q0 = np.concatenate([[0.0]] + [[q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                                        q[i, ji[f"{l}_knee"]]] for l in LEGS])

        def resid(x):
            dz = x[0]; p0 = base + np.array([0, 0, dz]); r = []
            for k, l in enumerate(LEGS):
                z = paw_low_z(l, x[1 + 3*k: 4 + 3*k], R0, p0)
                e = z - a.clearance
                if planted[i, k]:
                    r.append(a.w_ground * max(0.0, e))                 # float: soft
                    r.append(a.w_ground * a.pen_mult * min(0.0, e))    # penetrate: hard
                else:
                    r.append(a.w_ground * a.pen_mult * min(0.0, e))    # swing: only no-penetration
            # torso/head/tail must not sink through the floor either. Without this
            # the solver plants the paws, then the whole-body correction lifts the
            # robot to clear the torso and undoes them (Eccentric: 0 paws down).
            r.append(a.w_ground * a.pen_mult * min(0.0, nonleg0 + dz - a.clearance))
            r.append(a.w_rootz * dz)
            r.extend(a.w_reg * (x[1:] - q0[1:]))
            return np.array(r)

        # Lowest NON-LEG link at dz=0. These links hang off the root only, so their
        # height shifts 1:1 with dz - no need to re-run FK inside the residual.
        nonleg0 = all_low_z_nonleg(i, q, rq, rp, base[2])
        q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)   # Stage 3 sits exactly on limits
        z0 = np.array([paw_low_z(l, q0[1+3*k:4+3*k], R0, base) for k, l in enumerate(LEGS)])
        n_before += int(((z0 < 0.005) & planted[i]).sum())
        r = least_squares(resid, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=120)
        x = r.x; rz_new[i] = base[2] + x[0]
        for k, l in enumerate(LEGS):
            q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]], q_new[i, ji[f"{l}_knee"]] = x[1+3*k:4+3*k]
        p1 = base + np.array([0, 0, x[0]])
        z1 = np.array([paw_low_z(l, x[1+3*k:4+3*k], R0, p1) for k, l in enumerate(LEGS)])
        n_after += int(((z1 < 0.005) & planted[i]).sum())

    # light temporal smoothing to remove per-frame solver chatter
    if a.smooth > 0:
        rr = int(np.ceil(3*a.smooth)); w = np.exp(-0.5*(np.arange(-rr, rr+1)/a.smooth)**2); w /= w.sum()
        for c in range(12):
            q_new[:, c] = np.convolve(np.pad(q_new[:, c], (rr, rr), mode="edge"), w, "valid")
        rz_new = np.convolve(np.pad(rz_new, (rr, rr), mode="edge"), w, "valid")

    # Smoothing above runs AFTER the per-frame ground solve, so it partially undoes
    # it (measured up to 14 mm of reintroduced float). Restore contact with a
    # root-z-only correction - 1 DOF, so it barely perturbs the smoothed motion.
    for _pass in range(2):
        dzc = np.zeros(T)
        for i in range(T):
            R0 = quat_to_mat(rq[i]); base = np.array([rp[i, 0], rp[i, 1], rz_new[i]])
            # Ground on the lowest point of EVERY collision link, not just the
            # planted paws: a swing paw or a thigh can be lower than a planted paw,
            # and that is exactly the penetration Stage 4 physics cannot tolerate.
            dzc[i] = a.clearance - all_low_z(i, q_new, rq, rp, base[2])
        if a.smooth > 0:
            rr2 = max(1, int(np.ceil(3 * 0.6)))
            w2 = np.exp(-0.5 * (np.arange(-rr2, rr2 + 1) / 0.6) ** 2); w2 /= w2.sum()
            dzc = np.convolve(np.pad(dzc, (rr2, rr2), mode="edge"), w2, "valid")
        rz_new = rz_new + dzc

    # final, HONEST measurement - after every modification
    n_final = 0; low_final = np.zeros(T)
    for i in range(T):
        R0 = quat_to_mat(rq[i]); base = np.array([rp[i, 0], rp[i, 1], rz_new[i]])
        z = np.array([paw_low_z(l, [q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                                    q_new[i, ji[f"{l}_knee"]]], R0, base) for l in LEGS])
        n_final += int(((z < 0.005) & planted[i]).sum())
        low_final[i] = all_low_z(i, q_new, rq, rp, rz_new[i])

    rp[:, 2] = rz_new
    dq = np.abs(q_new[:, :12] - q[:, :12])
    d["dof_positions"] = q_new.astype(np.float32)
    bp = d["body_positions"].astype(float).copy(); bp[:, 0, 2] = rz_new
    d["body_positions"] = bp.astype(np.float32)
    d["root_pos"] = rp.astype(np.float32); d["root_quat"] = rq.astype(np.float32)
    dt = 1.0 / float(m["fps"])
    d["dof_velocities"] = np.gradient(q_new, dt, axis=0).astype(np.float32)
    np.savez(a.out, **d)
    tot = int(planted.sum())
    print(f"[[ planted paws ON the floor (measured AFTER smoothing): "
          f"{n_before}/{tot} -> {n_final}/{tot}  ({100*n_before/tot:.0f}% -> {100*n_final/tot:.0f}%)")
    print(f"[[ lowest point of WHOLE robot per frame: mean {low_final.mean()*1000:+.2f} mm | "
          f"worst FLOAT {low_final.max()*1000:+.2f} mm | worst PENETRATION {low_final.min()*1000:+.2f} mm")
    print(f"[[ frames penetrating >1mm: {int((low_final < -0.001).sum())}/{T}  "
          f"| floating >2mm: {int((low_final > 0.002).sum())}/{T}")
    print(f"[[ leg joint change: mean {np.degrees(dq.mean()):.2f} deg  max {np.degrees(dq.max()):.2f} deg")
    print(f"[[ root z change   : mean {np.abs(rz_new-m['body_positions'][:,0,2]).mean()*1000:.1f} mm")
    print(f"[[ wrote {a.out}")


main()
