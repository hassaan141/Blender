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
    ap.add_argument("--anchor-sigma", type=float, default=0.0,
                    help="0 = one CONSTANT anchor per stance run (its median). >0 = a "
                         "Gaussian-smoothed version of the incoming path with this sigma "
                         "in frames. Constant is the physically correct target, but where "
                         "the leg cannot reach it (DeadPan's 134-frame hind stance drifts "
                         "99 mm because the body out-travels the leg) the solver chatters "
                         "against it and slip goes UP. Smoothing removes the chatter "
                         "without demanding the impossible.")
    ap.add_argument("--restore", type=int, default=1,
                    help="leg-only re-projection passes AFTER the temporal smoothing. "
                         "The Gaussian that removes per-frame solver chatter also blurs "
                         "the contact solution back out - measured, it re-introduced "
                         "~40%% of the planted-paw slip the pass had just removed, and "
                         "up to 14 mm of float. This pass restores the floor contact and "
                         "the stance XY anchor without re-smoothing. 0 = old behaviour.")
    ap.add_argument("--force-planted", default=None,
                    help="a,b : force all four paws planted over frames [a,b]. Stage 4 fix #3 "
                         "(contact-timing). Use for a short flight phase the open-loop "
                         "controller cannot land - it removes the hop, so check the visual cost. "
                         "Several windows may be given, separated by ';'. "
                         "stage4/support_windows.py prints them ready to paste.")
    ap.add_argument("--force-legs", default=None,
                    help="restrict --force-planted to these legs, e.g. 'bl,br'. Eccentric "
                         "is the case: Ashley holds the hind paws 57-82 mm off the floor "
                         "through the whole sit, which leaves the robot balanced on the "
                         "rear EDGE of its torso hull - a line contact with no roll "
                         "support, and it rolls over by frame 36. Planting only the hind "
                         "paws turns that line into a 337 cm^2 polygon.")
    ap.add_argument("--w-anchor", type=float, default=0.0,
                    help="weight on holding a PLANTED paw at a constant world XY over its "
                         "whole stance run. OFF by default because it MEASURABLY REGRESSES: "
                         "solve_spatial_retarget already locks the stance anchors, and a "
                         "second anchor derived independently here fights that solution "
                         "instead of reinforcing it. Measured on DeadPan (--restore 1, "
                         "stage2/slip_audit.py) - w-anchor 0: 100% of planted paws on the "
                         "floor, 1.20 mm/frame material slip, 26.7 deg max pose change; "
                         "w-anchor 60: 97%, 1.83 mm/frame, 88.3 deg. Keep it at 0 unless the "
                         "incoming motion has NOT been through the Stage-2 contact solve.")
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
    # Prefer the Stage-2 contact schedule when the motion carries it: those are the
    # same runs solve_spatial_retarget locked its world anchors to, so the two
    # passes then agree on WHEN a paw is planted. The height heuristic below is a
    # looser fallback (0.18 x leg length) kept for motions that predate the field.
    if "source_contacts" in m.files:
        planted = np.asarray(m["source_contacts"], bool).copy()
        print("[[ stance schedule: Stage-2 source_contacts (authoritative)")
    else:
        s = np.load(a.source, allow_pickle=True)
        toe = np.stack([s[f"toe_{al}"].astype(float)[:, 2] for al in ALEGS], 1)
        leglen = float(s["rest_lengths"].sum(1).mean())
        g = np.percentile(toe, 3.0)
        planted = (toe - g) < 0.18 * leglen
        print("[[ stance schedule: source toe-height heuristic (no source_contacts in motion)")
    if len(planted) != T:
        idx = np.clip(np.round(np.linspace(0, len(planted) - 1, T)).astype(int), 0, len(planted) - 1)
        planted = planted[idx]
    if a.force_planted:
        default_legs = ([LEGS.index(x) for x in a.force_legs.split(",")] if a.force_legs
                        else list(range(4)))
        spans = []
        # "10,14;77,80"            -> those windows, --force-legs (or all four)
        # "10,14:fl,bl;77,80:fr"   -> per-window leg lists, which is what
        #                             stage4/extend_stance.py emits: the minimum set
        #                             that turns a two-paw line into a polygon.
        for part in a.force_planted.split(";"):
            if ":" in part:
                rng, legspec = part.split(":", 1)
                legs_f = [LEGS.index(x) for x in legspec.split(",")]
            else:
                rng, legs_f = part, default_legs
            fa, fb = (int(x) for x in rng.split(","))
            for _k in legs_f:
                planted[fa:fb + 1, _k] = True
            spans.append(f"{fa}-{fb}[{'+'.join(LEGS[k] for k in legs_f)}]")
        print(f"[[ contact-timing: forcing {len(spans)} window(s): {', '.join(spans)}")
    print(f"[[ source planted paws: mean {planted.sum(1).mean():.2f}/4  "
          f"all-four frames {int((planted.sum(1) == 4).sum())}/{T}")

    lim = {l: kin.leg_limits(l) for l in LEGS}
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}

    def paw_low(leg, qleg, R0, p0):
        """(lowest world z, world xy of that lowest hull vertex)."""
        R, p = R0, p0.copy()
        for nm, qi in zip(kin.leg_chain(leg), qleg):
            J = kin.j[nm]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
        w = hull[leg] @ R.T + p
        j = int(np.argmin(w[:, 2]))
        return w[j, 2], w[j, :2]

    def paw_low_z(leg, qleg, R0, p0):
        return paw_low(leg, qleg, R0, p0)[0]

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

    # --- stance world-XY anchors ------------------------------------------------
    # One constant horizontal contact position per contiguous planted run, taken as
    # the MEDIAN of what the incoming motion already achieves. Median (not the first
    # frame) so the correction is shared symmetrically across the run and the
    # authored placement is preserved on average.
    anchor_xy = np.full((T, 4, 2), np.nan)
    if a.w_anchor > 0:
        for k, l in enumerate(LEGS):
            xy0 = np.array([paw_low(l, [q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                                        q[i, ji[f"{l}_knee"]]],
                                    quat_to_mat(rq[i]), rp[i])[1] for i in range(T)])
            i = 0
            while i < T:
                if not planted[i, k]:
                    i += 1; continue
                j = i + 1
                while j < T and planted[j, k]:
                    j += 1
                if a.anchor_sigma > 0 and (j - i) > 2:
                    sg = a.anchor_sigma
                    rr = int(np.ceil(3 * sg))
                    w = np.exp(-0.5 * (np.arange(-rr, rr + 1) / sg) ** 2); w /= w.sum()
                    seg = xy0[i:j]
                    anchor_xy[i:j, k] = np.stack(
                        [np.convolve(np.pad(seg[:, c], (rr, rr), mode="edge"), w, "valid")
                         for c in range(2)], 1)
                else:
                    anchor_xy[i:j, k] = np.median(xy0[i:j], axis=0)
                i = j
        drift = np.array([np.linalg.norm(anchor_xy[i, k] - paw_low(
                              LEGS[k], [q[i, ji[f"{LEGS[k]}_SY_J"]], q[i, ji[f"{LEGS[k]}_SP_J"]],
                                        q[i, ji[f"{LEGS[k]}_knee"]]],
                              quat_to_mat(rq[i]), rp[i])[1])
                          for i in range(T) for k in range(4) if planted[i, k]])
        print(f"[[ stance XY anchors: {int(np.isfinite(anchor_xy[:,:,0]).sum())} planted foot-frames | "
              f"incoming drift from anchor mean {drift.mean()*1000:.1f} mm max {drift.max()*1000:.1f} mm")

    q_new = q.copy(); rz_new = rp[:, 2].copy()
    n_before = 0; n_after = 0
    for i in range(T):
        R0 = quat_to_mat(rq[i]); base = rp[i].copy()
        q0 = np.concatenate([[0.0]] + [[q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]],
                                        q[i, ji[f"{l}_knee"]]] for l in LEGS])

        def resid(x):
            dz = x[0]; p0 = base + np.array([0, 0, dz]); r = []
            for k, l in enumerate(LEGS):
                z, xy = paw_low(l, x[1 + 3*k: 4 + 3*k], R0, p0)
                e = z - a.clearance
                if planted[i, k]:
                    r.append(a.w_ground * max(0.0, e))                 # float: soft
                    r.append(a.w_ground * a.pen_mult * min(0.0, e))    # penetrate: hard
                    if a.w_anchor > 0 and np.isfinite(anchor_xy[i, k, 0]):
                        r.extend(a.w_anchor * (xy - anchor_xy[i, k]))  # no skating
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

    # Re-project the SMOOTHED pose back onto its contact constraints, legs only.
    # Same residual as the main solve minus the root-z variable, and regularised
    # toward the smoothed pose, so it corrects contact without undoing smoothing.
    for _rp in range(max(0, a.restore)):
        for i in range(T):
            R0 = quat_to_mat(rq[i]); p0 = np.array([rp[i, 0], rp[i, 1], rz_new[i]])
            qs = np.concatenate([[q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                                  q_new[i, ji[f"{l}_knee"]]] for l in LEGS])
            qs = np.clip(qs, lo[1:] + 1e-9, hi[1:] - 1e-9)

            def rres(x, i=i, R0=R0, p0=p0, qs=qs):
                r = []
                for k, l in enumerate(LEGS):
                    z, xy = paw_low(l, x[3*k:3*k+3], R0, p0)
                    e = z - a.clearance
                    if planted[i, k]:
                        r.append(a.w_ground * max(0.0, e))
                        r.append(a.w_ground * a.pen_mult * min(0.0, e))
                        if a.w_anchor > 0 and np.isfinite(anchor_xy[i, k, 0]):
                            r.extend(a.w_anchor * (xy - anchor_xy[i, k]))
                    else:
                        r.append(a.w_ground * a.pen_mult * min(0.0, e))
                r.extend(a.w_reg * (x - qs))
                return np.array(r)

            rr = least_squares(rres, qs, bounds=(lo[1:], hi[1:]),
                               xtol=1e-10, ftol=1e-10, max_nfev=100)
            for k, l in enumerate(LEGS):
                (q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                 q_new[i, ji[f"{l}_knee"]]) = rr.x[3*k:3*k+3]

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
    # Record the stance mask this pass actually enforced. With --force-planted it
    # differs from Ashley's source_contacts, and every downstream audit (slip,
    # wrench feasibility) must score the reference against the schedule it was
    # built to, not the one it no longer follows.
    d["stage4_planted"] = np.asarray(planted, bool)
    d["dof_positions"] = q_new.astype(np.float32)
    if "body_positions" in d:      # AMP-style schema; the Stage-2 solver output has none
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
    z_in = (m["body_positions"][:, 0, 2] if "body_positions" in m.files else m["root_pos"][:, 2])
    print(f"[[ root z change   : mean {np.abs(rz_new-z_in).mean()*1000:.1f} mm")
    print(f"[[ wrote {a.out}")


main()
