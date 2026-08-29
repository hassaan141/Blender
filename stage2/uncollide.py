"""Resolve SELF-COLLISION between the robot's own links, locally, without moving the
paws or the body.

The retargeter constrains the floor and the joint limits and nothing else, so a clip
that folds the legs up under a sitting body drives links through each other. Nothing
downstream notices: PhysX runs with self-collision off, the balance audits only look
at what touches z = 0, and every joint limit is satisfied - the pose is legal and
visibly wrong.

On DeadPan the offender is NOT the torso, which is what it looks like from outside.
It is leg against leg: at frame 0 the hind-left shank is 30.9 mm inside the FRONT-LEFT
THIGH (688 vertices), the hind-right shank 29.7 mm inside the front-right thigh, and
by frame 30 the two right shanks are 22.5 mm inside each other. The front thigh sits
at the body's side, so it reads as the hind leg passing through the body.

Pairs joined by a joint are skipped, and every pair is measured against its ZERO-POSE
value and only the excess is treated as a collision - the *_shoulder_pitch hip
housings sit 37.2 mm inside the torso at rest by design, and the torso 28 mm inside
them, and neither is a defect.

Per affected frame and leg the three leg joints are re-solved to satisfy

    hard   no link of this leg inside any non-adjacent link, plus --clearance
    hard   no hull vertex below the floor
    hard   a paw already on the floor keeps its exact world contact point
    soft   the ankle stays put, the pose stays close, the neighbours stay close

with a hard bound (--max-change) on how far any joint may move, so the correction
stays local instead of flipping the knee to its other branch. Only frames that
actually collide are touched, plus --ramp frames either side, faded in and out.

  python3 stage2/uncollide.py --motion in.npz --out out.npz
"""
import argparse, itertools, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "stage4"))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def outer_polytope(pts, n_dir=600):
    """Half-spaces of a polytope that CONTAINS pts, from its support function.
    Conservative: anything this calls outside really is outside."""
    i = np.arange(n_dir)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n_dir
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    d = np.stack([r * np.cos(phi * i), r * np.sin(phi * i), z], 1)
    return d, -(pts @ d.T).max(0)


def farthest(pts, n):
    if len(pts) <= n:
        return pts
    sel = [int(np.argmax(pts[:, 2]))]
    d2 = ((pts - pts[sel[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(d2)); sel.append(j)
        d2 = np.minimum(d2, ((pts - pts[j]) ** 2).sum(1))
    ext = [int(np.argmin(pts[:, c])) for c in range(3)] + \
          [int(np.argmax(pts[:, c])) for c in range(3)]
    return pts[np.unique(sel + ext)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clearance", type=float, default=0.002)
    ap.add_argument("--ramp", type=int, default=4)
    ap.add_argument("--trigger", type=float, default=0.003)
    ap.add_argument("--max-change", type=float, default=20.0)
    ap.add_argument("--w-pen", type=float, default=4000.0)
    ap.add_argument("--w-floor", type=float, default=4000.0)
    ap.add_argument("--w-contact", type=float, default=1200.0)
    ap.add_argument("--w-ankle", type=float, default=20.0)
    ap.add_argument("--w-reg", type=float, default=2.0)
    ap.add_argument("--w-cont", type=float, default=8.0)
    ap.add_argument("--contact-tol", type=float, default=0.005)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--probe", type=int, default=140)
    ap.add_argument("--only", default=None,
                    help="restrict the FIX to these frame ranges, e.g. '0,112;330,375'; "
                         "collisions are still reported everywhere")
    a = ap.parse_args()

    kin = V4Kin(URDF); cm = ContactModel()
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)
    adj = set()
    for n, j in kin.j.items():
        adj.add((j["parent"], j["child"])); adj.add((j["child"], j["parent"]))

    m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
    q_in = m["dof_positions"].astype(float)
    names = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q_in)
    ji = {f"{l}_{s}": names.index(f"{l}_{s}") for l in LEGS for s in ("SY_J", "SP_J", "knee")}
    lim = {l: kin.leg_limits(l) for l in LEGS}

    MOV = {l: [f"{l}_shoulder_pitch", f"{l}_knee"] for l in LEGS}   # this leg's links
    ALL = [k for k in cm.hull]
    POLY = {k: outer_polytope(cm.hull[k]) for k in ALL}
    PROBE = {k: farthest(cm.hull[k], a.probe) for k in ALL}

    def fk(qq, i):
        dd = {n: qq[i, names.index(n)] for n in names}
        fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], dd.get(jn, 0.0)), pj)
                st.append(J["child"])
        return fr

    def leg_frames(leg, x, fr):
        """(R,p) of this leg's two hulled links, and the shank-tip origin."""
        R, p = fr["origin"]
        out = {}
        for nm_, qi in zip(kin.leg_chain(leg), x):
            J = kin.j[nm_]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
            out[J["child"]] = (R, p)
        return out, p

    # zero-pose baseline for every (obstacle, moving-link) pair
    fr0 = fk(np.zeros((1, 21)), 0) if False else None
    q0z = np.zeros((1, 21)); rp_s, rq_s = rp, rq
    rp, rq = np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])
    fr0 = fk(q0z, 0)
    rp, rq = rp_s, rq_s

    def depth(obst, RA, pA, mov, RB, pB):
        D, b = POLY[obst]
        w = PROBE[mov] @ RB.T + pB
        return float(-(((w - pA) @ RA) @ D.T + b).max(1).min())

    base = {}
    for l in LEGS:
        for mv in MOV[l]:
            for ob in ALL:
                if ob == mv or (ob, mv) in adj:
                    continue
                if ob.startswith(l + "_"):
                    continue          # same leg: its own joint limits govern it
                RA, pA = fr0[ob]; RB, pB = fr0[mv]
                # Clamp at zero. The baseline exists to forgive links that are NESTED
                # at rest (the hip housings sit 38 mm inside the torso). For a pair
                # that is far APART at rest the raw value is a large negative
                # clearance, and subtracting it manufactures a huge fake penetration -
                # that is what produced the 180 mm readings.
                base[(ob, mv)] = max(0.0, depth(ob, RA, pA, mv, RB, pB))

    def leg_depth(l, x, fr):
        """Worst excess penetration of this leg's links into anything non-adjacent."""
        lf, tip = leg_frames(l, x, fr)
        worst = -1e9
        for mv in MOV[l]:
            if mv not in lf:
                continue
            RB, pB = lf[mv]
            for ob in ALL:
                k = (ob, mv)
                if k not in base:
                    continue
                RA, pA = fr[ob]
                worst = max(worst, depth(ob, RA, pA, mv, RB, pB) - base[k])
        return worst, lf, tip

    def paw_low(l, lf):
        R, p = lf[f"{l}_knee"]
        w = cm.hull[f"{l}_knee"] @ R.T + p
        j = int(np.argmin(w[:, 2]))
        return w[j], float(w[j, 2])

    # ---- locate ---------------------------------------------------------------
    q = q_in.copy()
    dep0 = np.zeros((T, 4))
    for i in range(T):
        fr = fk(q, i)
        for k, l in enumerate(LEGS):
            x = np.array([q[i, ji[f"{l}_SY_J"]], q[i, ji[f"{l}_SP_J"]], q[i, ji[f"{l}_knee"]]])
            dep0[i, k] = max(0.0, leg_depth(l, x, fr)[0])
    hit = dep0 > a.trigger
    if a.only:
        keep = np.zeros(T, bool)
        for part in a.only.split(";"):
            s_, e_ = (int(x) for x in part.split(","))
            keep[max(0, s_):min(T, e_ + 1)] = True
        skipped = int((hit & ~keep[:, None]).any(1).sum())
        hit &= keep[:, None]
        print(f"[[ --only {a.only}: leaving {skipped} colliding frame(s) outside it alone")
    print(f"[[ {os.path.basename(a.motion)}: self-collision on "
          f"{int(hit.any(1).sum())}/{T} frames (> {a.trigger*1000:.0f} mm), "
          f"worst {dep0.max()*1000:.1f} mm")
    runs = {}
    for k, l in enumerate(LEGS):
        idx = np.where(hit[:, k])[0]
        rr = []
        for _, g in itertools.groupby(enumerate(idx), lambda t: t[1] - t[0]):
            gg = [x[1] for x in g]
            rr.append([max(0, gg[0] - a.ramp), min(T - 1, gg[-1] + a.ramp)])
        mg = []
        for r in rr:
            if mg and r[0] <= mg[-1][1]:
                mg[-1][1] = max(mg[-1][1], r[1])
            else:
                mg.append(r)
        runs[k] = [tuple(x) for x in mg]
        if runs[k]:
            print(f"[[   {l}: {int(hit[:,k].sum()):3d} frames, max {dep0[:,k].max()*1000:5.1f} mm"
                  f" | " + ", ".join(f"{s}-{e}" for s, e in runs[k][:8]))
    if not hit.any():
        np.savez(a.out, **d); print("[[ nothing to do"); return

    # ---- solve ----------------------------------------------------------------
    q_new = q_in.copy()
    mc = np.radians(a.max_change)
    for sw in range(max(1, a.sweeps)):
        for k, l in enumerate(LEGS):
            for (s0, e0) in runs[k]:
                seq = range(s0, e0 + 1) if sw % 2 == 0 else range(e0, s0 - 1, -1)
                for i in seq:
                    fr = fk(q_new, i)
                    q0 = np.array([q_in[i, ji[f"{l}_SY_J"]], q_in[i, ji[f"{l}_SP_J"]],
                                   q_in[i, ji[f"{l}_knee"]]])
                    lo = np.maximum(lim[l][:, 0], q0 - mc)
                    hi = np.minimum(lim[l][:, 1], q0 + mc)
                    q0c = np.clip(q0, lo + 1e-9, hi - 1e-9)
                    _, lf0, tip0 = leg_depth(l, q0c, fr)
                    c0, low0 = paw_low(l, lf0)
                    grounded = low0 < a.contact_tol
                    nb = [np.array([q_new[j, ji[f"{l}_SY_J"]], q_new[j, ji[f"{l}_SP_J"]],
                                    q_new[j, ji[f"{l}_knee"]]])
                          for j in (i - 1, i + 1) if s0 <= j <= e0]
                    x0 = np.clip(np.array([q_new[i, ji[f"{l}_SY_J"]],
                                           q_new[i, ji[f"{l}_SP_J"]],
                                           q_new[i, ji[f"{l}_knee"]]]), lo + 1e-10, hi - 1e-10)

                    def res(x, fr=fr, q0c=q0c, c0=c0, grounded=grounded, nb=nb,
                            tip0=tip0, l=l):
                        dep, lf, tip = leg_depth(l, x, fr)
                        c, low = paw_low(l, lf)
                        r = [a.w_pen * max(0.0, dep + a.clearance),
                             a.w_floor * min(0.0, low)]
                        if grounded:
                            r.extend(a.w_contact * (c - c0))
                        else:
                            r.extend(a.w_ankle * (tip - tip0))
                        r.extend(a.w_reg * (x - q0c))
                        for qn in nb:
                            r.extend(a.w_cont * (x - qn))
                        return np.array(r)

                    rr_ = least_squares(res, x0, bounds=(lo, hi), xtol=1e-9,
                                        ftol=1e-9, max_nfev=60)
                    (q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                     q_new[i, ji[f"{l}_knee"]]) = rr_.x

    for k, l in enumerate(LEGS):
        cols = [ji[f"{l}_SY_J"], ji[f"{l}_SP_J"], ji[f"{l}_knee"]]
        for (s0, e0) in runs[k]:
            n = e0 - s0 + 1; t = np.arange(n)
            fade = np.minimum(smoothstep((t + 1) / max(1, a.ramp)),
                              smoothstep((n - t) / max(1, a.ramp)))[:, None]
            q_new[s0:e0+1, cols] = (q_in[s0:e0+1, cols]
                                    + fade * (q_new[s0:e0+1, cols] - q_in[s0:e0+1, cols]))

    # ---- honest after-measurement ---------------------------------------------
    dep1 = np.zeros((T, 4)); cerr = []; lowmin = np.zeros(T)
    for i in range(T):
        fr = fk(q_new, i); fro = fk(q_in, i)
        lows = []
        for k, l in enumerate(LEGS):
            x = np.array([q_new[i, ji[f"{l}_SY_J"]], q_new[i, ji[f"{l}_SP_J"]],
                          q_new[i, ji[f"{l}_knee"]]])
            dd, lf, _ = leg_depth(l, x, fr)
            dep1[i, k] = max(0.0, dd)
            c, low = paw_low(l, lf); lows.append(low)
            x0 = np.array([q_in[i, ji[f"{l}_SY_J"]], q_in[i, ji[f"{l}_SP_J"]],
                           q_in[i, ji[f"{l}_knee"]]])
            _, lf0, _ = leg_depth(l, x0, fro)
            c0, low0 = paw_low(l, lf0)
            if low0 < a.contact_tol:
                cerr.append(np.linalg.norm(c - c0))
        lowmin[i] = min(lows)
    dq = np.abs(q_new - q_in)
    ch_ = np.where(dq[:, :12].max(1) > 1e-9)[0]
    print(f"[[ AFTER: self-collision on {int((dep1 > a.trigger).any(1).sum())}/{T} frames, "
          f"worst {dep1.max()*1000:.1f} mm")
    print(f"[[ floor: worst paw penetration {lowmin.min()*1000:+.1f} mm "
          f"({int((lowmin < -0.001).sum())} frames below -1 mm)")
    if cerr:
        print(f"[[ grounded paw kept at its world point: mean {np.mean(cerr)*1000:.2f} mm "
              f"p95 {np.percentile(cerr,95)*1000:.2f} max {np.max(cerr)*1000:.2f} mm")
    print(f"[[ frames touched {len(ch_)}/{T} | leg joint change mean "
          f"{np.degrees(dq[:, :12].mean()):.2f} deg max {np.degrees(dq[:, :12].max()):.2f} deg"
          f" | expressive joints {np.degrees(dq[:, 12:].max()):.4f} deg")
    d["dof_positions"] = q_new.astype(np.float32)
    d["dof_velocities"] = np.gradient(q_new, 1.0 / float(m["fps"]), axis=0).astype(np.float32)
    np.savez(a.out, **d)
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
