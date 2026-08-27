"""Stage 4 - exact contact-wrench feasibility audit, and the minimal body offset
that restores it.

Method adapted from research/opt-mimic-traj-opt (centroidal dynamics + friction
cone + unilateral contact). Only the METHOD is borrowed; every physical number
here comes from the Bingo v4 URDF, the validated collision hulls, or the Stage-4
ground material.

Why this replaces the cart-table ZMP in stage4/dynamic_audit.py
---------------------------------------------------------------
Cart-table assumes a point mass and ignores the limbs' angular momentum. Bingo's
head alone is 29% of its mass and the legs swing hard, so that approximation is
not good enough here. Because every Bingo contact lies on the same plane z=0, the
full 6-D contact-wrench feasibility problem collapses EXACTLY (no approximation)
to three conditions:

    F   = M * (a_com + g z)                     required net contact force
    Mo  = hdot + c x F                          required net moment about the origin
    CoP = (-Mo_y / F_z, Mo_x / F_z)             centre of pressure on z=0

    (1) F_z > 0                                 contact cannot pull
    (2) |F_xy| <= mu * F_z                      friction cone
    (3) CoP inside the convex hull of the loaded feet

hdot is the derivative of the exact centroidal angular momentum, summed over every
link with its URDF inertia - not a single-rigid-body stand-in.

Refinement
----------
Only the body offset dp(t) is free; joints, orientation, contact schedule and foot
placement are untouched, so the authored performance survives. c = c_ref + dp and
a = d2c/dt2 are affine in dp, and multiplying (3) through by F_z > 0 turns it into

    n_x * (-Mo_y) + n_y * (Mo_x) <= F_z * (n . q)

which is also affine. The whole problem is therefore a convex QP: minimise
||dp||^2 + smoothness subject to linear feasibility, with per-frame slack so an
impossible frame is REPORTED rather than silently distorting its neighbours.

  python3 stage4/wrench_refine.py --motion stage2/out/cheeky_retarget.npz   # audit
  python3 stage4/wrench_refine.py --motion ... --out ... --cap 0.04         # refine
"""
import argparse, itertools, os, sys
import numpy as np
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2")); sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel
from support_audit import hull2d, margin, polygon_area

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")
G = 9.81


def link_inertials(path):
    out = {}
    for ln in ET.parse(path).getroot().findall("link"):
        inr = ln.find("inertial")
        if inr is None:
            continue
        o = inr.find("origin"); it = inr.find("inertia")
        c = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        I = np.array([[float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
                      [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
                      [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))]])
        out[ln.get("name")] = (float(inr.find("mass").get("value")), c, I)
    return out


def centroidal(motion, tol):
    """Exact CoM, centroidal angular momentum, and the loaded-foot support set."""
    kin = V4Kin(URDF); cm = ContactModel(); LI = link_inertials(URDF)
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)
    m = np.load(motion, allow_pickle=True)
    q = m["dof_positions"].astype(float); nm = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q); fps = float(m["fps"]); dt = 1.0 / fps
    names = list(LI)
    P = {l: np.zeros((T, 3)) for l in names}
    R = {l: np.zeros((T, 3, 3)) for l in names}
    sup = []                      # world contact points (all links) per frame
    for i in range(T):
        d = {n: q[i, nm.index(n)] for n in nm}
        fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
                st.append(J["child"])
        for l in names:
            R[l][i], P[l][i] = fr[l]
        pts = []
        for ln, (Rk, pk) in fr.items():
            if ln not in cm.hull:
                continue
            w = cm.hull[ln] @ Rk.T + pk
            sel = w[w[:, 2] < tol]
            if len(sel):
                pts.append(sel[:, :2])
        sup.append(hull2d(np.concatenate(pts, 0)) if pts else np.zeros((0, 2)))
    M = sum(LI[l][0] for l in names)
    Cl = {l: np.einsum("tij,j->ti", R[l], LI[l][1]) + P[l] for l in names}
    C = sum(LI[l][0] * Cl[l] for l in names) / M
    V = {l: np.gradient(Cl[l], dt, axis=0) for l in names}
    h = np.zeros((T, 3))
    for l in names:
        mm, _, Il = LI[l]
        dR = np.gradient(R[l], dt, axis=0)
        S = np.einsum("tij,tkj->tik", dR, R[l])
        w = np.stack([S[:, 2, 1] - S[:, 1, 2], S[:, 0, 2] - S[:, 2, 0],
                      S[:, 1, 0] - S[:, 0, 1]], 1) * 0.5
        Iw = np.einsum("tij,jk,tlk->til", R[l], Il, R[l])
        h += np.einsum("tij,tj->ti", Iw, w) + mm * np.cross(Cl[l] - C, V[l])
    return dict(M=M, C=C, h=h, sup=sup, T=T, fps=fps, dt=dt, rp=rp, rq=rq, motion=m)


def evaluate(C, hdot, M, sup, dt, mu):
    """Per-frame CoP, its margin inside the support hull, and the required mu."""
    T = len(C)
    a = np.gradient(np.gradient(C, dt, axis=0), dt, axis=0)
    F = M * (a + np.array([0.0, 0.0, G]))
    Mo = hdot + np.cross(C, F)
    Fz = F[:, 2]
    ok = Fz > 1e-6
    cop = np.full((T, 2), np.nan)
    cop[ok, 0] = -Mo[ok, 1] / Fz[ok]
    cop[ok, 1] = Mo[ok, 0] / Fz[ok]
    mg = np.array([margin(sup[i], cop[i]) if ok[i] and len(sup[i]) else -9.99
                   for i in range(T)])
    mu_req = np.where(ok, np.linalg.norm(F[:, :2], axis=1) / np.maximum(Fz, 1e-9), np.inf)
    return dict(cop=cop, margin=mg, mu=mu_req, Fz=Fz, F=F, Mo=Mo, airborne=~ok)


def lp_feasible(pts, F, Mo, mu):
    """Exact 6-D contact-wrench feasibility by LP.

    The CoP test above covers the moment about the HORIZONTAL axes only. It says
    nothing about the vertical moment M_z, which a turn needs and which has to come
    entirely out of the tangential forces - and with one or two nearly collinear
    contact points there may be no way to produce it. Laidback's 90 deg turn is
    exactly that case, so the CoP test scores it feasible while physics cannot
    perform it. This solves, per frame,

        min ||s||_1  s.t.  sum f_i = F + s[0:3],  sum p_i x f_i = Mo + s[3:6],
                           f_iz >= 0,  |f_ix| <= mu f_iz,  |f_iy| <= mu f_iz

    over the contact-set hull vertices. s = 0 means a legal force distribution
    exists. The pyramid is the inscribed (conservative) approximation of the cone.
    Returns (residual force N, residual moment N m, residual yaw moment N m).
    """
    from scipy.optimize import linprog
    n = len(pts)
    if n == 0:
        return np.inf, np.inf, np.inf
    nv = 3 * n + 6                       # f_i, then |s| bounds as 6 extra vars
    Aeq = np.zeros((6, nv)); beq = np.concatenate([F, Mo])
    for i, p in enumerate(pts):
        Aeq[0:3, 3*i:3*i+3] = np.eye(3)
        # p x f for a contact on the floor, p = (px, py, 0):
        #   (py*fz,  -px*fz,  px*fy - py*fx)
        Aeq[3:6, 3*i:3*i+3] = np.array([[0.0, 0.0, p[1]],
                                        [0.0, 0.0, -p[0]],
                                        [-p[1], p[0], 0.0]])
    Aeq[0:6, 3*n:3*n+6] = -np.eye(6)     # slack, free sign, cost on |s|
    ub = np.zeros((4 * n, nv)); 
    for i in range(n):
        ub[4*i+0, 3*i+0] = 1.0;  ub[4*i+0, 3*i+2] = -mu
        ub[4*i+1, 3*i+0] = -1.0; ub[4*i+1, 3*i+2] = -mu
        ub[4*i+2, 3*i+1] = 1.0;  ub[4*i+2, 3*i+2] = -mu
        ub[4*i+3, 3*i+1] = -1.0; ub[4*i+3, 3*i+2] = -mu
    bounds = []
    for i in range(n):
        bounds += [(None, None), (None, None), (0.0, None)]
    bounds += [(None, None)] * 6
    # |s| is linearised with 6 auxiliary variables t >= |s| and cost on t.
    nv2 = nv + 6
    Aeq2 = np.hstack([Aeq, np.zeros((6, 6))])
    ub2 = np.hstack([ub, np.zeros((4 * n, 6))])
    extra = np.zeros((12, nv2))
    for k in range(6):
        extra[2*k, 3*n+k] = 1.0;  extra[2*k, nv+k] = -1.0     #  s_k <= t_k
        extra[2*k+1, 3*n+k] = -1.0; extra[2*k+1, nv+k] = -1.0  # -s_k <= t_k
    A_ub = np.vstack([ub2, extra]); b_ub = np.zeros(len(A_ub))
    c = np.zeros(nv2); c[nv:nv+3] = 1.0; c[nv+3:nv+6] = 10.0    # moments in N m
    bounds2 = bounds + [(0.0, None)] * 6
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq2, b_eq=beq, bounds=bounds2,
                method="highs")
    if not r.success:
        return np.inf, np.inf, np.inf
    s = r.x[3*n:3*n+6]
    return float(np.linalg.norm(s[:3])), float(np.linalg.norm(s[3:])), float(abs(s[5]))


def report(tag, ev, mu, T):
    mg, mr, ab = ev["margin"], ev["mu"], ev["airborne"]
    feas = (mg > 0) & (mr <= mu) & ~ab
    print(f"[[ {tag}: wrench-feasible {int(feas.sum())}/{T} = {100*feas.mean():.0f}%"
          f" | CoP margin median {np.median(mg)*1000:+.1f} mm worst {mg.min()*1000:+.0f} mm"
          f" | mu_req p90 {np.percentile(mr[np.isfinite(mr)],90):.2f} "
          f"max {np.nanmax(mr[np.isfinite(mr)]):.2f} | frames mu>{mu:g}: {int((mr>mu).sum())}"
          f" | airborne {int(ab.sum())}")
    bad = np.where(~feas)[0]
    if len(bad):
        r = []
        for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
            gg = [x[1] for x in g]; r.append((gg[0], gg[-1]))
        print("     infeasible windows: " + ", ".join(f"{x}-{y}" for x, y in r[:14])
              + (" ..." if len(r) > 14 else ""))
    return feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", default=None, help="omit to audit only")
    ap.add_argument("--cap", type=float, default=0.04, help="max |body offset|, metres")
    ap.add_argument("--cap-z", type=float, default=0.005,
                    help="separate, tighter bound on the VERTICAL part. Feasibility is "
                         "won almost entirely by moving the body horizontally over its "
                         "support; letting the solver drop it instead just buries the "
                         "hulls (measured on DeadPan: 15 frames up to 3.5 mm through "
                         "the floor after reik_root).")
    ap.add_argument("--mu", type=float, default=0.5,
                    help="the Stage-4 ground plane's actual friction (IsaacLab "
                         "GroundPlaneCfg default 0.5). Do not raise it here to buy "
                         "feasibility the simulator will not grant.")
    ap.add_argument("--margin", type=float, default=0.005,
                    help="CoP must sit this far inside the support hull")
    ap.add_argument("--contact-tol", type=float, default=0.006)
    ap.add_argument("--lp", action="store_true",
                    help="also run the exact 6-D LP feasibility test per frame "
                         "(includes the vertical/yaw moment the CoP test misses)")
    ap.add_argument("--w-smooth", type=float, default=200.0)
    ap.add_argument("--w-offset", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.006,
                    help="metres/frame limit on the offset. Without it the solver "
                         "teleports the body between a frame it can fix and a "
                         "neighbouring flight frame it cannot.")
    a = ap.parse_args()

    D = centroidal(a.motion, a.contact_tol)
    T, dt, M = D["T"], D["dt"], D["M"]
    hdot = np.gradient(D["h"], dt, axis=0)
    print(f"[[ {os.path.basename(a.motion)}  T={T} @ {D['fps']:g} fps  mass {M:.4f} kg  "
          f"mu {a.mu}  support tol {a.contact_tol*1000:.0f} mm")
    ev0 = evaluate(D["C"], hdot, M, D["sup"], dt, a.mu)
    report("BEFORE", ev0, a.mu, T)
    if a.lp:
        rf = np.zeros(T); rm = np.zeros(T); ry = np.zeros(T)
        for i in range(T):
            rf[i], rm[i], ry[i] = lp_feasible(D["sup"][i], ev0["F"][i], ev0["Mo"][i], a.mu)
        ok = (rf < 1e-3) & (rm < 1e-4)
        print(f"[[ EXACT 6-D LP: feasible {int(ok.sum())}/{T} = {100*ok.mean():.0f}% | "
              f"residual force p95 {np.percentile(rf[np.isfinite(rf)],95):.3f} N | "
              f"moment p95 {np.percentile(rm[np.isfinite(rm)],95):.4f} N m | "
              f"YAW-moment residual mean {np.mean(ry[np.isfinite(ry)]):.4f} "
              f"max {np.max(ry[np.isfinite(ry)]):.4f} N m")
        bad = np.where(~ok)[0]
        if len(bad):
            r = []
            for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
                gg = [x[1] for x in g]; r.append((gg[0], gg[-1]))
            print("     LP-infeasible windows: " + ", ".join(f"{x}-{y}" for x, y in r[:14])
                  + (" ..." if len(r) > 14 else ""))
    if not a.out:
        return

    import casadi as ca
    opti = ca.Opti()
    dp = opti.variable(3, T); opti.set_initial(dp, 0.0)
    C = ca.DM(D["C"].T) + dp
    J = a.w_offset * ca.sumsqr(dp)
    for i in range(1, T):
        J += a.w_smooth * ca.sumsqr(dp[:, i] - dp[:, i - 1])
    sl = opti.variable(1, T); opti.subject_to(sl >= 0); J += 1e5 * ca.sumsqr(sl)
    nfix = 0
    for i in range(T):
        im, ip = max(i - 1, 0), min(i + 1, T - 1)
        if i == 0 or i == T - 1:
            opti.subject_to(dp[:, i] == 0); nfix += 1
            continue
        if len(D["sup"][i]) < 3 or ev0["airborne"][i]:
            # Airborne or degenerate support: no contact force exists to shape, so
            # there is nothing to constrain. Leave the frame FREE (the smoothness
            # term carries the offset across it) rather than pinning it to zero,
            # which would make the body jump in and out of the correction.
            nfix += 1
            continue
        acc = (C[:, ip] - 2 * C[:, i] + C[:, im]) / (dt * dt)
        F = M * (acc + ca.DM([0, 0, G]))
        Mo = ca.DM(hdot[i]) + ca.cross(C[:, i], F)
        Fz = F[2]
        opti.subject_to(Fz >= 0.05 * M * G)          # keep the contact loaded
        P = D["sup"][i]
        # The margin has to fit inside the polygon. A two-paw stance is a SLIVER a
        # few millimetres wide once the two hulls' contact vertices are hulled
        # together, and demanding 8-20 mm inside it is simply infeasible - the
        # solver then spends its slack there and gives up on the frame, which is
        # exactly the frame that needs help. Scale the request down to a fraction of
        # the shape's own inradius, approximated for a convex polygon by
        # 2 * area / perimeter.
        _A = polygon_area(P)
        _L = sum(float(np.linalg.norm(P[k] - P[(k + 1) % len(P)])) for k in range(len(P)))
        _r = 2.0 * _A / max(_L, 1e-9)
        mg = min(a.margin, 0.4 * _r)
        for k in range(len(P)):
            q0, q1 = P[k], P[(k + 1) % len(P)]
            e = q1 - q0; n = np.array([e[1], -e[0]]); n /= np.linalg.norm(n) + 1e-12
            # hull2d returns CCW, so the OUTWARD normal is (e_y, -e_x)
            opti.subject_to(n[0] * (-Mo[1]) + n[1] * Mo[0]
                            <= Fz * float(n @ q0) - mg * Fz + sl[i])
        opti.subject_to(F[0] <= a.mu * Fz + sl[i]); opti.subject_to(-F[0] <= a.mu * Fz + sl[i])
        opti.subject_to(F[1] <= a.mu * Fz + sl[i]); opti.subject_to(-F[1] <= a.mu * Fz + sl[i])
        opti.subject_to(ca.sumsqr(dp[:, i]) <= a.cap ** 2)
    for i in range(T):
        opti.subject_to(ca.sumsqr(dp[:, i]) <= a.cap ** 2)
        if i:
            opti.subject_to(ca.sumsqr(dp[:, i] - dp[:, i - 1]) <= a.step ** 2)
    opti.minimize(J)
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "max_iter": 3000, "tol": 1e-7, "acceptable_tol": 1e-5})
    try:
        sol = opti.solve(); conv = True
    except RuntimeError:
        sol = opti.debug; conv = False
        print("[[ WARNING ipopt did not converge - reporting the last iterate")
    DP = np.array(sol.value(dp)).reshape(3, T).T
    SL = np.array(sol.value(sl)).reshape(-1)
    print(f"[[ frames left un-refined (flight / degenerate support / ends): {nfix}")
    print(f"[[ body offset: mean {np.linalg.norm(DP,axis=1).mean()*1000:.1f} mm  "
          f"max {np.linalg.norm(DP,axis=1).max()*1000:.1f} mm (cap {a.cap*1000:.0f} mm)  "
          f"| max step {np.linalg.norm(np.diff(DP,axis=0),axis=1).max()*1000:.1f} mm/frame")
    print(f"[[ unmet constraint slack: max {SL.max():.4f} on "
          f"{int((SL>1e-4).sum())} frames (0 = fully feasible)")
    ev1 = evaluate(D["C"] + DP, hdot, M, D["sup"], dt, a.mu)
    report("AFTER ", ev1, a.mu, T)
    m = D["motion"]
    out = {k: m[k] for k in m.files}
    rp = D["rp"] + DP
    # Always write the ROOT PAIR together. The bake_conform schema carries the root
    # only inside body_positions/body_rotations; adding a lone root_pos makes every
    # downstream reader take the root_pos branch and then fail looking for root_quat.
    out["root_pos"] = rp.astype(np.float32)
    out["root_quat"] = D["rq"].astype(np.float32)
    if "body_positions" in out:
        bp = out["body_positions"].astype(float).copy(); bp[:, 0] = rp
        out["body_positions"] = bp.astype(np.float32)
    out["wrench_offset"] = DP.astype(np.float32)
    out["wrench_mu"] = np.array(a.mu)
    np.savez(a.out, **out)
    print(f"[[ wrote {a.out}  (body offset only; re-run the leg IK to consume it)")


if __name__ == "__main__":
    main()
