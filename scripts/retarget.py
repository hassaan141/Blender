"""Stage 2+3: raw Blender Cartesian motion -> Bingo motion .npz (spec A.7).

Pipeline:
  axis convert -> hip-relative scale -> rigid base fit -> per-leg analytic-ish IK
  -> limit clamp -> head/tail decompose -> velocities -> pack

Joint angles come out already in URDF sign convention, because FK is built
directly from the URDF's own axis vectors. The A.3 sign map is therefore never
applied by hand -- it is verified as a check instead.

Usage:
    python3 scripts/retarget.py raw/deadpan_raw.json --urdf bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf \
        --out motions/bingo_deadpan.npz [--start-t 5.5 --end-t 8.0]
"""

import json, argparse, sys
import numpy as np
import xml.etree.ElementTree as ET

SHANK_LEN = 0.120                      # spec A.2 / A.6 -- must equal bingo_amp_env.py
LEGS = ["fl", "fr", "bl", "br"]
DOF_ORDER = [f"{l}_{j}" for l in LEGS for j in ("SY_J", "SP_J", "knee")]
DOF_ORDER = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee"]
BODY_NAMES = ["origin", "fl_knee", "fr_knee", "bl_knee", "br_knee"]


# ---------------------------------------------------------------- URDF ----
def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])


def axis_rot(axis, q):
    """Rodrigues rotation about a unit axis."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.j = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0])
            lim = j.find("limit")
            self.j[j.get("name")] = dict(
                xyz=xyz, R=rpy_to_mat(*rpy), axis=axis,
                lo=float(lim.get("lower")), hi=float(lim.get("upper")),
                vel=float(lim.get("velocity")))

    def chain(self, leg):
        return [f"{leg}_SY_J", f"{leg}_SP_J", f"{leg}_knee"]

    def leg_fk(self, leg, q):
        """Return (tip_pos, knee_R, knee_pos) in the origin frame."""
        T_R, T_p = np.eye(3), np.zeros(3)
        for name, qi in zip(self.chain(leg), q):
            J = self.j[name]
            T_p = T_p + T_R @ J["xyz"]
            T_R = T_R @ J["R"] @ axis_rot(J["axis"], qi)
        tip = T_p + T_R @ np.array([0, 0, -SHANK_LEN])
        return tip, T_R, T_p

    def sy_pos(self, leg):
        return self.j[f"{leg}_SY_J"]["xyz"]

    def leg_reach(self, leg):
        """thigh + shank, metres."""
        return np.linalg.norm(self.j[f"{leg}_knee"]["xyz"]) + SHANK_LEN

    def limits(self, leg):
        return np.array([[self.j[n]["lo"], self.j[n]["hi"]] for n in self.chain(leg)])


# ------------------------------------------------------------------ IK ----
def solve_leg(urdf, leg, target, q0, lim, iters=60):
    """Damped least-squares IK for one 3-DOF leg. Returns (q, err_metres)."""
    q = q0.copy()
    lam = 1e-3
    for _ in range(iters):
        tip, _, _ = urdf.leg_fk(leg, q)
        r = target - tip
        e = np.linalg.norm(r)
        if e < 1e-6:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            dq = np.zeros(3); dq[k] = 1e-6
            tk, _, _ = urdf.leg_fk(leg, q + dq)
            J[:, k] = (tk - tip) / 1e-6
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
        except np.linalg.LinAlgError:
            break
        qn = np.clip(q + step, lim[:, 0], lim[:, 1])
        tn, _, _ = urdf.leg_fk(leg, qn)
        if np.linalg.norm(target - tn) < e:
            q, lam = qn, max(lam * 0.5, 1e-6)
        else:
            lam *= 4.0
            if lam > 1e4:
                break
    tip, _, _ = urdf.leg_fk(leg, q)
    return q, float(np.linalg.norm(target - tip))


# Both knee branches x both SP signs. A 3-DOF leg has two elbow-up/down
# solutions for most targets; warm-starting alone locks whichever the seed
# happened to land in, for the whole clip.
BRANCH_SEEDS = [np.array([0.0, -0.30,  0.60]), np.array([0.0,  0.30, -0.60]),
                np.array([0.0, -0.30, -0.60]), np.array([0.0,  0.30,  0.60])]


def solve_leg_best(urdf, leg, target, q_prev, lim, w_cont=0.004, allow_switch=True):
    """Solve, and if the warm start lands badly, retry from each branch seed.

    Scored on foot error + a penalty per joint sitting on a limit + a
    continuity term, so we only switch branch when it clearly pays.
    """
    def at_limit(q):
        return int(sum(abs(q[m]-lim[m,0]) < 1e-6 or abs(q[m]-lim[m,1]) < 1e-6 for m in range(3)))

    q, e = solve_leg(urdf, leg, target, q_prev, lim)
    if not allow_switch or (e < 2e-3 and at_limit(q) == 0):
        # Mid-clip branch switches are forbidden: they are a discontinuity in
        # joint space and show up as a huge spike in the finite-differenced
        # velocities, which is worse than a little clamping.
        return q, e, False
    best = (e + 0.02 * at_limit(q), q, e)
    for s in BRANCH_SEEDS:
        qc, ec = solve_leg(urdf, leg, target, s, lim)
        score = ec + 0.02 * at_limit(qc) + w_cont * float(np.linalg.norm(qc - q_prev))
        if score < best[0]:
            best = (score, qc, ec)
    return best[1], best[2], not np.allclose(best[1], q)


def solve_chain_rot(joints, R_target, q0, iters=80):
    """Fit a serial rotation chain (list of (axis, lo, hi)) to a target rotation."""
    q = q0.copy()
    lam = 1e-3

    def fk(qq):
        R = np.eye(3)
        for (ax, _, _), qi in zip(joints, qq):
            R = R @ axis_rot(ax, qi)
        return R

    def res(qq):
        E = fk(qq).T @ R_target
        return np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) * 0.5

    n = len(joints)
    lo = np.array([j[1] for j in joints]); hi = np.array([j[2] for j in joints])
    for _ in range(iters):
        r = res(q); e = np.linalg.norm(r)
        if e < 1e-8:
            break
        J = np.zeros((3, n))
        for k in range(n):
            dq = np.zeros(n); dq[k] = 1e-6
            J[:, k] = (res(q + dq) - r) / 1e-6
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(n), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        qn = np.clip(q + step, lo, hi)
        if np.linalg.norm(res(qn)) < e:
            q, lam = qn, max(lam * 0.5, 1e-6)
        else:
            lam *= 4.0
            if lam > 1e4:
                break
    return q, float(np.linalg.norm(res(q)))


# ------------------------------------------------------------- helpers ----
def kabsch(P, Q):
    """Rigid R,t minimising |R@P_i + t - Q_i|. P,Q are (N,3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def mat_to_quat(R):
    """wxyz."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


# Rig is -Y forward, +Z up, character-left +X.  Robot is +X forward, +Y left.
# Rotation by +90 deg about Z maps one onto the other.
AXIS = np.array([[0., -1., 0.],
                 [1.,  0., 0.],
                 [0.,  0., 1.]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stance-shift", default="auto",
                    help="'auto' (default) searches a shared body-height offset "
                         "that re-centres the clip in the leg's reachable band; "
                         "'off' reproduces the old behaviour; or a value in metres.")
    ap.add_argument("--start-t", type=float, default=None)
    ap.add_argument("--end-t", type=float, default=None)
    a = ap.parse_args()

    d = json.load(open(a.raw))
    urdf = Urdf(a.urdf)
    fr = d["frames"]
    if a.start_t is not None:
        fr = [f for f in fr if f["t"] >= a.start_t]
    if a.end_t is not None:
        fr = [f for f in fr if f["t"] <= a.end_t]
    T = len(fr)
    hz = float(d["sample_hz"])
    print(f"clip {d['source']}  {T} samples @ {hz} Hz  ({T/hz:.2f} s)")

    # --- static check: does the documented A.3 crouch give a symmetric stance? ---
    crouch = {"fl": [0, -0.30, 0.60], "fr": [0, 0.30, -0.60],
              "bl": [0, -0.30, 0.60], "br": [0, -0.30, -0.60]}
    tips0 = {l: urdf.leg_fk(l, np.array(crouch[l], float))[0] for l in LEGS}
    print("A.3 crouch stance check (tip xyz in origin frame):")
    for l in LEGS:
        print(f"   {l}: {np.round(tips0[l], 4)}")
    zs = np.array([tips0[l][2] for l in LEGS])
    print(f"   foot height spread {1000*(zs.max()-zs.min()):.1f} mm "
          f"(0 = sign map consistent);  stance height {-zs.mean():.4f} m (spec 0.19-0.20)")

    # --- scale: per-leg limb-length ratio ---------------------------------
    rest = d["rest_bone_lengths"]
    s_leg = {}
    for l in LEGS:
        rig_len = rest[l]["sp"] + rest[l]["kn"]
        s_leg[l] = urdf.leg_reach(l) / rig_len
    s_master = float(np.mean(list(s_leg.values())))
    print("scale per leg:", {k: round(v, 6) for k, v in s_leg.items()},
          f" master={s_master:.6f} m/unit")

    P = lambda f, l, k: AXIS @ np.array(f[l][k])          # axis-converted, raw units
    urdf_sy = np.array([urdf.sy_pos(l) for l in LEGS])

    # --- pass 1: rigid base fit per frame ---------------------------------
    R_b = np.zeros((T, 3, 3)); t_b = np.zeros((T, 3)); fit_res = np.zeros(T)
    for i, f in enumerate(fr):
        rig_sy = np.array([P(f, l, "sy") for l in LEGS]) * s_master
        R, t = kabsch(rig_sy, urdf_sy)          # maps rig -> urdf layout
        # we want origin pose in world: invert
        R_b[i] = R.T
        t_b[i] = -R.T @ t
        fit_res[i] = np.sqrt(np.mean(np.sum((rig_sy @ R.T + t - urdf_sy) ** 2, axis=1)))
    print(f"base fit residual: mean {fit_res.mean()*1000:.1f} mm  "
          f"max {fit_res.max()*1000:.1f} mm  (constant part = hip-layout mismatch, "
          f"variation = spine flex the robot cannot do)")
    print(f"  spine-flex component (std): {fit_res.std()*1000:.1f} mm")

    # --- pass 2: IK ---------------------------------------------------------
    dof = np.zeros((T, 12), np.float64)
    tips_w = np.zeros((T, 4, 3)); knee_R = np.zeros((T, 4, 3, 3)); knee_p = np.zeros((T, 4, 3))
    err = np.zeros((T, 4)); clamped = np.zeros(12, int); n_switch = np.zeros(4, int)
    # Nominal crouch seed, per leg, in URDF convention (spec A.3). Seeding all
    # four the same puts fr/br in the wrong IK branch and they never recover,
    # because each frame warm-starts from the previous one.
    lims = {l: urdf.limits(l) for l in LEGS}
    # Branch is chosen ONCE per leg, scored over frames sampled across the whole
    # clip -- frame 0 alone is not representative, and switching mid-clip is a
    # joint-space discontinuity that wrecks the finite-differenced velocities.
    def leg_target(f, l):
        v = (P(f, l, "tip") - P(f, l, "sy")) * s_leg[l]
        return urdf.sy_pos(l) + R_b[fr.index(f)].T @ v

    probe = [fr[j] for j in np.linspace(0, T - 1, min(24, T), dtype=int)]
    q_prev = {}
    for l in LEGS:
        best = None
        for sd in BRANCH_SEEDS:
            errs, clam = [], 0
            for f in probe:
                qc, ec = solve_leg(urdf, l, leg_target(f, l), sd, lims[l])
                errs.append(ec)
                clam += sum(abs(qc[m]-lims[l][m,0]) < 1e-6 or abs(qc[m]-lims[l][m,1]) < 1e-6
                            for m in range(3))
            score = float(np.mean(errs)) + 0.02 * clam / len(probe)
            if best is None or score < best[0]:
                best = (score, sd)
        q_prev[l] = best[1].copy()
        print(f"  branch {l}: seed {np.round(best[1],2)}  score {best[0]*1000:.2f} mm-equiv")

    seed = {l: q_prev[l].copy() for l in LEGS}

    # All foot targets up front, hip-relative (foot placed w.r.t. its OWN hip,
    # scaled by that leg's limb ratio, so the hip-width mismatch never enters).
    tgt = np.zeros((T, 4, 3))
    for i, f in enumerate(fr):
        for k, l in enumerate(LEGS):
            tgt[i, k] = urdf.sy_pos(l) + R_b[i].T @ ((P(f, l, "tip") - P(f, l, "sy")) * s_leg[l])

    # ---- stance shift: bring the clip INTO the reachable band ---------------
    # The amplitude ladder below shrinks foot motion about the clip's own mean.
    # That cannot help when the MEAN itself lies outside the leg's reachable
    # band (147-204 mm from the hip): at any amplitude, including zero, the pose
    # is still out of range, so the ladder bottoms out and clamps anyway. This
    # is the actual reason the crouch-heavy clips ended up at 50% amplitude and
    # still clamped 39-48% of frames.
    #
    # Shifting the whole body vertically re-centres the clip in the workspace
    # first, and costs only absolute stance height -- the motion AMPLITUDE, which
    # is what carries the performance, survives. One shift is shared by all four
    # legs because they share one body; per-leg shifts would distort it.
    # Measured recovery of authored amplitude: Timid 8%->35%, Enthusiastic
    # 18%->38%, Laidback 20%->37%, Cheeky 35%->50%, DeadPan/Eccentric unchanged.
    idx0 = np.linspace(0, T - 1, min(120, T), dtype=int)

    # Raising the body trades one failure for another: it clears the knee's
    # LOWER limit but drives the knee toward ZERO, which is the straight-leg
    # singularity. There the leg has almost no leverage on body height, IK is
    # ill-conditioned, and the solver pops -- the measured cause of the 20.85
    # rad/s velocity spikes in the conform-rig walk cycle. So the objective has
    # to charge for both, or the search happily straightens the legs out.
    KNEE_MIN = 0.15          # rad; below this the knee is effectively locked

    def stance_probe(dz, alpha=1.0, idx=None):
        """(clamped_frac, near-straight_frac, mean foot error) at body rise dz."""
        idx = idx0 if idx is None else idx
        clamp = sing = 0.0; errs = []
        for k, l in enumerate(LEGS):
            col = tgt[:, k].copy(); col[:, 2] -= dz
            ctr = col.mean(0); c = s = 0
            for i in idx:
                q, e = solve_leg(urdf, l, ctr + alpha * (col[i] - ctr), seed[l], lims[l])
                c += sum(abs(q[m]-lims[l][m, 0]) < 1e-6 or abs(q[m]-lims[l][m, 1]) < 1e-6
                         for m in range(3))
                s += abs(q[2]) < KNEE_MIN          # q[2] is the knee
                errs.append(e)
            clamp += c / (3.0 * len(idx)); sing += s / len(idx)
        return clamp / len(LEGS), sing / len(LEGS), float(np.mean(errs))

    # The objective is HOW MUCH OF THE AUTHORED MOTION SURVIVES, so shift and
    # amplitude must be chosen together: the pipeline can already trade amplitude
    # for feasibility, and judging a shift at full amplitude alone makes it shift
    # clips that amplitude reduction would have handled on its own.
    #
    # For each candidate shift, find the largest amplitude that clears the
    # project's existing gates, then keep the shift that preserves the most
    # amplitude. Ties go to the smaller shift, then to better foot tracking --
    # so "no shift" wins whenever it is not actually beaten. No invented weights.
    # Constrained search, NOT a weighted sum -- weights here would be free
    # parameters tuned by eye, the exact failure the spec calls out in A.8.
    # Feasibility is a hard gate (the project's own 0.5% limit gate, plus a cap
    # on near-straight knees); among the shifts that pass, take the one that best
    # reproduces the AUTHORED foot placement. "No shift" therefore wins whenever
    # it is not actually beaten.
    #
    # Feasibility is judged at FULL amplitude. Note this does not simply
    # duplicate the amplitude ladder below: shrinking amplitude pulls poses
    # toward the clip mean, so when the mean itself sits near the straight-leg
    # singularity, reducing amplitude makes that WORSE, not better. Only a shift
    # moves the mean. A joint search over (shift, amplitude) was tried and is too
    # slow in pure Python -- every probe is a full IK solve.
    CLAMP_GATE, SING_GATE = 0.005, 0.05
    if a.stance_shift == "off":
        dz_best = 0.0
    elif a.stance_shift == "auto":
        rows = [(dz,) + stance_probe(dz) for dz in np.arange(-0.060, 0.0605, 0.005)]
        feas = [r for r in rows if r[1] <= CLAMP_GATE and r[2] <= SING_GATE]
        if feas:
            dz_best = min(feas, key=lambda r: (r[3], abs(r[0])))[0]
        else:
            dz_best = min(rows, key=lambda r: (r[1] + r[2], abs(r[0])))[0]
            print("  note: no stance shift makes this clip fully feasible; "
                  "taking the least-infeasible one")
        if abs(abs(dz_best) - 0.060) < 1e-9:
            print("  WARNING stance-shift search hit its boundary")
    else:
        dz_best = float(a.stance_shift)

    c0, s0, e0 = stance_probe(0.0)
    if abs(dz_best) > 1e-9:
        tgt[:, :, 2] -= dz_best
        c1, s1, e1 = stance_probe(0.0)
        print(f"stance shift: body raised {dz_best*1000:+.1f} mm | at-limit "
              f"{c0*100:.1f}%->{c1*100:.1f}% | near-straight knee "
              f"{s0*100:.1f}%->{s1*100:.1f}% | foot err "
              f"{e0*1000:.1f}->{e1*1000:.1f} mm")
    else:
        print(f"stance shift: none | at-limit {c0*100:.1f}% | near-straight knee "
              f"{s0*100:.1f}% | foot err {e0*1000:.1f} mm")

    def clamp_frac(l, k, alpha, idx):
        """Fraction of joint-frames on a limit for leg l at amplitude alpha."""
        c = 0
        ctr = tgt[:, k].mean(0)
        for i in idx:
            q, _ = solve_leg(urdf, l, ctr + alpha * (tgt[i, k] - ctr), seed[l], lims[l])
            c += sum(abs(q[m]-lims[l][m,0]) < 1e-6 or abs(q[m]-lims[l][m,1]) < 1e-6 for m in range(3))
        return c / (3.0 * len(idx))

    # Amplitude reduction: shrink each leg's foot motion about its own mean until
    # clamping clears the <0.5% gate. Reducing toward the clip mean (not the
    # doc crouch) keeps the stance the animator authored and only shrinks travel.
    idx = np.linspace(0, T - 1, min(240, T), dtype=int)
    alpha = {}
    for k, l in enumerate(LEGS):
        a_ok = 1.0
        for cand in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
            if clamp_frac(l, k, cand, idx) < 0.005:
                a_ok = cand
                break
            a_ok = cand
        alpha[l] = a_ok
    # Force L/R pairs to share an amplitude. Scaling legs independently clears
    # clamping but manufactures a left/right asymmetry -- the exact
    # leg-sacrifice failure mode the spec warns about (A.9 / B.4.1).
    for a_, b_ in (("fl", "fr"), ("bl", "br")):
        m = min(alpha[a_], alpha[b_]); alpha[a_] = alpha[b_] = m
    print("amplitude scale per leg (L/R paired):", alpha)

    for i in range(T):
        for k, l in enumerate(LEGS):
            ctr = tgt[:, k].mean(0)
            target = ctr + alpha[l] * (tgt[i, k] - ctr)
            tgt[i, k] = target
            # Solve from the fixed branch seed, not the previous frame: warm
            # starting from t-1 lets the solution drift into a clamped corner
            # and it cannot climb back out. Within a fixed branch the solve is
            # continuous in the target, so continuity is preserved anyway.
            q, e = solve_leg(urdf, l, target, seed[l], lims[l])
            dof[i, 3*k:3*k+3] = q
            err[i, k] = e

    # --- rate limit: minimal smoothing that brings |q̇| under the joint limit --
    def gsmooth(x, sig):
        if sig <= 0:
            return x
        r = int(np.ceil(3 * sig))
        w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); w /= w.sum()
        pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
        return np.stack([np.convolve(pad[:, c], w, "valid") for c in range(x.shape[1])], 1)

    dt = 1.0 / hz
    sigma = 0.0
    for cand in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]:
        if np.abs(np.gradient(gsmooth(dof, cand), dt, axis=0)).max() <= 10.0:
            sigma = cand
            break
        sigma = cand
    dof = gsmooth(dof, sigma)
    print(f"rate-limit smoothing: sigma {sigma} samples ({sigma/hz*1000:.1f} ms)")

    # --- recompute all body quantities FROM the final joint angles ------------
    for i in range(T):
        for k, l in enumerate(LEGS):
            q = dof[i, 3*k:3*k+3]
            tip, KR, KP = urdf.leg_fk(l, q)
            err[i, k] = np.linalg.norm(tgt[i, k] - tip)
            tips_w[i, k] = t_b[i] + R_b[i] @ tip
            knee_R[i, k] = R_b[i] @ KR
            knee_p[i, k] = t_b[i] + R_b[i] @ KP
            for m in range(3):
                if abs(q[m] - lims[l][m, 0]) < 1e-6 or abs(q[m] - lims[l][m, 1]) < 1e-6:
                    clamped[3*k+m] += 1

    # --- Stage 4: contact-consistent root ------------------------------------
    # Leg targets are hip-relative, so a planted foot only holds still in the
    # world if the ROOT moves to make it so. Solving that here is what stops the
    # reference asking for a foot that is planted and sliding at the same time.
    ct = np.array([[f["contact"][l] for l in LEGS] for f in fr], bool)

    def slip_mm():
        s = []
        for i in range(1, T):
            p = ct[i] & ct[i - 1]
            if p.any():
                s.append(np.linalg.norm(tips_w[i, p, :2] - tips_w[i - 1, p, :2], axis=1).max())
        return 1000 * float(np.sum(s)) / max(ct.sum(0).mean(), 1) if s else 0.0

    travel0, slip0 = float(np.linalg.norm(t_b[-1, :2] - t_b[0, :2])), slip_mm()

    # vertical: planted feet sit on z = 0
    dz = np.full(T, np.nan)
    for i in range(T):
        if ct[i].any():
            dz[i] = -tips_w[i, ct[i], 2].mean()
    ok = ~np.isnan(dz)
    dz = np.interp(np.arange(T), np.arange(T)[ok], dz[ok])       # flight phases
    dz = gsmooth(dz[:, None], 6)[:, 0]
    t_b[:, 2] += dz; tips_w[:, :, 2] += dz[:, None]; knee_p[:, :, 2] += dz[:, None]

    # horizontal: cancel world motion of feet that stay planted between frames
    dxy = np.zeros((T, 2))
    for i in range(1, T):
        p = ct[i] & ct[i - 1]
        if p.any():
            dxy[i] = -(tips_w[i, p, :2] - tips_w[i - 1, p, :2]).mean(0)
    off = gsmooth(np.cumsum(dxy, axis=0), 6)
    t_b[:, :2] += off; tips_w[:, :, :2] += off[:, None, :]; knee_p[:, :, :2] += off[:, None, :]

    # The "foot" is the shank TIP -- a point -- but the paw mesh hangs ~28-30 mm
    # below it (measured from *_knee.STL). Resting the tip on z=0 buries the paw.
    # Also use a robust percentile of PLANTED feet as the floor: the raw minimum
    # lets one outlier frame on one leg set the ground for the whole clip.
    # Shift so the LOWEST paw surface over the whole clip just touches z=0. A
    # percentile floor reads nicer on paper but lets outlier frames punch through
    # the ground, which is worse than a few frames of float.
    PAW_DROP = 0.0288
    zshift = tips_w[:, :, 2].min() - PAW_DROP
    t_b[:, 2] -= zshift; tips_w[:, :, 2] -= zshift; knee_p[:, :, 2] -= zshift
    print(f"contact-consistent root: foot slip {slip0:.1f} -> {slip_mm():.1f} mm/stance "
          f"(gate <5) | root travel {travel0:.3f} -> {np.linalg.norm(t_b[-1,:2]-t_b[0,:2]):.3f} m")
    print(f"ground shift {zshift*1000:+.1f} mm  -> min foot z {tips_w[:,:,2].min()*1000:.1f} mm")
    print(f"IK foot error: mean {err.mean()*1000:.2f} mm  p95 {np.percentile(err,95)*1000:.2f} mm  "
          f"max {err.max()*1000:.2f} mm")
    pc = 100.0 * clamped / T
    bad = {DOF_ORDER[i]: f"{pc[i]:.1f}%" for i in range(12) if pc[i] > 0.5}
    print(f"frames at a joint limit: {'none >0.5%' if not bad else bad}")

    # --- sign-map check (A.3) ----------------------------------------------
    sgn = {n: np.sign(np.mean(dof[:, i])) for i, n in enumerate(DOF_ORDER)}
    print("mean-sign check SP:", {k: int(v) for k, v in sgn.items() if "SP" in k},
          " knee:", {k: int(v) for k, v in sgn.items() if "knee" in k})

    # --- head / tail --------------------------------------------------------
    head_j = [(urdf.j["head_pitch_joint"]["axis"], urdf.j["head_pitch_joint"]["lo"], urdf.j["head_pitch_joint"]["hi"]),
              (urdf.j["head_yaw"]["axis"], urdf.j["head_yaw"]["lo"], urdf.j["head_yaw"]["hi"]),
              (urdf.j["head_roll"]["axis"], urdf.j["head_roll"]["lo"], urdf.j["head_roll"]["hi"])]
    tail_j = [(urdf.j["tail_pitch"]["axis"], urdf.j["tail_pitch"]["lo"], urdf.j["tail_pitch"]["hi"]),
              (urdf.j["tail_yaw"]["axis"], urdf.j["tail_yaw"]["lo"], urdf.j["tail_yaw"]["hi"])]
    ht = np.zeros((T, 5)); ht_res = np.zeros((T, 2))
    qh = np.zeros(3); qt = np.zeros(2)
    R_head0 = R_tail0 = None
    for i, f in enumerate(fr):
        Rh = AXIS @ quat_to_mat(np.array(f["head"]["q"])) @ AXIS.T
        Rt = AXIS @ quat_to_mat(np.array(f["tail1"]["q"])) @ AXIS.T
        if R_head0 is None:
            R_head0, R_tail0 = Rh.copy(), Rt.copy()
        qh, eh = solve_chain_rot(head_j, R_b[i].T @ Rh @ R_head0.T, qh)
        qt, et = solve_chain_rot(tail_j, R_b[i].T @ Rt @ R_tail0.T, qt)
        ht[i, :3], ht[i, 3:] = qh, qt
        ht_res[i] = (eh, et)
    print(f"head/tail decompose residual: head {ht_res[:,0].mean():.4f} rad  "
          f"tail {ht_res[:,1].mean():.4f} rad  (unrepresentable rotation)")

    # --- velocities ---------------------------------------------------------
    dof_vel = np.gradient(dof, dt, axis=0)
    body_pos = np.concatenate([t_b[:, None, :], tips_w], axis=1)          # (T,5,3)
    body_rot = np.zeros((T, 5, 4))
    for i in range(T):
        body_rot[i, 0] = mat_to_quat(R_b[i])
        for k in range(4):
            body_rot[i, k+1] = mat_to_quat(knee_R[i, k])
    body_lin = np.gradient(body_pos, dt, axis=0)
    body_ang = np.zeros((T, 5, 3))
    for b in range(5):
        for i in range(1, T-1):
            Ra = quat_to_mat(body_rot[i-1, b]); Rc = quat_to_mat(body_rot[i+1, b])
            E = Ra.T @ Rc
            body_ang[i, b] = np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]]) * 0.5 / dt
        body_ang[0, b] = body_ang[1, b]; body_ang[-1, b] = body_ang[-2, b]

    vmax = np.abs(dof_vel).max()
    print(f"max |dof velocity| {vmax:.2f} rad/s  (limit 10)  "
          f"frames over: {(np.abs(dof_vel).max(1) > 10).sum()}")

    contacts = np.array([[f["contact"][l] for l in LEGS] for f in fr], bool)
    print(f"duty factor: {dict(zip(LEGS, contacts.mean(0).round(2)))}")

    np.savez(a.out,
             fps=np.array(hz),
             dof_names=np.array(DOF_ORDER),
             body_names=np.array(BODY_NAMES),
             dof_positions=dof.astype(np.float32),
             dof_velocities=dof_vel.astype(np.float32),
             body_positions=body_pos.astype(np.float32),
             body_rotations=body_rot.astype(np.float32),
             body_linear_velocities=body_lin.astype(np.float32),
             body_angular_velocities=body_ang.astype(np.float32),
             # extensions (spec B.3 S6 / B.5.6)
             contacts=contacts,
             phase=np.linspace(0, 1, T, dtype=np.float32),
             head_tail_positions=ht.astype(np.float32),
             source=np.array(d["source"]))
    print(f"wrote {a.out}")


main()
