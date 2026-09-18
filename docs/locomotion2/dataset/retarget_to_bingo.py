#!/usr/bin/env python3
"""Retarget a Locomotion 2 dog-walk NPZ (extract_walk.py output) to Bingo v4.

Reuses the IK/scaling/contact-lock machinery from `scripts/retarget.py`
(the repo's existing Stage 2 spatial-retargeting tool) rather than
reimplementing it: `Urdf`, `solve_leg`, `solve_leg_best`, `kabsch`,
`mat_to_quat`/`quat_to_mat`, `BRANCH_SEEDS`, `SHANK_LEN`, `LEGS`,
`DOF_ORDER`, `BODY_NAMES` are copied verbatim below (that file executes a
CLI `main()` unconditionally at import time -- it was not written to be
imported as a library -- so the functions are copied with attribution
instead of imported live; nothing about their logic is changed).

What's different from `scripts/retarget.py` (which retargets a Blender
rig's Cartesian export): the source here is a SMAL-fit dog skeleton
(`extract_walk.py`'s output NPZ), already heading-aligned to +X, ground at
z=0, real-world metres. So:
  - no AXIS conversion (the Blender rig is -Y-forward; our source is
    already +X-forward);
  - per-leg scale comes from the dog's *own measured* mean hip-to-paw
    distance (no `rest_bone_lengths` metadata to read), matched against
    `urdf.leg_reach(leg)`;
  - the per-frame rigid base fit (Kabsch) uses the dog's four HIP
    ("leg_top") keypoints against Bingo's fixed SY-joint layout, exactly
    the same way `scripts/retarget.py` fits the Blender rig's SY bones --
    this is what carries the dog's root/body motion onto Bingo's rigid
    body every frame, translation and rotation together, without treating
    "root motion" and "leg articulation" as separate problems;
  - contact-consistent root correction uses the *source's own*
    `foot_contact` array (already computed by `extract_walk.py`'s
    Otsu+hysteresis detector) instead of re-deriving contacts from foot
    height, so Locomotion 2's contact timing is the single source of
    truth throughout the pipeline, not re-decided at each stage.

Output schema matches `scripts/retarget.py`'s exactly (fps, dof_names,
body_names, dof_positions, dof_velocities, body_positions, body_rotations,
body_linear_velocities, body_angular_velocities, contacts, source) --
directly loadable by IsaacLab's `MotionLoader` and hence by the existing,
already-working `scripts/replay_motion.py` (kinematic replay) and
`scripts/playback_physics.py` (physics feasibility), unmodified.

Usage:
    /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python retarget_to_bingo.py \\
        --source ../motions/source/dog02_walk_02.npz \\
        --urdf "/pub0/muhammadf/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \\
        --out ../motions/retarget/bingo_dog02_walk_02.npz
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# =====================================================================
# Copied verbatim from scripts/retarget.py (not imported: that module
# calls main() unconditionally at the bottom, so importing it would try
# to run its own CLI). Source: /pub0/muhammadf/Blender/scripts/retarget.py
# =====================================================================
SHANK_LEN = 0.120
LEGS = ["fl", "fr", "bl", "br"]
DOF_ORDER = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee"]
BODY_NAMES = ["origin", "fl_knee", "fr_knee", "bl_knee", "br_knee"]
PAW_DROP = 0.0288


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])


def axis_rot(axis, q):
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
        return np.linalg.norm(self.j[f"{leg}_knee"]["xyz"]) + SHANK_LEN

    def limits(self, leg):
        return np.array([[self.j[n]["lo"], self.j[n]["hi"]] for n in self.chain(leg)])


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


BRANCH_SEEDS = [np.array([0.0, -0.30,  0.60]), np.array([0.0,  0.30, -0.60]),
                np.array([0.0, -0.30, -0.60]), np.array([0.0,  0.30,  0.60])]


def kabsch(P, Q):
    """Rigid R,t minimising |R@P_i + t - Q_i|. P,Q are (N,3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def mat_to_quat(R):
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


def gsmooth(x, sig):
    if sig <= 0:
        return x
    r = int(np.ceil(3 * sig))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); w /= w.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], w, "valid") for c in range(x.shape[1])], 1)


# =====================================================================
# New: adapter from extract_walk.py's dog-NPZ schema to this pipeline.
# =====================================================================
# Source leg_order = [left_front, left_rear, right_front, right_rear];
# Bingo LEGS = [fl, fr, bl, br]. Both conventions are body-relative
# left/right with the animal/robot's own forward = +X, so the mapping is
# direct anatomical left<->left, front<->front -- no mirroring needed
# (unlike the Blender rig case in scripts/retarget.py, which mirrors .L/.R).
SRC_LEG_TO_BINGO = {"left_front": "fl", "right_front": "fr", "left_rear": "bl", "right_rear": "br"}


def load_source(path: Path):
    d = dict(np.load(path, allow_pickle=True))
    leg_order = [str(x) for x in d["leg_order"]]
    bingo_to_src = {v: k for k, v in SRC_LEG_TO_BINGO.items()}
    bingo_idx = [leg_order.index(bingo_to_src[leg]) for leg in LEGS]
    return d, bingo_idx


def solve_leg_best(urdf, leg, target, q_prev, lim, w_cont=0.004):
    """Copied from scripts/retarget.py: try the warm start, and if it lands
    badly, retry from each branch seed, scored on error + limit + continuity."""
    def at_limit(q):
        return int(sum(abs(q[m]-lim[m,0]) < 1e-6 or abs(q[m]-lim[m,1]) < 1e-6 for m in range(3)))
    q, e = solve_leg(urdf, leg, target, q_prev, lim)
    if e < 2e-3 and at_limit(q) == 0:
        return q, e
    best = (e + 0.02 * at_limit(q), q, e)
    for s in BRANCH_SEEDS:
        qc, ec = solve_leg(urdf, leg, target, s, lim)
        score = ec + 0.02 * at_limit(qc) + w_cont * float(np.linalg.norm(qc - q_prev))
        if score < best[0]:
            best = (score, qc, ec)
    return best[1], best[2]


def yaw_only_matrix(R):
    """Rotation-about-Z-only matrix built from R's local-X axis heading,
    discarding roll/pitch. R: (...,3,3)."""
    x = R[..., :, 0]
    yaw = np.arctan2(x[..., 1], x[..., 0])
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros(R.shape)
    out[..., 0, 0] = c; out[..., 0, 1] = -s
    out[..., 1, 0] = s; out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out, yaw


def compute_root_pose(root_method: str, d: dict, s_body: float, yaw_smooth_sigma: float = 0.0):
    """Three ways to get Bingo's per-frame origin pose (R_b, t_b) from the
    source clip -- see module docstring / CLI help for why (A) was replaced.

    A. 'kabsch': per-frame rigid fit of the dog's 4 hip keypoints to Bingo's
       fixed hip layout. Lets hip/scapula articulation noise leak into what
       is supposed to be a single global root orientation -- this produced
       5-16 degree frame-to-frame rotation swings not explained by real
       heading change, and pinned fr_SY_J/br_SY_J at their limits for the
       majority of frames.
    B. 'smal': use the SMAL fit's own root_pos/root_orient_rotmat directly
       (already heading-aligned to +X and ground-zeroed by extract_walk.py's
       normalize_motion -- the "single static normalization transform").
       This is the fit's own estimate of body translation+orientation, not
       re-derived from 4 noisy leg-attachment points.
    C. 'yaw_only': like B, but the orientation is reduced to a pure
       rotation-about-Z (heading) built from root_orient's local-X axis,
       discarding whatever roll/pitch the SMAL fit estimated -- in case that
       roll/pitch is itself fit noise rather than real dog motion. Optional
       temporal smoothing on the yaw angle.
    """
    root_pos = d["root_pos"]
    root_orient = d["root_orient_rotmat"]
    T = root_pos.shape[0]
    t_b = root_pos * s_body
    if root_method == "smal":
        R_b = root_orient.copy()
    elif root_method == "yaw_only":
        R_b, yaw = yaw_only_matrix(root_orient)
        if yaw_smooth_sigma > 0:
            yaw_s = gsmooth(yaw[:, None], yaw_smooth_sigma)[:, 0]
            c, s = np.cos(yaw_s), np.sin(yaw_s)
            R_b = np.zeros((T, 3, 3))
            R_b[:, 0, 0] = c; R_b[:, 0, 1] = -s
            R_b[:, 1, 0] = s; R_b[:, 1, 1] = c
            R_b[:, 2, 2] = 1.0
    else:
        raise ValueError(root_method)
    return R_b, t_b


def retarget(source_npz: Path, urdf_path: Path, out_npz: Path, report_path: Path,
             root_method: str = "smal", yaw_smooth_sigma: float = 0.0):
    d, bingo_idx = load_source(source_npz)
    urdf = Urdf(str(urdf_path))
    T = d["timestamps"].shape[0]
    hz = float(d["effective_fps_mean"])
    hip_pos = d["hip_pos"][:, bingo_idx, :]     # (T,4,3) reordered to LEGS
    paw_pos = d["paw_pos"][:, bingo_idx, :]
    contacts_src = d["foot_contact"][:, bingo_idx]  # (T,4) bool, LEGS order

    # --- per-leg scale: dog's own measured hip-to-paw length vs Bingo's reach.
    # Scaling by the MEAN extension (first attempt) puts the dog's PEAK
    # extension (7-19% above its own mean, in this clip) at 107-119% of
    # Bingo's max leg reach -- literally unreachable at the moments of
    # fullest stride, which is exactly when accurate foot placement matters
    # most. Scale by PEAK extension instead, mapped to RESERVE_FRAC of
    # Bingo's max reach, so even the most-extended frame leaves real IK
    # margin instead of sitting at/over the kinematic limit.
    RESERVE_FRAC = 0.90
    dog_leg_len = {l: float(np.linalg.norm(paw_pos[:, k] - hip_pos[:, k], axis=1).mean()) for k, l in enumerate(LEGS)}
    dog_leg_peak = {l: float(np.linalg.norm(paw_pos[:, k] - hip_pos[:, k], axis=1).max()) for k, l in enumerate(LEGS)}
    s_leg = {l: RESERVE_FRAC * urdf.leg_reach(l) / dog_leg_peak[l] for l in LEGS}
    print(f"dog leg lengths (m): mean={dog_leg_len} peak={dog_leg_peak}")
    print(f"per-leg scale to Bingo (peak-based, {RESERVE_FRAC:.0%} reserve): {s_leg}")

    urdf_sy = np.array([urdf.sy_pos(l) for l in LEGS])

    # --- body scale: a SEPARATE scale from the per-leg reach scale above.
    # First attempt reused scripts/retarget.py's exact approach verbatim,
    # which scales the hip constellation by the mean *leg-length* ratio
    # (s_leg) -- that is correct for the Blender rig it was built for
    # (authored at roughly Bingo's own body proportions already), but wrong
    # here: this dog's hip-to-hip span is ~0.33m front-to-back vs Bingo's
    # ~0.12m -- almost 3x -- while its leg length happens to be close to
    # Bingo's leg reach (~0.2m each), so leg-length ratio (~1.1x) is the
    # wrong scale for the BODY and produced a 135mm mean base-fit residual
    # (bigger than Bingo's entire ~78mm hip span!). Fixed with a proper
    # Umeyama similarity-transform scale fit to the mean hip constellation
    # (residual then drops to ~5mm) -- computed once, not per-frame, since
    # a real rigid robot has one fixed body scale.
    mean_hip = hip_pos.mean(axis=0)
    Pc, Qc = mean_hip.mean(0), urdf_sy.mean(0)
    Pd, Qd = mean_hip - Pc, urdf_sy - Qc
    H = Pd.T @ Qd
    U, S, Vt = np.linalg.svd(H)
    d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
    var_p = (Pd ** 2).sum() / len(mean_hip)
    s_body = float(np.sum(S * np.array([1, 1, d_sign])) / (len(mean_hip) * var_p))
    print(f"body scale (Umeyama, hip constellation -> Bingo hip layout): {s_body:.4f}")

    # --- pass 1: Bingo's per-frame origin pose (root_method: see compute_root_pose)
    if root_method == "kabsch":
        R_b = np.zeros((T, 3, 3)); t_b = np.zeros((T, 3)); fit_res = np.zeros(T)
        for i in range(T):
            dog_hips_scaled = hip_pos[i] * s_body
            R, t = kabsch(dog_hips_scaled, urdf_sy)
            R_b[i] = R.T
            t_b[i] = -R.T @ t
            fit_res[i] = np.sqrt(np.mean(np.sum((dog_hips_scaled @ R.T + t - urdf_sy) ** 2, axis=1)))
        print(f"[kabsch] base fit residual (hip-layout mismatch + dog spine flex Bingo can't replicate): "
              f"mean {fit_res.mean()*1000:.1f} mm max {fit_res.max()*1000:.1f} mm")
    else:
        R_b, t_b = compute_root_pose(root_method, d, s_body, yaw_smooth_sigma)
        fit_res = np.zeros(T)  # not fitted -- root pose comes directly from the source
        print(f"[{root_method}] root pose taken directly from source root_pos/root_orient_rotmat (no per-frame hip fit)")

    # --- leg targets, hip-relative, per-leg scale
    tgt = np.zeros((T, 4, 3))
    for i in range(T):
        for k, l in enumerate(LEGS):
            tgt[i, k] = urdf.sy_pos(l) + R_b[i].T @ ((paw_pos[i, k] - hip_pos[i, k]) * s_leg[l])

    lims = {l: urdf.limits(l) for l in LEGS}

    # Bingo's own documented natural standing/crouch pose (spec A.3, also
    # reproduced verbatim in scripts/retarget.py's own sanity check) -- these
    # ARE exactly BRANCH_SEEDS[0] (fl/bl) and BRANCH_SEEDS[1]/[2] (fr/br), so
    # "prefer the seed's own branch" is the same thing as "prefer the
    # anatomically natural knee/hip-pitch direction," not an arbitrary bias.
    CROUCH = {"fl": np.array([0., -0.30, 0.60]), "fr": np.array([0., 0.30, -0.60]),
              "bl": np.array([0., -0.30, 0.60]), "br": np.array([0., -0.30, -0.60])}

    # --- branch selection once per leg, scored across the whole clip (avoids
    # a mid-clip elbow-up/down flip, which is a joint-space discontinuity).
    # A first version scored purely on foot error + limit-avoidance, with no
    # preference for which of the 4 elbow-up/down x sign branches is used --
    # it let `fl` land in a branch with the OPPOSITE sign of SP/knee from
    # Bingo's own documented crouch (spec: fl SP should be ~-0.25, this
    # branch gave +0.37 to +0.65), which is kinematically valid (low foot
    # error) but anatomically inverted -- rendered in Isaac as the leg
    # tucking up under the body instead of extending toward the ground,
    # which is also *why* fl_SY_J was saturating (that inverted branch has
    # much less real yaw authority left). Fixed with a mild bias toward
    # Bingo's own documented natural-pose branch for each leg.
    NATURAL_BIAS = 0.30  # weight on distance-from-crouch-seed, in the same
                          # units as foot error (metres); mild enough that a
                          # genuinely infeasible natural branch still loses.
    seed = {}
    for l in LEGS:
        best = None
        for sd in BRANCH_SEEDS:
            errs, clam = [], 0
            for i in range(T):
                qc, ec = solve_leg(urdf, l, tgt[i, LEGS.index(l)], sd, lims[l])
                errs.append(ec)
                clam += sum(abs(qc[m]-lims[l][m,0]) < 1e-6 or abs(qc[m]-lims[l][m,1]) < 1e-6 for m in range(3))
            natural_dist = float(np.linalg.norm(sd - CROUCH[l]))
            score = float(np.mean(errs)) + 0.02 * clam / T + NATURAL_BIAS * natural_dist
            if best is None or score < best[0]:
                best = (score, sd)
        seed[l] = best[1].copy()
        print(f"  branch {l}: seed {np.round(best[1],2)} score {best[0]*1000:.2f} mm-equiv")

    # --- stance-shift search: raise/lower the body to clear joint limits &
    # avoid the near-straight-knee singularity, same principle+gates as
    # scripts/retarget.py (0.5% clamp gate, 5% near-straight-knee gate).
    KNEE_MIN = 0.15
    CLAMP_GATE, SING_GATE = 0.005, 0.05

    def stance_probe(dz):
        clamp = sing = 0.0; errs = []
        for k, l in enumerate(LEGS):
            col = tgt[:, k].copy(); col[:, 2] -= dz
            for i in range(T):
                q, e = solve_leg(urdf, l, col[i], seed[l], lims[l])
                clamp += sum(abs(q[m]-lims[l][m,0]) < 1e-6 or abs(q[m]-lims[l][m,1]) < 1e-6 for m in range(3)) / 3.0
                sing += float(abs(q[2]) < KNEE_MIN)
                errs.append(e)
        n = T * len(LEGS)
        return clamp / n, sing / n, float(np.mean(errs))

    rows = [(dz,) + stance_probe(dz) for dz in np.arange(-0.04, 0.0405, 0.005)]
    feas = [r for r in rows if r[1] <= CLAMP_GATE and r[2] <= SING_GATE]
    if feas:
        dz_best = min(feas, key=lambda r: (r[3], abs(r[0])))[0]
    else:
        dz_best = min(rows, key=lambda r: (r[1] + r[2], abs(r[0])))[0]
        print("  note: no stance shift makes this clip fully feasible; taking the least-infeasible one")
    c0, s0, e0 = stance_probe(0.0)
    if abs(dz_best) > 1e-9:
        tgt[:, :, 2] -= dz_best
        c1, s1, e1 = stance_probe(0.0)
        print(f"stance shift: body raised {dz_best*1000:+.1f} mm | at-limit {c0*100:.1f}%->{c1*100:.1f}% "
              f"| near-straight knee {s0*100:.1f}%->{s1*100:.1f}% | foot err {e0*1000:.1f}->{e1*1000:.1f} mm")
    else:
        print(f"stance shift: none | at-limit {c0*100:.1f}% | near-straight knee {s0*100:.1f}% | foot err {e0*1000:.1f} mm")

    # --- amplitude reduction if still clamping after the shift (L/R paired,
    # per scripts/retarget.py: independent per-leg scaling manufactures a
    # left/right gait asymmetry that isn't in the source).
    def clamp_frac(l, k, alpha):
        c = 0; ctr = tgt[:, k].mean(0)
        for i in range(T):
            q, _ = solve_leg(urdf, l, ctr + alpha * (tgt[i, k] - ctr), seed[l], lims[l])
            c += sum(abs(q[m]-lims[l][m,0]) < 1e-6 or abs(q[m]-lims[l][m,1]) < 1e-6 for m in range(3))
        return c / (3.0 * T)

    alpha = {}
    for k, l in enumerate(LEGS):
        a_ok = 1.0
        # Floored at 0.5, matching scripts/retarget.py's own deliberate floor:
        # tested extending this down to 0.2 for this clip -- it does clear
        # fl_SY_J's saturation further (38%->27%) with excellent foot error
        # (0.12mm), but an 80% swing-amplitude cut on the front legs visibly
        # flattens the gait, which fails "preserve overall visual character"
        # even if the joint-limit number improves. Not worth that trade.
        for cand in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
            a_ok = cand
            if clamp_frac(l, k, cand) < 0.005:
                break
        alpha[l] = a_ok
    for a_, b_ in (("fl", "fr"), ("bl", "br")):
        m = min(alpha[a_], alpha[b_]); alpha[a_] = alpha[b_] = m
    print("amplitude scale per leg (L/R paired):", alpha)
    if any(v < 1.0 for v in alpha.values()):
        for k, l in enumerate(LEGS):
            ctr = tgt[:, k].mean(0)
            tgt[:, k] = ctr + alpha[l] * (tgt[:, k] - ctr)

    # --- final IK solve, fixed branch seed per leg (not warm-started frame to
    # frame -- continuity within a fixed branch is preserved by the target's
    # own continuity, and warm-starting risks drifting into a clamped corner)
    dof = np.zeros((T, 12)); err = np.zeros((T, 4)); clamped = np.zeros(12, int)
    for i in range(T):
        for k, l in enumerate(LEGS):
            q, e = solve_leg(urdf, l, tgt[i, k], seed[l], lims[l])
            dof[i, 3*k:3*k+3] = q
            err[i, k] = e

    dt = 1.0 / hz
    sigma = 0.0
    for cand in [0, 1, 2, 3, 4, 6, 8]:
        if np.abs(np.gradient(gsmooth(dof, cand), dt, axis=0)).max() <= 10.0:
            sigma = cand
            break
        sigma = cand
    dof = gsmooth(dof, sigma)
    print(f"rate-limit smoothing: sigma {sigma} samples ({sigma/hz*1000:.1f} ms)")

    # --- recompute body quantities from final joint angles
    tips_w = np.zeros((T, 4, 3)); knee_R = np.zeros((T, 4, 3, 3)); knee_p = np.zeros((T, 4, 3))
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

    # --- contact-consistent root: use the SOURCE's own foot_contact (from
    # extract_walk.py's Otsu+hysteresis detector), not a re-derived one, so
    # contact timing has a single source of truth through the pipeline.
    ct = contacts_src

    def slip_mm(tw):
        s = []
        for i in range(1, T):
            p = ct[i] & ct[i - 1]
            if p.any():
                s.append(np.linalg.norm(tw[i, p, :2] - tw[i - 1, p, :2], axis=1).max())
        return 1000 * float(np.sum(s)) / max(ct.sum(0).mean(), 1) if s else 0.0

    travel0, slip0 = float(np.linalg.norm(t_b[-1, :2] - t_b[0, :2])), slip_mm(tips_w)

    dz_arr = np.full(T, np.nan)
    for i in range(T):
        if ct[i].any():
            dz_arr[i] = -tips_w[i, ct[i], 2].mean()
    ok = ~np.isnan(dz_arr)
    if ok.any():
        dz_arr = np.interp(np.arange(T), np.arange(T)[ok], dz_arr[ok])
        dz_arr = gsmooth(dz_arr[:, None], min(6, T // 4 or 1))[:, 0]
        t_b[:, 2] += dz_arr; tips_w[:, :, 2] += dz_arr[:, None]; knee_p[:, :, 2] += dz_arr[:, None]

    dxy = np.zeros((T, 2))
    for i in range(1, T):
        p = ct[i] & ct[i - 1]
        if p.any():
            dxy[i] = -(tips_w[i, p, :2] - tips_w[i - 1, p, :2]).mean(0)
    off = gsmooth(np.cumsum(dxy, axis=0), min(6, T // 4 or 1))
    t_b[:, :2] += off; tips_w[:, :, :2] += off[:, None, :]; knee_p[:, :, :2] += off[:, None, :]

    zshift = tips_w[:, :, 2].min() - PAW_DROP
    t_b[:, 2] -= zshift; tips_w[:, :, 2] -= zshift; knee_p[:, :, 2] -= zshift
    slip1 = slip_mm(tips_w)
    travel1 = float(np.linalg.norm(t_b[-1, :2] - t_b[0, :2]))
    print(f"contact-consistent root: foot slip {slip0:.1f} -> {slip1:.1f} mm/stance (gate <5) "
          f"| root travel {travel0:.3f} -> {travel1:.3f} m")
    print(f"ground shift {zshift*1000:+.1f} mm -> min foot z {tips_w[:,:,2].min()*1000:.1f} mm")
    print(f"IK foot error: mean {err.mean()*1000:.2f} mm p95 {np.percentile(err,95)*1000:.2f} mm max {err.max()*1000:.2f} mm")
    pc = 100.0 * clamped / T
    bad = {DOF_ORDER[i]: f"{pc[i]:.1f}%" for i in range(12) if pc[i] > 0.5}
    print(f"frames at a joint limit: {'none >0.5%' if not bad else bad}")

    dof_vel = np.gradient(dof, dt, axis=0)
    body_pos = np.concatenate([t_b[:, None, :], tips_w], axis=1)
    body_rot = np.zeros((T, 5, 4))
    for i in range(T):
        body_rot[i, 0] = mat_to_quat(R_b[i])
        for k in range(4):
            body_rot[i, k+1] = mat_to_quat(knee_R[i, k])
    body_lin = np.gradient(body_pos, dt, axis=0)
    body_ang = np.zeros((T, 5, 3))
    for b in range(5):
        for i in range(1, T-1):
            Ra, Rc = body_rot[i-1, b], body_rot[i+1, b]
            from_q = lambda q: np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                                          [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
                                          [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]])
            E = from_q(Ra).T @ from_q(Rc)
            body_ang[i, b] = np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]]) * 0.5 / dt
        body_ang[0, b] = body_ang[1, b]; body_ang[-1, b] = body_ang[-2, b]

    vmax = np.abs(dof_vel).max()
    print(f"max |dof velocity| {vmax:.2f} rad/s (limit 10) frames over: {(np.abs(dof_vel).max(1) > 10).sum()}")
    print(f"duty factor: {dict(zip(LEGS, ct.mean(0).round(2)))}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz,
             fps=np.array(hz), dof_names=np.array(DOF_ORDER), body_names=np.array(BODY_NAMES),
             dof_positions=dof.astype(np.float32), dof_velocities=dof_vel.astype(np.float32),
             body_positions=body_pos.astype(np.float32), body_rotations=body_rot.astype(np.float32),
             body_linear_velocities=body_lin.astype(np.float32), body_angular_velocities=body_ang.astype(np.float32),
             contacts=ct, source=np.array(str(source_npz.name)))
    print(f"wrote {out_npz}")

    metrics = dict(
        source=str(source_npz.name), dog_leg_len_m=dog_leg_len, scale_per_leg=s_leg,
        base_fit_residual_mm={"mean": float(fit_res.mean()*1000), "max": float(fit_res.max()*1000)},
        stance_shift_mm=float(dz_best*1000), amplitude_per_leg=alpha,
        ik_foot_error_mm={"mean": float(err.mean()*1000), "p95": float(np.percentile(err,95)*1000), "max": float(err.max()*1000)},
        joint_limit_pct={DOF_ORDER[i]: float(pc[i]) for i in range(12)},
        max_dof_velocity_rad_s=float(vmax), frames_over_vel_limit=int((np.abs(dof_vel).max(1) > 10).sum()),
        foot_slip_mm_per_stance={"before_correction": slip0, "after_correction": slip1},
        root_travel_m={"before_correction": travel0, "after_correction": travel1},
        duty_factor={l: float(v) for l, v in zip(LEGS, ct.mean(0))},
        rate_limit_smoothing_samples=int(sigma),
        ground_shift_mm=float(zshift*1000),
    )
    return metrics, dict(t_b=t_b, R_b=R_b, tips_w=tips_w, dof=dof, dof_vel=dof_vel, ct=ct, timestamps=d["timestamps"], lims=lims)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--urdf", type=Path,
                     default=Path("/pub0/muhammadf/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--root-method", choices=["kabsch", "smal", "yaw_only"], default="smal",
                     help="How to derive Bingo's per-frame origin pose. 'smal' (default) uses the "
                          "source's own root_pos/root_orient_rotmat directly. See compute_root_pose's "
                          "docstring for why 'kabsch' (the original approach) was replaced.")
    ap.add_argument("--yaw-smooth", type=float, default=2.0,
                     help="Gaussian smoothing sigma (samples) applied to the heading angle, "
                          "'yaw_only' method only.")
    args = ap.parse_args()
    report_path = args.report or args.out.with_name(args.out.stem + "_metrics.json")
    metrics, _ = retarget(args.source, args.urdf, args.out, report_path,
                           root_method=args.root_method, yaw_smooth_sigma=args.yaw_smooth)
    import json
    metrics["root_method"] = args.root_method
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
