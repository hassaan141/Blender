"""Build the morphology-tolerant AMP expert transition buffer from lifelike_dog BVH clips.

Design (see AMP_PIPELINE_REPORT.md for the full rationale): rather than retargeting
dog joint ANGLES onto Bingo's own joint space (the InterPet4D approach, which
produced a kinematically-valid but dynamically-infeasible motion -- see
docs/locomotion2/LOCOMOTION2_PIPELINE_REPORT.md), the AMP discriminator compares a
morphology-tolerant STYLE FEATURE VECTOR computed independently on each side (dog
reference vs. simulated Bingo) from its OWN skeleton, each normalized by its OWN
characteristic leg scale. Neither side is ever mapped into the other's joint space.

Per-frame feature schema (61-dim), all quantities expressed in the character's own
root-local frame (rotate world vectors by R_root^T) so absolute world position and
heading are never encoded, per the brief:
    [0:3]   root-local linear velocity, normalized by leg_scale  ("leg-lengths/s")
    [3:6]   root-local angular velocity (rad/s, already scale-free)
    [6:9]   gravity vector in the root-local frame (unit vector when static)
    [9:33]  4 legs x 2 segments (hip->mid, mid->paw) x 3 = per-segment UNIT
            direction vectors in the root-local frame (direction only -> already
            scale-invariant, no leg-scale normalization needed)
    [33:45] 4 legs x 3: paw position relative to root, root-local frame,
            normalized by leg_scale
    [45:57] 4 legs x 3: paw velocity relative to root, root-local frame,
            normalized by leg_scale
    [57:61] 4 legs: ground-contact state in {0,1} (Otsu+hysteresis on world paw
            height, per-window)
Leg order is always [fl, fr, bl, br] to match Bingo's canonical order.

A CRITICAL finding from dataset_inspect.py motivates normalizing velocity too
(not just position, as the brief's wording literally asked for): this dog's
natural walk speed is ~1.0-1.5 m/s, 4-6x Bingo's 0.15-0.30 m/s target. Encoding
RAW root velocity as a style feature would force the discriminator to prefer
Bingo moving at dog speed to "look expert", fighting the velocity-tracking task
reward directly (the exact AMP failure mode the prior rev_3-era experiment's
tuning notes already flag). Normalizing by leg_scale turns speed into a
dimensionless "leg-lengths per second" rate that both a big dog and a small
robot can match at their OWN natural pace.

Expert transitions are (feature_t, feature_t+1) pairs, built within a window only
(never across a window/clip boundary). Left/right mirroring doubles the walk data:
mirrored about the plane containing the window's mean heading direction and
world-up, using a proper Householder reflection applied to raw joint
positions/rotations BEFORE recomputing features (so velocities/angular velocities
fall out as ordinary finite differences of an already-consistent mirrored
trajectory, not error-prone hand-flipped pseudovectors).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bvh_parser import detect_scale, forward_kinematics, parse_bvh

DATA_DIR = "/pub0/muhammadf/Blender/dataset/lifelike_dog/raw_bvh/raw_bvh_data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LEGS = ["fl", "fr", "bl", "br"]
# dog bone names per canonical leg (anatomical L/R matches Bingo's L/R directly,
# same convention already established in docs/locomotion2/dataset/retarget_to_bingo.py)
DOG_JOINTS = {
    "fl": dict(hip="b_LeftArm", mid="b_LeftForeArm", paw="b_LeftHand"),
    "fr": dict(hip="b_RightArm", mid="b_RightForeArm", paw="b_RightHand"),
    "bl": dict(hip="b_LeftLegUpper", mid="b_LeftAnkle", paw="b_LeftToe"),
    "br": dict(hip="b_RightLegUpper", mid="b_RightAnkle", paw="b_RightToe"),
}
MIRROR_PAIR = {"fl": "fr", "fr": "fl", "bl": "br", "br": "bl"}
ROOT_NAME = "Bip01"

BINGO_LEG_SCALE = 0.2036  # m, mean of Urdf.leg_reach() over fl/fr/bl/br (v4 physics URDF)
# Each side normalizes by its OWN leg scale (not a shared constant) -- that's the whole
# point of a morphology-tolerant feature: a 0.6m-legged dog and a 0.2m-legged robot
# should each read as "~1 leg-length" when their leg is near full extension. Measured
# as the p95 hip->paw straight-line distance over the walk windows (a "near-full-reach"
# stat, robust to a handful of outlier frames, analogous to Urdf.leg_reach()'s use of
# the geometric max reach for Bingo).
DOG_LEG_SCALE = 0.58  # m, p95 hip->paw distance measured across the selected walk windows

# expert windows selected by dataset_inspect.py / DATASET_REPORT.md.
# "idle": genuine stationary sub-windows found dataset-wide (excluding aggressive/
#   turning clips -- only from clips otherwise in-scope for this first controller).
# "walk": heading-stable, straightness>0.85, duration>=2s windows scanned fresh below
#   from the two long quad_walk captures (not the "hit"/"star_walk" clips).
IDLE_WINDOWS = [
    ("dog_idle_002.bvh", 0.7, 2.0),
    ("dog_quad_walk_001.bvh", 0.1, 1.3),
]
WALK_SOURCE_FILES = ["dog_quad_walk_001.bvh", "dog_quad_walk_002.bvh"]


def smooth(x, win):
    if win <= 1:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def rolling_circstd(theta, win):
    c, s = np.cos(theta), np.sin(theta)
    cw, sw = smooth(c, win), smooth(s, win)
    R = np.clip(np.sqrt(cw ** 2 + sw ** 2), 1e-9, 1.0)
    return np.sqrt(-2 * np.log(R))


def scan_walk_windows(fname: str, min_dur=2.0, min_straight=0.85):
    b = parse_bvh(os.path.join(DATA_DIR, fname))
    scale = detect_scale(b)
    pos, rot = forward_kinematics(b, scale=scale)
    names = [j.name for j in b.joints]
    root = pos[:, names.index(ROOT_NAME)]
    dt = b.frame_time
    fps = b.fps
    T = b.n_frames
    win_speed = max(1, int(round(0.25 * fps)))
    rx, rz = smooth(root[:, 0], win_speed), smooth(root[:, 2], win_speed)
    vx, vz = np.gradient(rx, dt), np.gradient(rz, dt)
    speed = np.sqrt(vx ** 2 + vz ** 2)
    heading = np.arctan2(vz, vx)
    win_head = max(1, int(round(1.0 * fps)))
    heading_std = rolling_circstd(heading, win_head)
    good = (speed > 0.04) & (heading_std < np.radians(20))
    min_len = int(round(min_dur * fps))
    windows = []
    i = 0
    while i < T:
        if good[i]:
            j = i
            while j < T and good[j]:
                j += 1
            if j - i >= min_len:
                disp = np.hypot(rx[j - 1] - rx[i], rz[j - 1] - rz[i])
                path_len = np.sum(np.hypot(np.diff(rx[i:j]), np.diff(rz[i:j])))
                straightness = disp / (path_len + 1e-9)
                if straightness >= min_straight:
                    windows.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return windows


def load_window(fname: str, t0: float, t1: float):
    """FK just what's needed for one window: root + 4 legs x {hip,mid,paw}."""
    b = parse_bvh(os.path.join(DATA_DIR, fname))
    scale = detect_scale(b)
    pos, rot = forward_kinematics(b, scale=scale)
    names = [j.name for j in b.joints]
    fps = b.fps
    i0, i1 = int(round(t0 * fps)), int(round(t1 * fps))
    i1 = min(i1, b.n_frames)
    root_pos = pos[i0:i1, names.index(ROOT_NAME)]
    root_rot = rot[i0:i1, names.index(ROOT_NAME)]
    legs = {}
    for leg, jn in DOG_JOINTS.items():
        legs[leg] = dict(
            hip=pos[i0:i1, names.index(jn["hip"])],
            mid=pos[i0:i1, names.index(jn["mid"])],
            paw=pos[i0:i1, names.index(jn["paw"])],
        )
    return dict(root_pos=root_pos, root_rot=root_rot, legs=legs, dt=1.0 / fps, fps=fps, file=fname, t0=t0, t1=t1)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(T,3,3) -> (T,4) wxyz quaternions, sign-continuous along T (for slerp)."""
    T = R.shape[0]
    q = np.zeros((T, 4))
    tr = np.trace(R, axis1=1, axis2=2)
    for t in range(T):
        Rt, trt = R[t], tr[t]
        if trt > 0:
            s = np.sqrt(trt + 1.0) * 2
            q[t] = [0.25 * s, (Rt[2, 1] - Rt[1, 2]) / s, (Rt[0, 2] - Rt[2, 0]) / s, (Rt[1, 0] - Rt[0, 1]) / s]
        else:
            i = int(np.argmax([Rt[0, 0], Rt[1, 1], Rt[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + Rt[0, 0] - Rt[1, 1] - Rt[2, 2]) * 2
                q[t] = [(Rt[2, 1] - Rt[1, 2]) / s, 0.25 * s, (Rt[0, 1] + Rt[1, 0]) / s, (Rt[0, 2] + Rt[2, 0]) / s]
            elif i == 1:
                s = np.sqrt(1.0 + Rt[1, 1] - Rt[0, 0] - Rt[2, 2]) * 2
                q[t] = [(Rt[0, 2] - Rt[2, 0]) / s, (Rt[0, 1] + Rt[1, 0]) / s, 0.25 * s, (Rt[1, 2] + Rt[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + Rt[2, 2] - Rt[0, 0] - Rt[1, 1]) * 2
                q[t] = [(Rt[1, 0] - Rt[0, 1]) / s, (Rt[0, 2] + Rt[2, 0]) / s, (Rt[1, 2] + Rt[2, 1]) / s, 0.25 * s]
        if t > 0 and np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _slerp(q0, q1, alpha):
    dot = np.clip(np.sum(q0 * q1, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    small = theta < 1e-6
    s0 = np.where(small, 1 - alpha, np.sin((1 - alpha) * theta) / np.sin(np.where(small, 1.0, theta)))
    s1 = np.where(small, alpha, np.sin(alpha * theta) / np.sin(np.where(small, 1.0, theta)))
    return s0[:, None] * q0 + s1[:, None] * q1


def resample_window(w: dict, target_fps: float = 30.0, stretch: float = 1.0) -> dict:
    """Resample a window from its native BVH fps down to the sim control rate
    (30 Hz = 120 Hz physics / decimation 4, matching the AMP env's sim cfg), so
    expert transition pairs span the SAME dt as the policy's own control-step
    transitions. Positions: linear interp. Rotations: quaternion slerp.

    `stretch` > 1 re-times the clip into slow motion by that factor: the SAME
    pose sequence (same normalized gait shape) is deemed to span `stretch`x as
    much wall-clock time, which divides every velocity-like feature (root vel,
    ang vel, paw vel) by `stretch` while leaving purely geometric channels
    (segment directions, paw-rel positions, contact pattern-vs-phase) unchanged.
    This is how the dog's native ~1.0-1.5 m/s walk gets re-timed to Bingo's
    0.15-0.30 m/s task band WITHOUT altering the gait's shape -- see
    AMP_PIPELINE_REPORT.md for why leg-scale position normalization alone does
    not fix this (the dog's stepping rate is high relative to leg length too:
    ~5-6 leg-lengths/s native vs. Bingo's ~0.7-1.5 leg-lengths/s target)."""
    src_fps = w["fps"]
    n_src = w["root_pos"].shape[0]
    t_src = np.arange(n_src) / src_fps * stretch
    dur = t_src[-1]
    n_dst = max(2, int(np.floor(dur * target_fps)) + 1)
    t_dst = np.arange(n_dst) / target_fps
    t_dst = np.clip(t_dst, 0, t_src[-1])

    def interp_pos(p):
        return np.stack([np.interp(t_dst, t_src, p[:, c]) for c in range(3)], axis=-1)

    def interp_rot(R):
        q = _mat_to_quat(R)
        idx = np.searchsorted(t_src, t_dst, side="right") - 1
        idx = np.clip(idx, 0, n_src - 2)
        alpha = (t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx] + 1e-12)
        qi = _slerp(q[idx], q[idx + 1], alpha)
        qi /= np.linalg.norm(qi, axis=-1, keepdims=True)
        return _quat_to_mat(qi)

    new_legs = {}
    for leg in LEGS:
        src = w["legs"][leg]
        new_legs[leg] = dict(hip=interp_pos(src["hip"]), mid=interp_pos(src["mid"]), paw=interp_pos(src["paw"]))
    return dict(root_pos=interp_pos(w["root_pos"]), root_rot=interp_rot(w["root_rot"]), legs=new_legs,
                dt=1.0 / target_fps, fps=target_fps, file=w["file"], t0=w["t0"], t1=w["t1"])


def mirror_window(w: dict) -> dict:
    """Householder-reflect the whole window across the plane spanned by the
    window's mean heading direction and world-up (Y), then swap L<->R legs.
    Applied to raw positions/rotations so all downstream derivatives (velocity,
    angular velocity) come out correct automatically."""
    root_pos = w["root_pos"]
    dt = w["dt"]
    rx, rz = root_pos[:, 0], root_pos[:, 2]
    vx, vz = np.gradient(rx, dt), np.gradient(rz, dt)
    h = np.array([vx.mean(), 0.0, vz.mean()])
    if np.linalg.norm(h) < 1e-6:
        h = np.array([1.0, 0.0, 0.0])
    h = h / np.linalg.norm(h)
    up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(up, h)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-12)
    M = np.eye(3) - 2.0 * np.outer(lateral, lateral)  # Householder reflection

    def refl_pos(p):
        return p @ M.T

    def refl_rot(R):
        return np.einsum("ij,tjk,kl->til", M, R, M)

    new_root_pos = refl_pos(root_pos)
    new_root_rot = refl_rot(w["root_rot"])
    new_legs = {}
    for leg in LEGS:
        src = w["legs"][MIRROR_PAIR[leg]]
        new_legs[leg] = dict(hip=refl_pos(src["hip"]), mid=refl_pos(src["mid"]), paw=refl_pos(src["paw"]))
    return dict(root_pos=new_root_pos, root_rot=new_root_rot, legs=new_legs, dt=dt, fps=w["fps"],
                file=w["file"] + "#mirror", t0=w["t0"], t1=w["t1"])


def rotmat_log(R: np.ndarray) -> np.ndarray:
    """(T,3,3) rotation matrices -> (T,3) axis*angle rotation vectors."""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    axis = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=-1)
    small = theta < 1e-6
    denom = np.where(small, 1.0, 2.0 * np.sin(np.where(small, 1.0, theta)))
    axis = axis / denom[:, None]
    return axis * theta[:, None]


def contact_mask(paw_y: np.ndarray) -> np.ndarray:
    """Otsu threshold + hysteresis on world paw height -> {0,1} contact."""
    lo, hi = np.percentile(paw_y, [2, 98])
    if hi - lo < 1e-4:
        return np.ones_like(paw_y)
    hist, edges = np.histogram(paw_y, bins=64, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    best_t, best_var = edges[0], -1
    cum_p, cum_pm = 0.0, 0.0
    total_m = np.sum(p * (edges[:-1] + edges[1:]) / 2)
    for k in range(len(p)):
        m = (edges[k] + edges[k + 1]) / 2
        cum_p += p[k]
        cum_pm += p[k] * m
        if 0 < cum_p < 1:
            mA = cum_pm / cum_p
            mB = (total_m - cum_pm) / (1 - cum_p)
            var = cum_p * (1 - cum_p) * (mA - mB) ** 2
            if var > best_var:
                best_var, best_t = var, edges[k + 1]
    low_t = lo + 0.4 * (best_t - lo)
    high_t = lo + 0.9 * (best_t - lo)
    contact = np.zeros(len(paw_y), dtype=bool)
    state = paw_y[0] < best_t
    for i, y in enumerate(paw_y):
        if state and y > high_t:
            state = False
        elif not state and y < low_t:
            state = True
        contact[i] = state
    return contact.astype(np.float32)


def compute_features(w: dict, leg_scale: float) -> tuple[np.ndarray, dict]:
    root_pos, root_rot, dt = w["root_pos"], w["root_rot"], w["dt"]
    T = root_pos.shape[0]
    root_vel_w = np.gradient(root_pos, dt, axis=0)
    root_vel_local = np.einsum("tji,tj->ti", root_rot, root_vel_w)  # R^T @ v
    root_vel_local /= leg_scale

    ang_vel_local = np.zeros((T, 3))
    if T > 1:
        R_rel = np.einsum("tji,tjk->tik", root_rot[:-1], root_rot[1:])  # R_t^T R_{t+1}
        rotvec = rotmat_log(R_rel) / dt
        ang_vel_local[:-1] = rotvec
        ang_vel_local[-1] = ang_vel_local[-2] if T > 1 else 0.0

    gravity_world = np.array([0.0, -1.0, 0.0])  # BVH is Y-up
    gravity_local = np.einsum("tji,j->ti", root_rot, gravity_world)

    seg_dirs = np.zeros((T, 4, 2, 3))
    paw_rel = np.zeros((T, 4, 3))
    contacts = np.zeros((T, 4))
    for li, leg in enumerate(LEGS):
        hip, mid, paw = w["legs"][leg]["hip"], w["legs"][leg]["mid"], w["legs"][leg]["paw"]
        d1 = mid - hip
        d1 = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-9)
        d2 = paw - mid
        d2 = d2 / (np.linalg.norm(d2, axis=-1, keepdims=True) + 1e-9)
        seg_dirs[:, li, 0] = np.einsum("tji,tj->ti", root_rot, d1)
        seg_dirs[:, li, 1] = np.einsum("tji,tj->ti", root_rot, d2)
        rel = paw - root_pos
        paw_rel[:, li] = np.einsum("tji,tj->ti", root_rot, rel) / leg_scale
        contacts[:, li] = contact_mask(paw[:, 1])

    paw_vel = np.gradient(paw_rel, dt, axis=0)  # already root-local + leg_scale-normalized

    feat = np.concatenate([
        root_vel_local, ang_vel_local, gravity_local,
        seg_dirs.reshape(T, 24), paw_rel.reshape(T, 12), paw_vel.reshape(T, 12), contacts,
    ], axis=-1)
    assert feat.shape[1] == 61
    diag = dict(contact_duty={leg: float(contacts[:, i].mean()) for i, leg in enumerate(LEGS)},
                speed_leglenps=float(np.linalg.norm(root_vel_local, axis=-1).mean()))
    return feat.astype(np.float32), diag


def main():
    windows_meta = []
    all_feats = []
    all_bounds = []  # (start_idx, end_idx) per contiguous window, for transition pairing

    def add_window(w, tag, stretch=1.0):
        feat, diag = compute_features(w, DOG_LEG_SCALE)
        start = sum(len(f) for f in all_feats)
        all_feats.append(feat)
        all_bounds.append((start, start + len(feat)))
        windows_meta.append(dict(file=w["file"], t0=w["t0"], t1=w["t1"], tag=tag,
                                  n_frames=len(feat), stretch=stretch, **diag))

    TARGET_FPS = 30.0  # 120Hz physics / decimation 4, matches the AMP env's control rate
    TARGET_WALK_SPEED_MPS = 0.22  # middle of Bingo's 0.15-0.30 m/s task band

    for fname, t0, t1 in IDLE_WINDOWS:
        w = resample_window(load_window(fname, t0, t1), TARGET_FPS)  # no re-timing: already near-zero speed
        add_window(w, "idle")
        add_window(mirror_window(w), "idle_mirror")

    walk_window_specs = []
    for fname in WALK_SOURCE_FILES:
        for (t0, t1) in scan_walk_windows(fname):
            walk_window_specs.append((fname, t0, t1))

    for fname, t0, t1 in walk_window_specs:
        w_native = load_window(fname, t0, t1)
        win_sm = max(1, int(round(0.25 * w_native["fps"])))
        rx, rz = smooth(w_native["root_pos"][:, 0], win_sm), smooth(w_native["root_pos"][:, 2], win_sm)
        vx, vz = np.gradient(rx, w_native["dt"]), np.gradient(rz, w_native["dt"])
        native_speed_mps = float(np.mean(np.sqrt(vx ** 2 + vz ** 2)))
        stretch = max(1.0, native_speed_mps / TARGET_WALK_SPEED_MPS)
        w = resample_window(w_native, TARGET_FPS, stretch=stretch)
        add_window(w, "walk", stretch=stretch)
        add_window(mirror_window(w), "walk_mirror", stretch=stretch)

    features = np.concatenate(all_feats, axis=0)

    # ---- validation ----
    n_nan = int(np.isnan(features).sum())
    jumps = []
    for s, e in all_bounds:
        if e - s > 1:
            d = np.linalg.norm(np.diff(features[s:e], axis=0), axis=-1)
            jumps.append(d.max())
    max_jump = max(jumps) if jumps else 0.0

    feat_t, feat_tp1, is_idle = [], [], []
    for (s, e), m in zip(all_bounds, windows_meta):
        if e - s > 1:
            feat_t.append(features[s:e - 1])
            feat_tp1.append(features[s + 1:e])
            is_idle.append(np.full(e - s - 1, m["tag"].startswith("idle")))
    feat_t = np.concatenate(feat_t, axis=0)
    feat_tp1 = np.concatenate(feat_tp1, axis=0)
    is_idle = np.concatenate(is_idle, axis=0)

    # ---- oversample idle transitions (run_01 finding: idle was <0.5% of the buffer,
    # so the discriminator essentially never learned "expert standing" and kept
    # pulling the policy toward walk-like leg motion even at cmd_vx=0, fighting the
    # task reward and causing 100% falls on "stand" -- see AMP_PIPELINE_REPORT.md).
    # Target: idle:walk ratio in the expert buffer ~= the training-time stand:walk
    # COMMAND ratio (stand_prob=0.20 in bingo_dog_amp_env_cfg.py), i.e. idle should
    # be ~20% of what the discriminator sees, not <1%.
    IDLE_TARGET_FRAC = 0.20
    n_idle, n_walk = int(is_idle.sum()), int((~is_idle).sum())
    if n_idle > 0 and n_walk > 0:
        target_idle_count = int(round(IDLE_TARGET_FRAC / (1 - IDLE_TARGET_FRAC) * n_walk))
        repeat = max(1, round(target_idle_count / n_idle))
        idle_idx = np.where(is_idle)[0]
        rep_idx = np.tile(idle_idx, repeat - 1)  # -1: the originals are already in feat_t/feat_tp1
        if len(rep_idx) > 0:
            feat_t = np.concatenate([feat_t, feat_t[rep_idx]], axis=0)
            feat_tp1 = np.concatenate([feat_tp1, feat_tp1[rep_idx]], axis=0)
        print(f"idle oversample: {n_idle} native idle transitions x{repeat} -> "
              f"{n_idle * repeat} of {len(feat_t)} total ({100*n_idle*repeat/len(feat_t):.1f}%)")

    out_path = os.path.join(CACHE_DIR, "dog_amp_expert.npz")
    np.savez(
        out_path,
        feat_t=feat_t.astype(np.float32), feat_tp1=feat_tp1.astype(np.float32),
        all_features=features.astype(np.float32),
        window_bounds=np.array(all_bounds),
        dog_leg_scale=np.array(DOG_LEG_SCALE),
        bingo_leg_scale=np.array(BINGO_LEG_SCALE),
        fps=np.array(30.0),  # resampled target control rate; see note below
        feature_dim=np.array(61),
        schema=np.array(
            "root_vel_local(3)/leg_scale, ang_vel_local(3), gravity_local(3), "
            "seg_dirs(4legsx2segx3=24), paw_rel(4x3=12)/leg_scale, paw_vel(4x3=12)/leg_scale, "
            "contacts(4). leg order fl,fr,bl,br. leg_scale is EACH SIDE'S OWN scale "
            "(dog_leg_scale here; the Isaac env uses bingo_leg_scale on the policy side)."
        ),
    )
    with open(os.path.join(OUT_DIR, "expert_buffer_report.md"), "w") as f:
        f.write("# Dog AMP Expert Buffer Report\n\n")
        f.write(f"Windows used: {len(windows_meta)} (incl. mirrored copies)\n\n")
        f.write("| file | tag | t0 | t1 | frames | contact duty fl/fr/bl/br | mean speed (leg-lengths/s) | stretch | approx m/s |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for m in windows_meta:
            cd = m["contact_duty"]
            approx_mps = m["speed_leglenps"] * DOG_LEG_SCALE
            f.write(f"| {m['file']} | {m['tag']} | {m['t0']:.2f} | {m['t1']:.2f} | {m['n_frames']} | "
                     f"{cd['fl']:.2f}/{cd['fr']:.2f}/{cd['bl']:.2f}/{cd['br']:.2f} | {m['speed_leglenps']:.2f} | "
                     f"{m['stretch']:.2f} | {approx_mps:.3f} |\n")
        f.write(f"\nTotal frames: {len(features)}  |  transition pairs: {len(feat_t)}\n")
        f.write(f"\nNaN count: {n_nan}  |  max per-frame feature-vector jump (L2): {max_jump:.3f}\n")
        f.write(f"\nSaved to `{out_path}`.\n")

    print(f"windows: {len(windows_meta)}  total frames: {len(features)}  transitions: {len(feat_t)}")
    print(f"NaN count: {n_nan}  max jump: {max_jump:.3f}")
    for m in windows_meta:
        print(m)

    # ---- debug plot ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    walk_i = next(i for i, m in enumerate(windows_meta) if m["tag"] == "walk")
    s, e = all_bounds[walk_i]
    axes[0, 0].plot(features[s:e, 0], label="vx_local (leg-len/s)")
    axes[0, 0].plot(features[s:e, 2], label="vz_local (leg-len/s)")
    axes[0, 0].legend(); axes[0, 0].set_title("root-local linear velocity (one walk window)")

    for i, leg in enumerate(LEGS):
        axes[0, 1].plot(features[s:e, 57 + i], label=leg)
    axes[0, 1].legend(); axes[0, 1].set_title("contact states (one walk window)")

    axes[1, 0].hist(features[:, 0], bins=40)
    axes[1, 0].set_title("distribution: root-local vx (leg-lengths/s), all frames")

    for i, leg in enumerate(LEGS):
        axes[1, 1].plot(features[s:e, 33 + i * 3 + 1], label=f"{leg} paw_rel_y")
    axes[1, 1].legend(); axes[1, 1].set_title("paw height rel. root, root-local Y (one walk window)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "expert_buffer_debug.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {os.path.join(OUT_DIR, 'expert_buffer_report.md')}")
    print(f"wrote {os.path.join(OUT_DIR, 'expert_buffer_debug.png')}")


if __name__ == "__main__":
    main()
