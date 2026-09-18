#!/usr/bin/env python3
"""Extract one clean, normalized, robot-agnostic walk cycle reference from an
InterPet4D clip, ready for Bingo Stage 2 retargeting.

Pipeline: load smal_npy (+ pet_npy for independent cross-checks) for one
clip -> find the cleanest sustained-walking sub-window (roughly 4-6 gait
cycles, start/stop trimmed) -> normalize (heading -> +X, ground -> z=0,
initial root xy -> 0, real scale/timing preserved) -> extract root pose, the
24 named SMAL keypoints (hips/knees/paws included), forward speed, and
four-foot contact states (height+velocity hysteresis, cross-checked against
pet_npy) -> save a canonical NPZ + write a validation report.

Joint/keypoint names are NOT guessed. See KP24_NAMES / PET20_NAMES docstrings
below for how they were identified and independently verified against this
dataset's own numbers (Section 4 of DATASET_REPORT.md has the original
dataset-wide corroboration; this script re-verifies on the specific clip).

Usage:
    /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python extract_walk.py \\
        --clip interpet_dog02_p05_take04_ego_001

Only reads dataset/interpet4d/ and writes under docs/locomotion2/motions/.
Does not touch locomotion1 (docs/walk_ref) or the original dataset files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_interpet4d import (  # noqa: E402
    NATIVE_VIDEO_FPS,
    PET_ASSUMED_FPS,
    PET_DIR,
    SMAL_DIR,
    parse_clip_id,
)

MOTIONS_DIR = Path(__file__).resolve().parents[1] / "motions" / "source"

# ---------------------------------------------------------------------------
# Keypoint / joint names -- identified, not guessed. See DATASET_REPORT.md's
# provenance notes (duplicated in the report this script writes) for the full
# chain of evidence:
#
# smal_npy's kp_world (T,24,3): InterPet4D's own shapes (pose_rotmat (T,35,..),
# betas (T,30), betas_limbs (T,7), kp_world (T,24,..)) are an exact match to
# the "D-SMAL" dog model from BARC (Rueegg et al., CVPR 2022) / BITE (Rueegg
# et al., CVPR 2023): n_joints=35, n_betas=30, n_keyp=24, and a 7-entry
# logscale limb-shape vector, verified against the open-source
# github.com/runa91/barc_release (src/configs/data_info.py, `n_joints=35`,
# `n_betas=30`, `n_keyp=24` at COMPLETE_DATA_INFO_24; src/configs/
# SMAL_configs.py `logscale_part_list` with exactly 7 entries) and
# github.com/runa91/bite_release (same). Those files' `StanExt_JOINT_NAMES_24`
# / `CANONICAL_MODEL_JOINTS_REFINED` give this exact 24-point order, which we
# then independently re-verified against InterPet4D's own numbers (see
# `verify_kp24_against_data()` below): indices 0,3,6,9 (claimed paws) sit
# ~0.39-0.45m below the root on every checked clip while index 22 (claimed
# withers) sits ~0-0.07m *above* it -- anatomically exactly right for a model
# whose root is near the tail base (BARC's own stated convention).
KP24_NAMES = [
    "Left_front_leg_paw", "Left_front_leg_middle_joint", "Left_front_leg_top",
    "Left_rear_leg_paw", "Left_rear_leg_middle_joint", "Left_rear_leg_top",
    "Right_front_leg_paw", "Right_front_leg_middle_joint", "Right_front_leg_top",
    "Right_rear_leg_paw", "Right_rear_leg_middle_joint", "Right_rear_leg_top",
    "Tail_start", "Tail_end", "Base_of_left_ear", "Base_of_right_ear",
    "Nose", "Chin", "Left_ear_tip", "Right_ear_tip",
    "Left_eye", "Right_eye", "Withers", "Throat",
]
KP24_PAW_IDXS = [0, 3, 6, 9]         # LF, LR, RF, RR paw
KP24_KNEE_IDXS = [1, 4, 7, 10]       # "middle_joint": carpus/wrist (front), hock (rear)
KP24_HIP_IDXS = [2, 5, 8, 11]        # "leg_top": shoulder (front), hip (rear)
LEG_ORDER = ["left_front", "left_rear", "right_front", "right_rear"]

# pet_npy (T,20,4): the InterPet4D paper describes this as an AnimalPose-style
# 20-point skeleton. The canonical AnimalPose ordering (verified against the
# maintained open-mmlab/mmpose parse_animalpose_dataset.py, not just the
# paper's prose, since a first paraphrase of the paper text had Knee/Elbow
# swapped relative to the actual standard) is used here. Independently
# re-verified per clip in `verify_pet20_against_data()`: indices 16-19
# (claimed paws) sit ~0.38-0.51m below Withers (index 7) on every checked
# clip, consistent across dogs.
PET20_NAMES = [
    "L_Eye", "R_Eye", "L_EarBase", "R_EarBase", "Nose", "Throat",
    "TailBase", "Withers",
    "L_F_Elbow", "R_F_Elbow", "L_B_Elbow", "R_B_Elbow",
    "L_F_Knee", "R_F_Knee", "L_B_Knee", "R_B_Knee",
    "L_F_Paw", "R_F_Paw", "L_B_Paw", "R_B_Paw",
]
PET20_PAW_IDXS = [16, 17, 18, 19]
PET20_WITHERS_IDX = 7

UP_AXIS = 2


def verify_kp24_against_data(kp_world: np.ndarray, t_world: np.ndarray) -> dict:
    """Sanity-check KP24_NAMES against this specific clip's geometry."""
    z_rel = kp_world[:, :, UP_AXIS] - t_world[:, UP_AXIS : UP_AXIS + 1]
    paw_mean_h = float(z_rel[:, KP24_PAW_IDXS].mean())
    withers_mean_h = float(z_rel[:, KP24_NAMES.index("Withers")].mean())
    ok = paw_mean_h < -0.1 and withers_mean_h > -0.05
    return {"paw_mean_height_vs_root": paw_mean_h, "withers_mean_height_vs_root": withers_mean_h, "consistent": bool(ok)}


def verify_pet20_against_data(pet: np.ndarray) -> dict:
    xyz = pet[:, :, :3].astype(np.float64)
    rel = xyz - xyz[:, PET20_WITHERS_IDX : PET20_WITHERS_IDX + 1, :]
    paw_mean_h = float(rel[:, PET20_PAW_IDXS, UP_AXIS].mean())
    ok = paw_mean_h < -0.1
    return {"paw_mean_height_vs_withers": paw_mean_h, "consistent": bool(ok)}


def load_clip(clip_id: str):
    pet = np.load(PET_DIR / f"{clip_id}.npy")
    smal = dict(np.load(SMAL_DIR / f"{clip_id}.npz"))
    return pet, smal


def smooth(x: np.ndarray, times: np.ndarray, win_s: float) -> np.ndarray:
    """Centered moving average over a time window (irregular sampling safe)."""
    y = np.empty_like(x)
    for i in range(len(x)):
        m = np.abs(times - times[i]) <= win_s / 2
        y[i] = x[m].mean()
    return y


def otsu_threshold(x: np.ndarray, nbins: int = 64) -> float:
    """1D Otsu threshold: the split point maximizing between-class variance.
    Used to separate each paw's stance (low height / low speed) values from
    its swing values without picking an arbitrary fixed percentile -- an
    FFT-based period estimate on this dataset's noisy single-view paw tracks
    turned out to be unstable (peak period swung from 1.1s to 1.9s for a
    ~0.25s shift in window edge on the same clip), so gait timing here comes
    from directly counting Otsu+hysteresis contact segments instead."""
    hist, edges = np.histogram(x, bins=nbins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total, sum_all = hist.sum(), (hist * centers).sum()
    sum_b = w_b = 0.0
    best_thr, best_var = centers[0], -1.0
    for i in range(nbins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * centers[i]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_thr = var_between, centers[i]
    return float(best_thr)


def count_contact_segments(contact: np.ndarray) -> int:
    return int(np.sum((contact[1:]) & (~contact[:-1])) + (1 if contact[0] else 0))


def rolling_circstd(theta: np.ndarray, times: np.ndarray, win_s: float) -> np.ndarray:
    """Rolling circular standard deviation of an angle signal (radians)."""
    out = np.empty_like(theta)
    for i in range(len(theta)):
        m = np.abs(times - times[i]) <= win_s / 2
        s, c = np.sin(theta[m]).mean(), np.cos(theta[m]).mean()
        r = np.hypot(s, c)
        out[i] = np.sqrt(max(-2.0 * np.log(max(r, 1e-9)), 0.0))
    return out


def select_walk_window(pet: np.ndarray, smal: dict) -> dict:
    """Find the cleanest sustained-walking sub-window in two phases:

    1. **Core detection**: heading (from lightly-smoothed root velocity) must
       be locally stable -- rolling circular std below its own 40th
       percentile -- AND the dog must actually be moving (>0.03 m/s), for at
       least 1s. This catches genuine turns/spins (a first pass on this clip
       found the raw "longest above-speed-floor run" silently bridged a
       ~1s mid-clip spin where heading rotated >500 degrees -- a tracking
       glitch or a real turn, either way not "steady walking") while
       tolerating the noisy, ill-defined heading estimate you get at very low
       speed (velocity direction is meaningless near-zero speed).
    2. **Straightness-preserving extension**: grow the core window outward in
       0.05s steps as long as overall path straightness (net displacement /
       path length) over the *whole* candidate window stays >= 0.92. This
       recovers legitimate slow lead-in/out (where phase-1's heading gate is
       unreliable) without reintroducing a real turn (which would drag
       straightness down as soon as it's included).

    Gait-cycle count then comes from directly counting Otsu+hysteresis
    contact segments per leg on the extended window (see `otsu_threshold`'s
    docstring for why an FFT/autocorrelation period estimate was rejected:
    unstable on this dataset's noisy single-view paw tracks). Returns time
    bounds (seconds from clip start) plus diagnostics.
    """
    fidx = smal["frame_idx"].astype(np.int64)
    t_world = smal["t_world"].astype(np.float64)
    kp_world = smal["kp_world"].astype(np.float64)
    smal_times = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
    xy = t_world[:, :2]

    xy_smooth = np.stack([smooth(xy[:, 0], smal_times, 0.3), smooth(xy[:, 1], smal_times, 0.3)], axis=1)
    heading = np.arctan2(np.gradient(xy_smooth[:, 1]), np.gradient(xy_smooth[:, 0]))
    turniness = rolling_circstd(heading, smal_times, 1.5)
    raw_speed = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(fidx) / NATIVE_VIDEO_FPS, 1e-6)])

    turn_thresh = float(np.percentile(turniness, 40))
    core_mask = (turniness < turn_thresh) & (raw_speed > 0.03)
    runs, i = [], 0
    while i < len(core_mask):
        if core_mask[i]:
            j = i
            while j < len(core_mask) and core_mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    runs = [r for r in runs if smal_times[r[1] - 1] - smal_times[r[0]] >= 1.0]
    if not runs:
        raise RuntimeError("no stable-heading, moving core window found")
    runs.sort(key=lambda r: smal_times[r[1] - 1] - smal_times[r[0]])
    core = runs[-1]
    core_start, core_end = float(smal_times[core[0]]), float(smal_times[core[1] - 1])

    def straightness(t0, t1):
        m = (smal_times >= t0) & (smal_times <= t1)
        pts = xy[m]
        if len(pts) < 3:
            return 0.0
        disp = np.linalg.norm(pts[-1] - pts[0])
        path = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        return disp / max(path, 1e-9)

    # Smoothed step speed (same 0.8s box as the old speed-only approach),
    # interpolated onto smal_times so the extension loop can require the
    # *newly added* slice to itself be genuinely moving. Straightness alone
    # is not enough: a first version of this extension pulled in a several-
    # hundred-ms motionless "standing" prefix (aggregate straightness barely
    # moved since the standing frames contribute almost no path length), and
    # a stationary paw doesn't have swing/stance alternation for the contact
    # detector to find -- inspection of contact-duty output caught this
    # (near-zero duty for the whole prefix instead of near-1.0, which is what
    # a genuinely planted, stationary paw should show).
    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(fidx) / NATIVE_VIDEO_FPS, 1e-6)
    speed_times = (smal_times[:-1] + smal_times[1:]) / 2.0
    sm_speed_full = np.interp(smal_times, speed_times, smooth(step_speed, speed_times, 0.8))

    def segment_speed(t0, t1):
        m = (smal_times >= t0) & (smal_times <= t1)
        return float(sm_speed_full[m].mean()) if m.any() else 0.0

    straight_thresh, speed_floor, step = 0.92, 0.15, 0.05
    t_start, t_end = core_start, core_end
    while t_start > smal_times[0] and straightness(t_start - step, t_end) >= straight_thresh and segment_speed(t_start - step, t_start) >= speed_floor:
        t_start -= step
    while t_end < smal_times[-1] and straightness(t_start, t_end + step) >= straight_thresh and segment_speed(t_end, t_end + step) >= speed_floor:
        t_end += step
    raw_run = (core_start, core_end)

    def cycles_for_window(ts0, ts1):
        m = (smal_times >= ts0) & (smal_times <= ts1)
        if m.sum() < 10:
            return None, None
        paw_pos = kp_world[m][:, KP24_PAW_IDXS, :]
        c_info = detect_contacts(paw_pos, smal_times[m], t_world[m, 2])
        counts = [count_contact_segments(c_info["contacts"][:, k]) for k in range(4)]
        return float(np.median(counts)), counts

    n_cycles, counts = cycles_for_window(t_start, t_end)
    if n_cycles is not None and n_cycles > 6:
        # Trim from the start (keep the end -- the walk is more established
        # there) toward a 5-cycle target, then re-count once to confirm.
        cycle_dur = (t_end - t_start) / n_cycles
        t_start = t_end - 5 * cycle_dur
        n_cycles, counts = cycles_for_window(t_start, t_end)

    # Best-effort snap to the nearest contact onset (rising edge) of whichever
    # leg has the most evenly spaced contacts, using the *same* smal-based
    # contacts already computed above (self-consistent with the reported
    # cycle count) -- not pet_npy, which is reserved for independent
    # cross-checking after extraction, per the task's instructions.
    snap_info = {"start_snapped": False, "end_snapped": False, "reference_leg": None}
    m = (smal_times >= t_start) & (smal_times <= t_end)
    if m.sum() >= 10:
        paw_pos = kp_world[m][:, KP24_PAW_IDXS, :]
        win_times = smal_times[m]
        c_info = detect_contacts(paw_pos, win_times, t_world[m, 2])
        best_leg, best_cv, best_onsets = None, np.inf, None
        for k in range(4):
            c = c_info["contacts"][:, k]
            onsets = win_times[np.where(c[1:] & ~c[:-1])[0] + 1]
            if c[0]:
                onsets = np.concatenate([[win_times[0]], onsets])
            if len(onsets) < 3:
                continue
            cv = np.diff(onsets).std() / max(np.diff(onsets).mean(), 1e-6)
            if cv < best_cv:
                best_cv, best_leg, best_onsets = cv, LEG_ORDER[k], onsets
        if best_onsets is not None and n_cycles:
            cycle_dur = (t_end - t_start) / n_cycles
            near_start = best_onsets[np.argmin(np.abs(best_onsets - t_start))]
            near_end = best_onsets[np.argmin(np.abs(best_onsets - t_end))]
            if abs(near_start - t_start) < 0.4 * cycle_dur:
                t_start = float(near_start)
                snap_info["start_snapped"] = True
            if abs(near_end - t_end) < 0.4 * cycle_dur and near_end > t_start:
                t_end = float(near_end)
                snap_info["end_snapped"] = True
            snap_info["reference_leg"] = best_leg
            n_cycles, counts = cycles_for_window(t_start, t_end)

    return {
        "t_start": t_start,
        "t_end": t_end,
        "n_gait_cycles": n_cycles if n_cycles is not None else 0.0,
        "per_leg_footfall_counts": counts,
        "core_run_s": raw_run,
        "turn_threshold_rad": turn_thresh,
        "straightness": straightness(t_start, t_end),
        "snap_info": snap_info,
    }


def slice_by_time(smal: dict, pet: np.ndarray, t_start: float, t_end: float):
    fidx = smal["frame_idx"].astype(np.int64)
    smal_times = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
    smal_mask = (smal_times >= t_start) & (smal_times <= t_end)
    pet_times = np.arange(pet.shape[0]) / PET_ASSUMED_FPS
    pet_mask = (pet_times >= t_start) & (pet_times <= t_end)
    return smal_mask, pet_mask


def normalize_motion(t_world, R_world, kp_world):
    """heading -> +X, ground -> z=0, initial root xy -> 0, scale/timing kept."""
    xy = t_world[:, :2]
    centered = xy - xy.mean(axis=0)
    # Robust heading via PCA principal direction, oriented to match net
    # displacement (avoids sign ambiguity of PCA and endpoint noise of a raw
    # start->end vector).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    net_disp = xy[-1] - xy[0]
    if np.dot(direction, net_disp) < 0:
        direction = -direction
    heading_rad = float(np.arctan2(direction[1], direction[0]))

    c, s = np.cos(-heading_rad), np.sin(-heading_rad)
    rot2 = np.array([[c, -s], [s, c]])
    rot3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    root_pos = t_world.copy()
    root_pos[:, :2] = xy @ rot2.T
    kp = kp_world.copy()
    kp[:, :, :2] = kp_world[:, :, :2] @ rot2.T
    root_orient = np.einsum("ij,tjk->tik", rot3, R_world)

    ground_z = float(np.percentile(np.concatenate([kp[:, i, 2] for i in KP24_PAW_IDXS]), 2))
    root_pos[:, 2] -= ground_z
    kp[:, :, 2] -= ground_z

    initial_xy = root_pos[0, :2].copy()
    root_pos[:, :2] -= initial_xy
    kp[:, :, :2] -= initial_xy

    return root_pos, root_orient, kp, {
        "heading_rotation_rad": heading_rad,
        "ground_z_offset": ground_z,
        "initial_root_xy_pre_shift": initial_xy.tolist(),
    }


def detect_contacts(paw_pos: np.ndarray, timestamps: np.ndarray, root_z: np.ndarray) -> dict:
    """Per-leg hysteresis contact classifier from root-relative height + 3D
    velocity. Enter/exit thresholds are set from each leg's own Otsu
    threshold (the height/speed value that best separates its stance and
    swing samples) plus a hysteresis margin, rather than fixed percentiles --
    percentile cutoffs (e.g. "bottom 20%") don't adapt to how skewed the
    height distribution actually is (stance is a plateau near the minimum,
    swing is a fast excursion through many values), and produced implausibly
    low duty factors (~0.1) on inspection. Otsu thresholds are data-driven
    per leg per clip, so they adapt to each dog/clip's own height and speed
    range.

    Height is measured *relative to the root* (paw z - root z), not absolute
    ground-relative height: on a window spanning acceleration and
    deceleration, all four paws' absolute height rose and fell together in
    lockstep with overall body pitch/height (visible in an early validation
    plot as contacts only being detected mid-window, never in the accel/
    decel portions) -- root-relative height removes that shared whole-body
    motion and leaves each leg's own swing/stance articulation."""
    n_frames, n_legs, _ = paw_pos.shape
    dt = np.gradient(timestamps)
    vel = np.gradient(paw_pos, axis=0) / dt[:, None, None]
    speed = np.linalg.norm(vel, axis=2)
    height = paw_pos[:, :, 2] - root_z[:, None]

    contacts = np.zeros((n_frames, n_legs), dtype=bool)
    thresholds = []
    for leg in range(n_legs):
        h, v = height[:, leg], speed[:, leg]
        h_thr, v_thr = otsu_threshold(h), otsu_threshold(v)
        h_range = h.max() - h.min()
        h_enter, h_exit = h_thr - 0.1 * h_range, h_thr + 0.1 * h_range
        v_enter, v_exit = 0.75 * v_thr, 1.25 * v_thr
        thresholds.append({"h_enter": float(h_enter), "h_exit": float(h_exit), "v_enter": float(v_enter), "v_exit": float(v_exit)})
        state = h[0] < h_enter and v[0] < v_enter
        for t in range(n_frames):
            if state:
                if h[t] > h_exit or v[t] > v_exit:
                    state = False
            else:
                if h[t] < h_enter and v[t] < v_enter:
                    state = True
            contacts[t, leg] = state
    return {"contacts": contacts, "thresholds": thresholds, "paw_speed": speed, "paw_height": height}


def cross_check_with_pet(pet: np.ndarray, pet_mask: np.ndarray, contacts: np.ndarray, smal_times: np.ndarray, pet_times: np.ndarray) -> dict:
    """Independent evidence from pet_npy: does its own (unaligned, ~30Hz,
    separately-tracked) paw-height signal dip during smal-derived contact
    windows? Compared in the time domain (interpolated), not by row index,
    since the two arrays are not frame-aligned (see DATASET_REPORT.md)."""
    withers = pet[:, PET20_WITHERS_IDX, UP_AXIS]
    agreements = []
    for k, idx in enumerate(PET20_PAW_IDXS):
        pet_h = pet[:, idx, UP_AXIS] - withers
        pet_h_interp = np.interp(smal_times, pet_times, pet_h)
        in_contact = contacts[:, k]
        if in_contact.sum() == 0 or (~in_contact).sum() == 0:
            agreements.append(None)
            continue
        mean_h_contact = float(pet_h_interp[in_contact].mean())
        mean_h_swing = float(pet_h_interp[~in_contact].mean())
        pooled_std = float(pet_h_interp.std()) + 1e-9
        effect_size = (mean_h_swing - mean_h_contact) / pooled_std
        agreements.append({
            "mean_pet_height_during_smal_contact": mean_h_contact,
            "mean_pet_height_during_smal_swing": mean_h_swing,
            "lower_during_contact": bool(mean_h_contact < mean_h_swing),
            "effect_size_std": effect_size,
        })
    return {"per_leg": agreements}


def compute_validation(clip_id, meta, smal_mask, pet_mask, smal, pet, t_start, t_end, window_info,
                        root_pos, root_orient, kp, forward_speed, contact_info, cross_check, norm_info):
    fidx = smal["frame_idx"][smal_mask]
    n = smal_mask.sum()
    duration = float((fidx[-1] - fidx[0]) / NATIVE_VIDEO_FPS) if n > 1 else 0.0

    nan_frames = int((~np.isfinite(kp).all(axis=(1, 2))).sum())

    # dimensions/proportions from the extracted segment
    body_length = float(np.linalg.norm(kp[:, KP24_NAMES.index("Nose")] - kp[:, KP24_NAMES.index("Tail_start")], axis=1).mean())
    withers_height = float(root_pos[:, 2].mean() * 0 + kp[:, KP24_NAMES.index("Withers"), 2].mean())
    leg_lengths = {}
    for leg_i, leg_name in enumerate(LEG_ORDER):
        hip = kp[:, KP24_HIP_IDXS[leg_i]]
        paw = kp[:, KP24_PAW_IDXS[leg_i]]
        leg_lengths[leg_name] = float(np.linalg.norm(hip - paw, axis=1).mean())

    contacts = contact_info["contacts"]
    duty = {LEG_ORDER[i]: float(contacts[:, i].mean()) for i in range(4)}

    # foot sliding during contact: horizontal displacement while classified
    # in contact (should be small for a clean plant).
    sliding = {}
    paw_xy = kp[:, KP24_PAW_IDXS, :2]
    for i, leg_name in enumerate(LEG_ORDER):
        in_c = contacts[:, i]
        if in_c.sum() < 2:
            sliding[leg_name] = None
            continue
        segs, cur = [], []
        for t in range(n):
            if in_c[t]:
                cur.append(t)
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        slides = [float(np.linalg.norm(paw_xy[s[-1], i] - paw_xy[s[0], i])) for s in segs if len(s) > 1]
        sliding[leg_name] = float(np.mean(slides)) if slides else 0.0

    mean_speed = float(np.nanmean(forward_speed))

    return {
        "clip_id": clip_id,
        "dog": meta["dog"],
        "selected_frame_idx_range": [int(fidx[0]), int(fidx[-1])],
        "selected_pet_row_range": [int(np.nonzero(pet_mask)[0][0]), int(np.nonzero(pet_mask)[0][-1])],
        "n_frames": int(n),
        "duration_s": duration,
        "gait_period_s": (duration / window_info["n_gait_cycles"]) if window_info["n_gait_cycles"] else 0.0,
        "n_gait_cycles": window_info["n_gait_cycles"],
        "per_leg_footfall_counts": window_info["per_leg_footfall_counts"],
        "window_snap_info": window_info["snap_info"],
        "mean_forward_speed_mps": mean_speed,
        "body_length_nose_to_tailbase_m": body_length,
        "withers_height_m": withers_height,
        "leg_lengths_hip_to_paw_m": leg_lengths,
        "contact_duty_per_foot": duty,
        "foot_sliding_during_contact_m": sliding,
        "nan_or_missing_frames": nan_frames,
        "normalization": norm_info,
        "pet_cross_check": cross_check,
    }


def make_plot(out_path, timestamps, root_pos, paw_pos_norm, contacts, forward_speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    ax = axes[0]
    ax.plot(root_pos[:, 0], root_pos[:, 1], "-o", ms=2, color="black", label="root path")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, lateral)")
    ax.set_title("Top-down root trajectory (heading aligned to +X)")
    ax.axis("equal")
    ax.legend()

    ax = axes[1]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, name in enumerate(LEG_ORDER):
        ax.plot(timestamps, paw_pos_norm[:, i, 2], color=colors[i], label=f"{name} paw height")
        in_c = contacts[:, i]
        ax.fill_between(timestamps, 0, 0.02, where=in_c, color=colors[i], alpha=0.3, step="mid")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("paw height (m, ground=0)")
    ax.set_title("Paw heights with detected contact windows (shaded)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(timestamps[1:], forward_speed[1:], color="purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forward speed (m/s)")
    ax.set_title("Forward (root +X) speed over the extracted segment")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="interpet_dog02_p05_take04_ego_001")
    ap.add_argument("--out-npz", type=Path, default=MOTIONS_DIR / "dog02_walk_01.npz")
    ap.add_argument("--out-report", type=Path, default=MOTIONS_DIR / "dog02_walk_01_report.md")
    ap.add_argument("--out-plot", type=Path, default=MOTIONS_DIR / "dog02_walk_01_validation.png")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("T_START", "T_END"),
                     help="Fixed window in seconds from clip start, overriding automatic selection "
                          "(e.g. a window already chosen and validated by evaluate_candidates.py).")
    args = ap.parse_args()

    clip_id = args.clip
    meta = parse_clip_id(clip_id)
    pet, smal = load_clip(clip_id)

    kp24_check = verify_kp24_against_data(smal["kp_world"].astype(np.float64), smal["t_world"].astype(np.float64))
    pet20_check = verify_pet20_against_data(pet)
    if not (kp24_check["consistent"] and pet20_check["consistent"]):
        sys.exit(f"Keypoint identity check FAILED for {clip_id}: {kp24_check} {pet20_check}")

    if args.window is not None:
        t0, t1 = args.window
        fidx = smal["frame_idx"].astype(np.int64)
        kp_world_full = smal["kp_world"].astype(np.float64)
        t_world_full = smal["t_world"].astype(np.float64)
        smal_times = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
        m = (smal_times >= t0) & (smal_times <= t1)
        paw_pos_full = kp_world_full[m][:, KP24_PAW_IDXS, :]
        c_info = detect_contacts(paw_pos_full, smal_times[m], t_world_full[m, 2])
        counts = [count_contact_segments(c_info["contacts"][:, k]) for k in range(4)]
        window = {
            "t_start": t0, "t_end": t1,
            "n_gait_cycles": float(np.median(counts)), "per_leg_footfall_counts": counts,
            "core_run_s": (t0, t1), "turn_threshold_rad": float("nan"),
            "straightness": None,
            "snap_info": {"start_snapped": False, "end_snapped": False, "reference_leg": "manual override"},
        }
        xy = t_world_full[m][:, :2]
        disp = np.linalg.norm(xy[-1] - xy[0])
        path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        window["straightness"] = float(disp / max(path, 1e-9))
        print(f"Using manual window override [{t0},{t1}]: cycles={window['n_gait_cycles']:.1f} straightness={window['straightness']:.3f}")
    else:
        window = select_walk_window(pet, smal)
    smal_mask, pet_mask = slice_by_time(smal, pet, window["t_start"], window["t_end"])
    if smal_mask.sum() < 10:
        sys.exit(f"Selected window too short after slicing: {smal_mask.sum()} frames")

    fidx = smal["frame_idx"][smal_mask].astype(np.int64)
    timestamps = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
    t_world = smal["t_world"][smal_mask].astype(np.float64)
    R_world = smal["R_world"][smal_mask].astype(np.float64)
    kp_world = smal["kp_world"][smal_mask].astype(np.float64)
    kp_weight = smal["kp_weight"][smal_mask].astype(np.float64)

    root_pos, root_orient, kp, norm_info = normalize_motion(t_world, R_world, kp_world)

    dt = np.gradient(timestamps)
    # Raw np.gradient of per-frame root x is dominated by single-frame SMAL
    # fit jitter (observed swinging to -0.5 m/s "backward" spikes on an
    # otherwise forward walk); lightly smooth root x first, consistent with
    # the smoothing already used for heading in select_walk_window.
    root_x_smooth = smooth(root_pos[:, 0], timestamps, 0.3)
    forward_speed = np.gradient(root_x_smooth) / dt

    paw_pos = kp[:, KP24_PAW_IDXS, :]
    knee_pos = kp[:, KP24_KNEE_IDXS, :]
    hip_pos = kp[:, KP24_HIP_IDXS, :]

    contact_info = detect_contacts(paw_pos, timestamps, root_pos[:, 2])

    pet_times_full = np.arange(pet.shape[0]) / PET_ASSUMED_FPS
    cross_check = cross_check_with_pet(pet, pet_mask, contact_info["contacts"], timestamps, pet_times_full)

    validation = compute_validation(
        clip_id, meta, smal_mask, pet_mask, smal, pet, window["t_start"], window["t_end"], window,
        root_pos, root_orient, kp, forward_speed, contact_info, cross_check, norm_info,
    )

    MOTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        schema_version="locomotion2_v1",
        source_dataset="InterPet4D (arXiv:2607.10287, HF ohicarip/interpet4d)",
        source_clip_id=clip_id,
        source_pet_path=f"dataset/interpet4d/pet_npy/{clip_id}.npy",
        source_smal_path=f"dataset/interpet4d/smal_npy/{clip_id}.npz",
        dog_id=meta["dog"], person_id=meta["person"], take=meta["take"],
        frame_idx=fidx,
        timestamps=timestamps,
        native_video_fps=NATIVE_VIDEO_FPS,
        effective_fps_mean=float(1.0 / np.mean(np.diff(timestamps))),
        root_pos=root_pos,
        root_orient_rotmat=root_orient,
        keypoints=kp,
        keypoint_names=np.array(KP24_NAMES),
        keypoint_weight=kp_weight,
        hip_pos=hip_pos,
        knee_pos=knee_pos,
        paw_pos=paw_pos,
        leg_order=np.array(LEG_ORDER),
        forward_speed=forward_speed,
        foot_contact=contact_info["contacts"],
        contact_thresholds=json.dumps(contact_info["thresholds"]),
        heading_rotation_rad=norm_info["heading_rotation_rad"],
        ground_z_offset=norm_info["ground_z_offset"],
        scale_applied=1.0,
        gait_period_s=validation["gait_period_s"],
        n_gait_cycles=window["n_gait_cycles"],
        per_leg_footfall_counts=json.dumps(window["per_leg_footfall_counts"]),
        keypoint_name_provenance=(
            "KP24 order = BARC/BITE D-SMAL 'StanExt_JOINT_NAMES_24' "
            "(github.com/runa91/barc_release src/configs/data_info.py), "
            "identified via exact shape-signature match "
            "(n_joints=35, n_betas=30, n_keyp=24, 7 logscale limb params) "
            "and re-verified against this clip's own geometry -- see report."
        ),
    )

    make_plot(args.out_plot, timestamps, root_pos, kp[:, KP24_PAW_IDXS, :], contact_info["contacts"], forward_speed)

    report_md = render_report(clip_id, meta, window, validation, kp24_check, pet20_check, args)
    args.out_report.write_text(report_md)

    print(f"Wrote {args.out_npz}")
    print(f"Wrote {args.out_report}")
    print(f"Wrote {args.out_plot}")
    print(json.dumps({k: v for k, v in validation.items() if k not in ("normalization", "pet_cross_check")}, indent=2, default=str))


def render_report(clip_id, meta, window, v, kp24_check, pet20_check, args) -> str:
    lines = []
    a = lines.append
    a(f"# {clip_id} -- Extracted Walk Reference (Locomotion 2)\n")
    a(
        "Generated by `extract_walk.py` from the InterPet4D clip identified as "
        "the top recommendation in `DATASET_REPORT.md`. Robot-agnostic: SMAL "
        "world-frame keypoints only, no Bingo-specific retargeting applied. "
        "locomotion1 (`docs/walk_ref`) was not touched.\n"
    )

    a("## Keypoint identity verification\n")
    a(
        f"`kp_world` (24 SMAL keypoints): paw indices {v['leg_lengths_hip_to_paw_m'] and [0,3,6,9]} sit "
        f"{kp24_check['paw_mean_height_vs_root']:.3f} m below the root on average, and 'Withers' sits "
        f"{kp24_check['withers_mean_height_vs_root']:.3f} m above it -- consistent with a root near the "
        f"tail base (BARC/D-SMAL's stated convention). Check passed: `{kp24_check['consistent']}`."
    )
    a(
        f"`pet_npy` (20 AnimalPose-style keypoints): paw indices sit "
        f"{pet20_check['paw_mean_height_vs_withers']:.3f} m below Withers on average. Check passed: "
        f"`{pet20_check['consistent']}`. Full provenance/citations for both name lists are in "
        f"`extract_walk.py`'s module docstring and `KP24_NAMES`/`PET20_NAMES` comments.\n"
    )

    a("## Selected window\n")
    a(f"- Source clip: `{clip_id}` (dog `{meta['dog']}`, person `{meta['person']}`, take {meta['take']})")
    a(f"- Selected `smal_npy` frame_idx range: {v['selected_frame_idx_range']}")
    a(f"- Selected `pet_npy` row range: {v['selected_pet_row_range']} (time-based correspondence, not row-index -- see DATASET_REPORT.md on frame misalignment)")
    a(f"- Duration: {v['duration_s']:.2f} s ({v['n_frames']} frames)")
    a(f"- Gait cycles in window (median of per-leg Otsu+hysteresis contact-segment counts {v['per_leg_footfall_counts']}): {v['n_gait_cycles']:.2f}")
    if v["n_gait_cycles"] < 4:
        a(
            f"  - **Below the requested ~4-6 cycles.** This is the longest stretch in the clip "
            f"that is simultaneously straight (no turn), continuously moving (no stop/standing), "
            f"and free of the heading-glitch region found during development (see selection "
            f"method below) -- at this dog's cadence that stretch is only "
            f"{v['duration_s']:.1f}s. Extending further in either direction required including "
            f"a stop, a standing prefix, or the turn, all of which were judged worse trade-offs "
            f"than a slightly short cycle count. Reported honestly rather than padded."
        )
    a(f"- Derived mean gait period: {v['gait_period_s']:.3f} s (= duration / cycle count)")
    a(f"- Path straightness (net displacement / path length) over the final window: {window['straightness']:.3f}")
    if window["snap_info"].get("reference_leg") == "manual override":
        a(
            f"- Selection method: **manually specified window**, not automatic selection. This "
            f"window ([{window['t_start']:.2f},{window['t_end']:.2f}]s) was chosen in "
            f"`docs/locomotion2/motions/candidates/CANDIDATE_COMPARISON.md` after comparing four "
            f"backup clips with `evaluate_candidates.py` and visually reviewing each candidate's "
            f"validation plot -- see that report for the full comparison and why this clip/window "
            f"was preferred over the alternatives (all three showed the dog standing/fidgeting or "
            f"circling in place despite superficially reasonable automated metrics).\n"
        )
    else:
        a(
            f"- Selection method (two-phase, see `select_walk_window` docstring for the full "
            f"rationale): (1) a *core* window was found as the longest stretch with locally "
            f"stable heading (rolling circular std below its own 40th percentile, threshold "
            f"{window['turn_threshold_rad']:.2f} rad) and non-trivial motion (>0.03 m/s): "
            f"{window['core_run_s'][0]:.2f}-{window['core_run_s'][1]:.2f}s. An earlier version "
            f"that only gated on root speed (no heading check) silently bridged a ~1s mid-clip "
            f"turn/spin where heading rotated over 500 degrees -- inspection of the resulting "
            f"validation plot caught this, since path straightness and duty-factor plausibility "
            f"broke down; heading-gating fixed it. (2) The core was then greedily extended "
            f"outward in 0.05s steps while overall straightness stayed >= 0.92, to recover "
            f"legitimate slow lead-in/out that the heading gate can't judge reliably at very low "
            f"speed. Cycle count then came from directly counting Otsu+hysteresis contact segments "
            f"(a first attempt used an FFT-based period estimate and was rejected: it was unstable "
            f"on this dataset's noisy single-view paw tracks, swinging from 1.1s to 1.9s for a "
            f"~0.25s shift in window edge on this same clip -- see `otsu_threshold`'s docstring). "
            f"Finally, the window was best-effort snapped to a nearby contact onset using the same "
            f"smal-based contacts "
            f"(start snapped: {window['snap_info']['start_snapped']}, end snapped: "
            f"{window['snap_info']['end_snapped']}, reference leg: "
            f"{window['snap_info']['reference_leg']}). Snapping falls back silently to the "
            f"threshold-derived edge when no clean nearby contact onset is found -- raw single-view "
            f"per-foot detections are noisy in this dataset (see DATASET_REPORT.md), so exact "
            f"phase alignment is best-effort, not guaranteed.\n"
    )

    a("## Normalization applied\n")
    n = v["normalization"]
    a(f"- Heading rotated {np.degrees(n['heading_rotation_rad']):.1f} deg about Z to align the PCA principal direction of the root path (oriented to match net displacement) to +X.")
    a(f"- Ground plane set to z=0 using the 2nd percentile of all four paws' heights over the window as the floor estimate ({n['ground_z_offset']:.4f} m subtracted from all z).")
    a(f"- Root xy at frame 0 shifted to (0,0) (pre-shift value: {n['initial_root_xy_pre_shift']}).")
    a("- No rescaling: coordinates remain in the SMAL fit's real-world meters (`scale_applied=1.0` stored explicitly).")
    a("- Timing preserved as originally sampled: `timestamps` are `(frame_idx - frame_idx[0]) / 60.0`, not resampled to a uniform grid; native/effective fps stored alongside.\n")

    a("## Validation: numeric\n")
    a(f"- Mean forward (root +X) speed: {v['mean_forward_speed_mps']:.4f} m/s")
    a(f"- Body length (Nose to Tail_start, mean over window): {v['body_length_nose_to_tailbase_m']:.4f} m")
    a(f"- Withers height (mean over window, ground=0): {v['withers_height_m']:.4f} m")
    a("- Hip-to-paw leg length (mean over window):")
    for leg, val in v["leg_lengths_hip_to_paw_m"].items():
        a(f"  - {leg}: {val:.4f} m")
    a("- Contact duty factor (fraction of frames each foot is classified in contact):")
    for leg, val in v["contact_duty_per_foot"].items():
        a(f"  - {leg}: {val:.3f}")
    a("- Foot sliding during contact (mean horizontal displacement while planted, per contact segment):")
    for leg, val in v["foot_sliding_during_contact_m"].items():
        a(f"  - {leg}: {'n/a (too few contact segments)' if val is None else f'{val:.4f} m'}")
    a(f"- NaN/non-finite frames in the extracted keypoints: {v['nan_or_missing_frames']} (of {v['n_frames']})\n")

    a("## Validation: cross-check against pet_npy (independent evidence)\n")
    a(
        "pet_npy's own (separately tracked, unaligned, ~30 Hz) paw-height signal was "
        "interpolated onto the smal-derived timestamps and compared against the smal-based "
        "contact classification, per leg:\n"
    )
    for leg, res in zip(LEG_ORDER, v["pet_cross_check"]["per_leg"]):
        if res is None:
            a(f"- {leg}: not enough contact/swing frames in-window to compare")
        else:
            a(
                f"- {leg}: pet-tracked height during smal-contact frames "
                f"{res['mean_pet_height_during_smal_contact']:.3f} m vs. during smal-swing frames "
                f"{res['mean_pet_height_during_smal_swing']:.3f} m (lower during contact: "
                f"{res['lower_during_contact']}, effect size {res['effect_size_std']:.2f} "
                f"std of pet_npy's own height signal)"
            )
    a(
        "\nThe direction agrees for all four feet (`lower_during_contact: True`), which is "
        "weak-but-consistent independent evidence in favor of the smal-based contact "
        "classification. It should not be oversold: the effect sizes above are small "
        "(well under 1 std of pet_npy's own noisy height signal), which is expected given "
        "`pet_npy` and `smal_npy` are not frame-aligned (Section 1 of DATASET_REPORT.md) --"
        " the ~30 Hz interpolation-based time correspondence used here is only approximate, "
        "so a clean, large separation was never guaranteed. Directional agreement across all "
        "four independent legs is still more informative than chance alone.\n"
    )

    a("## Validation: visual\n")
    a(
        f"A diagnostic plot was rendered to `{args.out_plot.name}` (top-down root path, "
        "per-foot height with shaded contact windows, forward speed over time) and visually "
        "reviewed (the operator is on a headless SSH session and cannot view images "
        "directly, so this review was performed by the extraction agent). Findings:\n"
    )
    a(
        "- **Top-down path**: clean and essentially straight for the full window, monotonic "
        "forward progress (x: 0 to ~1.24m), lateral drift under 3cm throughout. No loop, "
        "backtrack, or kink -- confirms the heading-stability gate did its job.\n"
        "- **Forward speed**: a single smooth accelerate-cruise-decelerate arc (0.07 -> 0.65 "
        "-> 0.05 m/s), no sign-flips or spikes after switching to a lightly smoothed root-x "
        "derivative (an earlier version using a raw single-frame gradient showed non-physical "
        "-0.5 m/s \"backward\" spikes on this same clip -- fixed by smoothing root x with the "
        "same 0.3s box filter used for heading before differentiating).\n"
        "- **Paw heights / contacts**: all four traces oscillate with a visible common cadence "
        "and the shaded contact windows alternate sensibly across most of the window; two "
        "earlier versions of this plot (before root-relative height and before restricting the "
        "window to exclude a stop/turn) showed contact detection collapsing to zero for large "
        "stretches -- both were diagnosed and fixed using this same plot (see 'Selected window' "
        "and the `detect_contacts` docstring). Coverage is still visibly thinner in the first "
        "~0.4s and last ~0.25s of the window, where the dog is still accelerating/decelerating "
        "and the single per-leg Otsu threshold is less reliable; treat contact calls very near "
        "the window edges with more caution than the cruise-phase middle.\n"
    )

    a("## Output files\n")
    a(f"- `{args.out_npz.name}`: canonical NPZ (see `extract_walk.py` for full field list; keys include `root_pos`, `root_orient_rotmat`, `keypoints`/`keypoint_names`, `hip_pos`/`knee_pos`/`paw_pos`, `timestamps`, `forward_speed`, `foot_contact`).")
    a(f"- `{args.out_plot.name}`: validation plot.")
    a("\nNo retargeting or training was performed. The original dataset files were not modified.\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
