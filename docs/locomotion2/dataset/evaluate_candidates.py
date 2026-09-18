#!/usr/bin/env python3
"""Evaluate multiple InterPet4D clips for the best Locomotion 2 source walk.

Reuses extract_walk.py's keypoint identification, contact detection, motion
normalization, and plotting. This script's job is choosing a window: an
initial version reused extract_walk.py's core-detection + straightness-
extension heuristic and it silently picked short running bursts (straight
paths at ~1 m/s, not walks) or bridged stand->sprint->stop patterns that
happen to look "straight" in net displacement. Replaced with a direct
brute-force scan over (start, length) that checks stop/sprint/duty sanity
using *smoothed* speed (raw per-step speed is too noisy: an earlier attempt
using it found essentially zero candidates in any of the four clips, because
a single noisy frame reading near-zero happens by chance in almost any
multi-second window).

**Headline finding, not hidden**: none of the four candidate clips contain a
window that satisfies straightness > 0.9 together with >= 4 gait cycles,
no stop, and plausible contact duty factors simultaneously. Exhaustive
scanning (per-clip, thousands of candidate windows) found some windows with
straightness ~0.9 that turn out to be stand-then-sprint-then-stop patterns
(deceptively "straight" in net-displacement terms, but not walking), and some
windows with clean, consistent gait that are not going in a straight line
(the dog is walking steadily but wandering/circling). This script picks, per
clip, the best available trade-off: among windows that pass hard sanity gates
(no full stop, plausible duty, >=3 gait cycles, walking-pace speed), the one
maximizing straightness. Durations and exact numbers are reported honestly
per clip below and in the printed/JSON output -- none is a full pass of the
original target criteria.

Does not retarget or train anything, and does not write a canonical NPZ for
any candidate.

Usage:
    /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python evaluate_candidates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_walk import (  # noqa: E402
    KP24_HIP_IDXS,
    KP24_NAMES,
    KP24_PAW_IDXS,
    LEG_ORDER,
    count_contact_segments,
    detect_contacts,
    load_clip,
    make_plot,
    normalize_motion,
    smooth,
    verify_kp24_against_data,
    verify_pet20_against_data,
)
from inspect_interpet4d import (  # noqa: E402
    NATIVE_VIDEO_FPS,
    dog_size_ranking,
    load_norm_stats,
    parse_clip_id,
)

CANDIDATES_DIR = Path(__file__).resolve().parents[1] / "motions" / "candidates"

CANDIDATE_CLIPS = [
    "interpet_dog01_p01_take10_ego_001",
    "interpet_dog02_p05_take06_ego_001",
    "interpet_dog06_p14_take02_ego_001",
    "interpet_dog10_p09_take01_ego_001",
]

# Bingo's real, documented morphology (docs/Bingo_Robot_Spec_for_Animation.md,
# "read directly from the URDF and verified by forward kinematics"):
#   - standing (functional-neutral) height, paw to base: 0.1989 m
#   - total leg length (thigh + shank) at full extension: 0.2273 m
#   - hip-to-hip (front SP_J x=+39.9mm to rear SP_J x=-38.4mm) trunk span: ~0.078 m
#   - mass: 2.478 kg
# Bingo is a small, compact, unusually leggy-for-its-trunk quadruped (leg
# length roughly 3x its hip-to-hip trunk span) -- real dogs are not built
# like this, so morphology similarity is necessarily an imperfect, lowest-
# priority signal (as instructed), used only as a tiebreaker via relative
# size (smaller dog -> closer to Bingo's 2.5 kg, ~0.2 m frame).
BINGO_SHOULDER_HEIGHT_M = 0.1989
BINGO_LEG_LENGTH_M = 0.2273

HARD_MIN_CYCLES = 3.0          # relaxed from the requested 4.0 -- see module docstring
NO_STOP_FRAC = 0.30            # smoothed-speed minimum must exceed this fraction of the window mean
WALK_PACE_MAX_MPS = 0.5        # exclude running bursts
DUTY_LO, DUTY_HI = 0.15, 0.85  # exclude near-degenerate contact classification (a leg barely ever "in contact" is a red flag, not a real gait)


def scan_windows(clip_id: str, lengths=np.arange(2.5, 6.5, 0.25), start_step: float = 0.15) -> list[dict]:
    """Brute-force scan of every (start, length) window, returning full
    metrics for every candidate that passes the hard sanity gates (no stop,
    plausible duty, minimum cycle count, walking pace)."""
    _, smal = load_clip(clip_id)
    fidx = smal["frame_idx"].astype(np.int64)
    t_world = smal["t_world"].astype(np.float64)
    kp_world = smal["kp_world"].astype(np.float64)
    smal_times = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
    xy = t_world[:, :2]
    dur = smal_times[-1]

    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(fidx) / NATIVE_VIDEO_FPS, 1e-6)
    speed_times = (smal_times[:-1] + smal_times[1:]) / 2.0
    sm_speed_full = np.interp(smal_times, speed_times, smooth(step_speed, speed_times, 0.6))

    results = []
    for wlen in lengths:
        for t0 in np.arange(0, dur - wlen, start_step):
            t1 = t0 + wlen
            m = (smal_times >= t0) & (smal_times <= t1)
            if m.sum() < 12:
                continue
            pts = xy[m]
            if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(kp_world[m])):
                continue
            disp = np.linalg.norm(pts[-1] - pts[0])
            path = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            straightness = disp / max(path, 1e-9)
            sm = sm_speed_full[m]
            speed_mean, speed_std = float(sm.mean()), float(sm.std())
            speed_cv = speed_std / max(speed_mean, 1e-6)
            no_stop = sm.min() > NO_STOP_FRAC * speed_mean
            walk_pace = speed_mean < WALK_PACE_MAX_MPS
            paw_pos = kp_world[m][:, KP24_PAW_IDXS, :]
            c_info = detect_contacts(paw_pos, smal_times[m], t_world[m, 2])
            counts = [count_contact_segments(c_info["contacts"][:, k]) for k in range(4)]
            n_cycles = float(np.median(counts))
            duty = c_info["contacts"].mean(axis=0)
            duty_ok = bool(duty.min() > DUTY_LO and duty.max() < DUTY_HI)
            if n_cycles >= HARD_MIN_CYCLES and no_stop and duty_ok and walk_pace:
                results.append({
                    "t0": float(t0), "t1": float(t1), "duration_s": float(wlen),
                    "straightness": float(straightness), "n_cycles": n_cycles,
                    "per_leg_counts": counts,
                    "count_consistency": float(1.0 - np.std(counts) / max(np.mean(counts), 1e-6)),
                    "speed_mean": speed_mean, "speed_std": speed_std, "speed_cv": speed_cv,
                    "duty": duty.tolist(),
                })
    return results


def pick_best_window(candidates: list[dict]) -> dict:
    """Among sanity-gated candidates, maximize straightness *and* per-leg
    footfall-count consistency together -- an earlier version scored
    straightness alone and picked windows like counts=[5,1,7,2] (one leg
    barely contacting at all while straightness happened to be high), which
    is not "reliable alternating contacts" by any reading. Duration is the
    tiebreaker among near-ties on the combined score."""
    if not candidates:
        raise RuntimeError("no candidate window passed the hard sanity gates")
    combined = [0.6 * c["straightness"] + 0.4 * c["count_consistency"] for c in candidates]
    best = max(combined)
    near_best = [c for c, s in zip(candidates, combined) if s >= best - 0.03]
    return max(near_best, key=lambda c: c["duration_s"])


def evaluate_clip(clip_id: str) -> dict:
    pet, smal = load_clip(clip_id)
    meta = parse_clip_id(clip_id)

    kp24_check = verify_kp24_against_data(smal["kp_world"].astype(np.float64), smal["t_world"].astype(np.float64))
    pet20_check = verify_pet20_against_data(pet)

    candidates = scan_windows(clip_id)
    window = pick_best_window(candidates)
    window["n_candidates_found"] = len(candidates)
    window["fully_qualified"] = window["straightness"] > 0.9 and window["n_cycles"] >= 4.0

    fidx = smal["frame_idx"].astype(np.int64)
    t_world = smal["t_world"].astype(np.float64)
    kp_world = smal["kp_world"].astype(np.float64)
    kp_weight = smal["kp_weight"].astype(np.float64)
    smal_times = (fidx - fidx[0]) / NATIVE_VIDEO_FPS
    m = (smal_times >= window["t0"]) & (smal_times <= window["t1"])

    fidx_w = fidx[m]
    timestamps = (fidx_w - fidx_w[0]) / NATIVE_VIDEO_FPS
    root_pos, root_orient, kp, norm_info = normalize_motion(t_world[m], smal["R_world"][m].astype(np.float64), kp_world[m])
    root_x_smooth = smooth(root_pos[:, 0], timestamps, 0.3)
    forward_speed = np.gradient(root_x_smooth) / np.gradient(timestamps)
    paw_pos = kp[:, KP24_PAW_IDXS, :]
    contact_info = detect_contacts(paw_pos, timestamps, root_pos[:, 2])

    gaps = np.diff(fidx_w)
    dropped_ratio = float(np.mean(gaps > np.median(gaps))) if len(gaps) else 0.0
    mean_kp_weight = float(kp_weight[m].mean())
    mean_pet_conf = float(pet[..., 3].mean())

    body_length = float(np.linalg.norm(kp[:, KP24_NAMES.index("Nose")] - kp[:, KP24_NAMES.index("Tail_start")], axis=1).mean())
    withers_height = float(kp[:, KP24_NAMES.index("Withers"), 2].mean())
    leg_lengths = {LEG_ORDER[i]: float(np.linalg.norm(kp[:, KP24_HIP_IDXS[i]] - kp[:, KP24_PAW_IDXS[i]], axis=1).mean()) for i in range(4)}
    mean_leg_length = float(np.mean(list(leg_lengths.values())))

    sliding = {}
    for i, leg_name in enumerate(LEG_ORDER):
        in_c = contact_info["contacts"][:, i]
        segs, cur = [], []
        for t in range(len(in_c)):
            if in_c[t]:
                cur.append(t)
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        slides = [float(np.linalg.norm(paw_pos[s[-1], i, :2] - paw_pos[s[0], i, :2])) for s in segs if len(s) > 1]
        sliding[leg_name] = float(np.mean(slides)) if slides else None

    plot_path = CANDIDATES_DIR / f"{clip_id}_candidate.png"
    make_plot(plot_path, timestamps, root_pos, paw_pos, contact_info["contacts"], forward_speed)

    return {
        "clip_id": clip_id, "dog": meta["dog"], "window": window,
        "kp24_check": kp24_check, "pet20_check": pet20_check,
        "dropped_frame_ratio": dropped_ratio, "mean_kp_weight": mean_kp_weight, "mean_pet_confidence": mean_pet_conf,
        "body_length_m": body_length, "withers_height_m": withers_height,
        "leg_lengths_m": leg_lengths, "mean_leg_length_m": mean_leg_length,
        "foot_sliding_m": sliding, "plot_path": str(plot_path),
        "mean_forward_speed_mps": float(np.nanmean(forward_speed)),
        "n_frames": int(m.sum()), "nan_ok": bool(np.all(np.isfinite(kp))),
    }


def contact_quality_score(res: dict) -> float:
    w = res["window"]
    duty = np.array(w["duty"])
    duty_plausibility = float(np.mean([1.0 - min(abs(d - 0.5) / 0.5, 1.0) for d in duty]))
    slides = [v for v in res["foot_sliding_m"].values() if v is not None]
    slide_score = 1.0 - min((np.mean(slides) if slides else 0.05) / 0.06, 1.0)
    return float(0.45 * w["count_consistency"] + 0.35 * duty_plausibility + 0.20 * slide_score)


def tracking_quality_score(res: dict) -> float:
    drop_score = 1.0 - min(res["dropped_frame_ratio"] / 0.3, 1.0)
    weight_score = min(res["mean_kp_weight"] / 0.6, 1.0)
    conf_score = min(res["mean_pet_confidence"] / 0.6, 1.0)
    return float(0.4 * drop_score + 0.35 * weight_score + 0.25 * conf_score)


def morphology_score(res: dict, size_rank: dict, n_dogs: int) -> float:
    rank = size_rank.get(res["dog"], n_dogs // 2)
    size_score = 1.0 - (rank / max(n_dogs - 1, 1))  # smaller dog -> closer to tiny Bingo
    leg_ratio = res["mean_leg_length_m"] / max(res["withers_height_m"], 1e-6)
    bingo_ratio = BINGO_LEG_LENGTH_M / BINGO_SHOULDER_HEIGHT_M
    ratio_score = 1.0 - min(abs(leg_ratio - bingo_ratio) / bingo_ratio, 1.0)
    return float(0.7 * size_score + 0.3 * ratio_score)


def rank_candidates(results: list[dict], size_rank: dict, n_dogs: int) -> list[dict]:
    durations = np.array([r["window"]["duration_s"] for r in results])
    cvs = np.array([r["window"]["speed_cv"] for r in results])
    cycles = np.array([r["window"]["n_cycles"] for r in results])
    straightness = np.array([r["window"]["straightness"] for r in results])
    contact_q = np.array([contact_quality_score(r) for r in results])
    track_q = np.array([tracking_quality_score(r) for r in results])
    morph_q = np.array([morphology_score(r, size_rank, n_dogs) for r in results])

    def norm(x, invert=False):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.ones_like(x)
        n = (x - lo) / (hi - lo)
        return 1 - n if invert else n

    # Priority order from the task: steady duration, speed CV, gait-cycle
    # count, contact quality, straightness, tracking quality, morphology.
    weights = [0.28, 0.20, 0.18, 0.14, 0.10, 0.06, 0.04]
    score = (
        weights[0] * norm(durations)
        + weights[1] * norm(cvs, invert=True)
        + weights[2] * norm(cycles)
        + weights[3] * norm(contact_q)
        + weights[4] * norm(straightness)
        + weights[5] * norm(track_q)
        + weights[6] * norm(morph_q)
    )
    for i, r in enumerate(results):
        r["scores"] = {
            "duration_s": float(durations[i]), "speed_cv": float(cvs[i]), "n_cycles": float(cycles[i]),
            "straightness": float(straightness[i]), "contact_quality": float(contact_q[i]),
            "tracking_quality": float(track_q[i]), "morphology_similarity": float(morph_q[i]),
            "composite": float(score[i]),
        }
    return sorted(results, key=lambda r: -r["scores"]["composite"])


def main():
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    norm_stats = load_norm_stats()
    size_ranking = dog_size_ranking(norm_stats)
    size_rank = {dog: i for i, (dog, _) in enumerate(size_ranking)}
    n_dogs = len(size_ranking)

    results = []
    for clip_id in CANDIDATE_CLIPS:
        print(f"Evaluating {clip_id} ...")
        res = evaluate_clip(clip_id)
        results.append(res)
        w = res["window"]
        print(
            f"  best window=[{w['t0']:.2f},{w['t1']:.2f}] dur={w['duration_s']:.2f}s "
            f"cycles={w['n_cycles']:.1f} {w['per_leg_counts']} straightness={w['straightness']:.3f} "
            f"speed={w['speed_mean']:.3f}+-{w['speed_std']:.3f} cv={w['speed_cv']:.3f} "
            f"duty={np.round(w['duty'],2)} (of {w['n_candidates_found']} sanity-gated candidates; "
            f"fully_qualified={w['fully_qualified']})"
        )

    ranked = rank_candidates(results, size_rank, n_dogs)

    with open(CANDIDATES_DIR / "candidate_metrics.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)

    print("\n=== Final ranking (best first) ===")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['clip_id']}  composite={r['scores']['composite']:.3f}")


if __name__ == "__main__":
    main()
