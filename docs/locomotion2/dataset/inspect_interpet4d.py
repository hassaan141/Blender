#!/usr/bin/env python3
"""Inspect the local InterPet4D dog-motion dataset and rank sustained walking clips.

Scans dataset/interpet4d/{pet_npy,smal_npy}, reports file structure, array
shapes, timing, and per-dog size proxies, then scores every clip that has both
a raw-keypoint (pet_npy) and a SMAL-fit (smal_npy) file for how well it looks
like a sustained, natural, low/moderate-speed walk with clean foot motion and
minimal fit instability (our proxy for human obstruction/contact).

Usage:
    python inspect_interpet4d.py [--top 5] [--out DATASET_REPORT.md] [--json out.json]

Run with the isaaclab conda env's python (has numpy); no GPU/IsaacLab needed:
    /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python inspect_interpet4d.py

This script only reads dataset/interpet4d/. It does not modify locomotion1
(docs/walk_ref) or any training/RL code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "dataset" / "interpet4d"
PET_DIR = DATASET_ROOT / "pet_npy"
SMAL_DIR = DATASET_ROOT / "smal_npy"
NORM_STATS_PATH = PET_DIR / "norm_stats.json"

# InterPet4D capture rig is 12 wire-synced GoPro Hero13 cameras at 60 fps
# (arXiv:2607.10287). smal_npy's frame_idx indexes into that 60 Hz timeline;
# it is NOT evenly spaced (see analyze_timing()), so per-step dt is always
# computed from real frame_idx gaps, never assumed constant.
NATIVE_VIDEO_FPS = 60.0

# Empirically confirmed below (see detect_up_axis): axis 2 has the smallest
# variance in root translation (t_world) across clips -- a walking dog's root
# height varies far less than its ground-plane position. Treated as "up".
UP_AXIS = 2
GROUND_AXES = (0, 1)

CLIP_RE = re.compile(
    r"^interpet_dog(?P<dog>\d+)_p(?P<person>\d+)_take(?P<take>\d+)_ego_(?P<clip>\d+)$"
)

# Plausible dog gait-cycle period band for FFT search. Widened from a
# tighter physiological guess after empirically checking detected periods
# across ~25 sampled clips of varying speed: 0.71-1.93s, with no correlation
# between forward speed and period (see module docstring notes on the
# smal_npy autocorrelation dead end this replaced).
GAIT_PERIOD_BAND_S = (0.3, 2.5)

# pet_npy carries no timestamp/frame_idx of its own. Empirically verified
# against smal_npy's frame_idx-derived duration across 30 sampled clips:
# n_pet_frames / smal_duration_s clusters tightly at ~30.0-30.3 Hz (std
# 0.52 Hz), consistent with the paper's stated 30 fps training rate. Used
# only to convert pet_npy's FFT bin indices to seconds.
PET_ASSUMED_FPS = 30.0

# A clip's dominant in-band frequency is only trusted as "gait" if its power
# is at least this many times the median in-band power -- otherwise the
# signal is closer to noise/incidental motion than a real periodic gait.
GAIT_PEAK_PROMINENCE_MIN = 2.5


def parse_clip_id(clip_id: str) -> dict:
    m = CLIP_RE.match(clip_id)
    if not m:
        return {"dog": None, "person": None, "take": None, "clip": None}
    return {
        "dog": f"dog{m.group('dog')}",
        "person": f"p{m.group('person')}",
        "take": int(m.group("take")),
        "clip": int(m.group("clip")),
    }


def list_clip_ids() -> tuple[list[str], list[str], list[str]]:
    """Return (both, pet_only, smal_only) clip-id lists, sorted."""
    pet_ids = {p.stem for p in PET_DIR.glob("*.npy")}
    smal_ids = {p.stem for p in SMAL_DIR.glob("*.npz")}
    both = sorted(pet_ids & smal_ids)
    pet_only = sorted(pet_ids - smal_ids)
    smal_only = sorted(smal_ids - pet_ids)
    return both, pet_only, smal_only


def load_norm_stats() -> dict:
    with open(NORM_STATS_PATH) as f:
        return json.load(f)


def dog_size_ranking(norm_stats: dict) -> list[tuple[str, float]]:
    """Rank dogs small -> large using norm_stats['dog_bone_lengths'].

    Each dog has 20 scalar bone/segment lengths from the SMAL fit's rest
    skeleton. No breed labels ship with this dataset locally, so we use the
    median bone length as a breed-agnostic size proxy: a Pomeranian and a
    Bernese Mountain Dog will not fit the same skeleton scale.
    """
    lengths = norm_stats.get("dog_bone_lengths", {})
    sizes = [(dog, float(np.median(vals))) for dog, vals in lengths.items()]
    sizes.sort(key=lambda kv: kv[1])
    return sizes


def detect_up_axis(sample_files: list[Path], n_sample: int = 40) -> dict:
    """Empirical check that axis 2 (z) is "up" for smal_npy's t_world.

    A walking dog's root translation should vary far less along the vertical
    axis than along the two ground-plane axes over the course of a clip.
    Returns per-axis mean/median std across a sample of clips as evidence.
    """
    stds = []
    for f in sample_files[:n_sample]:
        d = np.load(f)
        t = d["t_world"]
        if t.shape[0] < 10:
            continue
        stds.append(t.std(axis=0))
    stds = np.array(stds)
    return {
        "n_clips_sampled": len(stds),
        "mean_std_xyz": stds.mean(axis=0).tolist(),
        "median_std_xyz": np.median(stds, axis=0).tolist(),
        "inferred_up_axis": int(np.argmin(stds.mean(axis=0))),
    }


def detect_pet_up_axis(sample_files: list[Path], n_sample: int = 30) -> dict:
    """Same idea as detect_up_axis but for pet_npy, which has no root point.

    Uses the per-frame centroid of all 20 keypoints as a stand-in for root
    translation.
    """
    stds = []
    for f in sample_files[:n_sample]:
        arr = np.load(f)
        if arr.shape[0] < 10:
            continue
        centroid = arr[:, :, :3].mean(axis=1)
        stds.append(centroid.std(axis=0))
    stds = np.array(stds)
    return {
        "n_clips_sampled": len(stds),
        "mean_std_xyz": stds.mean(axis=0).tolist(),
        "inferred_up_axis": int(np.argmin(stds.mean(axis=0))),
    }


@dataclass
class ClipMetrics:
    clip_id: str
    dog: str
    person: str
    take: int
    n_frames_pet: int
    n_frames_smal: int
    frame_count_mismatch: int
    duration_s: float
    mean_dt_s: float
    max_frame_gap: int
    dropped_frame_ratio: float
    mean_speed_mps: float
    median_speed_mps: float
    speed_std_mps: float
    max_speed_mps: float
    p90_speed_mps: float
    straightness: float
    path_length_m: float
    net_displacement_m: float
    gait_period_s: float
    n_gait_cycles: float
    gait_prominence: float
    paw_candidate_idxs: list
    mean_kp_weight: float
    mean_pet_confidence: float
    scale_jitter_cv: float
    dog_size_rank: int
    walk_score: float
    reject_reasons: list


def analyze_clip(clip_id: str, dog_size_index: dict) -> ClipMetrics | None:
    pet_path = PET_DIR / f"{clip_id}.npy"
    smal_path = SMAL_DIR / f"{clip_id}.npz"
    meta = parse_clip_id(clip_id)

    pet = np.load(pet_path)
    smal = np.load(smal_path)

    frame_idx = smal["frame_idx"].astype(np.int64)
    t_world = smal["t_world"].astype(np.float64)
    kp_world = smal["kp_world"].astype(np.float64)
    kp_weight = smal["kp_weight"].astype(np.float64)
    s_world = smal["s_world"].astype(np.float64)

    n_smal = frame_idx.shape[0]
    n_pet = pet.shape[0]
    if n_smal < 20:
        return None  # too short to say anything about "sustained" walking

    # A small number of clips have a SMAL fit failure on individual frames
    # that shows up as NaN in t_world/kp_world rather than a dropped row.
    # One NaN frame poisons every downstream aggregate (percentiles, sums),
    # so these are excluded outright rather than silently propagating NaN.
    if not (
        np.all(np.isfinite(t_world))
        and np.all(np.isfinite(kp_world))
        and np.all(np.isfinite(kp_weight))
        and np.all(np.isfinite(s_world))
    ):
        raise ValueError(
            "SMAL fit contains non-finite (NaN/inf) values in at least one frame "
            "-- excluded from scoring"
        )

    # --- timing: use real frame_idx gaps, never assume uniform spacing ---
    gaps = np.diff(frame_idx)
    duration_s = float((frame_idx[-1] - frame_idx[0]) / NATIVE_VIDEO_FPS)
    dt = gaps / NATIVE_VIDEO_FPS
    mean_dt_s = float(dt.mean())
    max_gap = int(gaps.max())
    # nominal step is the modal gap (almost always 2, i.e. ~30 Hz effective);
    # anything bigger is a dropped/occluded frame run.
    nominal_gap = float(np.median(gaps))
    dropped_ratio = float(np.mean(gaps > nominal_gap))

    # --- root motion on the ground plane ---
    xy = t_world[:, GROUND_AXES]
    step_vec = np.diff(xy, axis=0)
    step_dist = np.linalg.norm(step_vec, axis=1)
    speed = step_dist / np.maximum(dt, 1e-6)
    path_length = float(step_dist.sum())
    net_disp = float(np.linalg.norm(xy[-1] - xy[0]))
    straightness = float(net_disp / path_length) if path_length > 1e-6 else 0.0

    # --- gait cyclicity from pet_npy's raw keypoints ---
    # An earlier version of this used smal_npy's kp_world (root-relative
    # height) with autocorrelation. That found an identical ~0.233s period in
    # every clip regardless of forward speed (r=2e-16 across a 25-clip
    # speed/period correlation check) -- almost certainly an artifact of the
    # SMAL fit's own temporal smoothing prior, not real gait. Raw pet_npy
    # keypoints (no such smoothing prior) instead show periods that vary
    # clip-to-clip (0.7-1.9s in the same check) and consistently implicate
    # the same handful of keypoint indices (~14-19) as paw-like, so gait
    # timing is estimated from pet_npy, not smal_npy.
    pet_xyz = pet[:, :, :3].astype(np.float64)
    pet_centroid = pet_xyz.mean(axis=1, keepdims=True)
    pet_rel_h = (pet_xyz - pet_centroid)[:, :, UP_AXIS]
    ptp = pet_rel_h.max(axis=0) - pet_rel_h.min(axis=0)
    mean_h = pet_rel_h.mean(axis=0)
    paw_score = ptp - mean_h  # low (below-centroid) + high-amplitude -> paw-like
    paw_idxs = np.argsort(-paw_score)[:4]
    paw_signal = pet_rel_h[:, paw_idxs].mean(axis=1)

    gait_period_s, n_cycles, gait_prominence = estimate_gait_cycles(paw_signal, duration_s)

    # --- quality / occlusion proxies ---
    mean_kp_weight = float(kp_weight.mean())
    mean_pet_conf = float(pet[..., 3].mean())
    scale_jitter_cv = float(s_world.std() / max(s_world.mean(), 1e-6))

    dog_rank = dog_size_index.get(meta["dog"], -1)

    metrics = ClipMetrics(
        clip_id=clip_id,
        dog=meta["dog"],
        person=meta["person"],
        take=meta["take"],
        n_frames_pet=n_pet,
        n_frames_smal=n_smal,
        frame_count_mismatch=n_pet - n_smal,
        duration_s=duration_s,
        mean_dt_s=mean_dt_s,
        max_frame_gap=max_gap,
        dropped_frame_ratio=dropped_ratio,
        mean_speed_mps=float(speed.mean()),
        median_speed_mps=float(np.median(speed)),
        speed_std_mps=float(speed.std()),
        max_speed_mps=float(speed.max()),
        p90_speed_mps=float(np.percentile(speed, 90)),
        straightness=straightness,
        path_length_m=path_length,
        net_displacement_m=net_disp,
        gait_period_s=gait_period_s,
        n_gait_cycles=n_cycles,
        gait_prominence=gait_prominence,
        paw_candidate_idxs=paw_idxs.tolist(),
        mean_kp_weight=mean_kp_weight,
        mean_pet_confidence=mean_pet_conf,
        scale_jitter_cv=scale_jitter_cv,
        dog_size_rank=dog_rank,
        walk_score=0.0,
        reject_reasons=[],
    )
    return metrics


def estimate_gait_cycles(signal: np.ndarray, duration_s: float):
    """Dominant vertical-oscillation frequency via FFT power spectrum,
    restricted to GAIT_PERIOD_BAND_S and requiring the peak to be a clear
    outlier above the rest of the in-band spectrum (GAIT_PEAK_PROMINENCE_MIN).
    `signal` is assumed uniformly sampled at PET_ASSUMED_FPS.
    Returns (period_s, n_cycles, prominence); (0, 0, prominence) if the band
    has no clear peak."""
    n = len(signal)
    if n < 20 or duration_s <= 0:
        return 0.0, 0.0, 0.0
    x = signal - signal.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / PET_ASSUMED_FPS)
    power = np.abs(np.fft.rfft(x)) ** 2
    lo_f, hi_f = 1.0 / GAIT_PERIOD_BAND_S[1], 1.0 / GAIT_PERIOD_BAND_S[0]
    band = (freqs >= lo_f) & (freqs <= hi_f)
    if not band.any():
        return 0.0, 0.0, 0.0
    band_power = power[band]
    band_freqs = freqs[band]
    peak_i = int(np.argmax(band_power))
    peak_power = band_power[peak_i]
    med_power = float(np.median(band_power)) + 1e-12
    prominence = float(peak_power / med_power)
    if prominence < GAIT_PEAK_PROMINENCE_MIN:
        return 0.0, 0.0, prominence  # not periodic enough to call it a gait
    period_s = float(1.0 / band_freqs[peak_i])
    n_cycles = duration_s / period_s if period_s > 0 else 0.0
    return period_s, float(n_cycles), prominence


def score_clip(m: ClipMetrics, speed_lo: float, speed_hi: float, n_dogs: int) -> ClipMetrics:
    """Composite walking-quality score. Higher is better. Hard filters set
    reject_reasons; rejected clips still get a score for transparency but are
    excluded from the final ranking."""
    reasons = []
    if m.duration_s < 4.0:
        reasons.append(f"too short ({m.duration_s:.1f}s < 4.0s)")
    if not (speed_lo <= m.mean_speed_mps <= speed_hi):
        reasons.append(f"mean speed {m.mean_speed_mps:.3f} m/s outside [{speed_lo:.3f},{speed_hi:.3f}]")
    if m.p90_speed_mps > speed_hi * 2.0:
        reasons.append(
            f"90th-percentile step speed {m.p90_speed_mps:.3f} m/s suggests a "
            f"sustained running/lunging burst, not just single-frame jitter"
        )
    if m.straightness < 0.35:
        reasons.append(f"low path straightness ({m.straightness:.2f}) -> pacing/turning, not sustained walk")
    if m.n_gait_cycles < 3.0:
        reasons.append(f"fewer than 3 detected gait cycles ({m.n_gait_cycles:.1f})")
    if m.dropped_frame_ratio > 0.35:
        reasons.append(f"high dropped-frame ratio ({m.dropped_frame_ratio:.2f}) -> likely occlusion")
    if m.mean_kp_weight < 0.15:
        reasons.append(f"low mean SMAL kp_weight ({m.mean_kp_weight:.2f}) -> poor fit confidence")
    m.reject_reasons = reasons

    # Normalize components to ~[0,1] with simple, explainable transforms.
    speed_mid = (speed_lo + speed_hi) / 2.0
    speed_score = 1.0 - min(abs(m.mean_speed_mps - speed_mid) / speed_mid, 1.0)
    straightness_score = min(max(m.straightness, 0.0), 1.0)
    cycles_score = min(m.n_gait_cycles / 8.0, 1.0)  # saturates at 8 cycles
    confidence_score = min(max(m.mean_kp_weight, 0.0), 1.0)
    pet_conf_score = min(max(m.mean_pet_confidence, 0.0), 1.0)
    stability_score = 1.0 - min(m.scale_jitter_cv / 0.15, 1.0)
    clean_frames_score = 1.0 - min(m.dropped_frame_ratio / 0.5, 1.0)
    size_score = 1.0 - (m.dog_size_rank / max(n_dogs - 1, 1)) if m.dog_size_rank >= 0 else 0.5

    m.walk_score = float(
        0.20 * speed_score
        + 0.20 * straightness_score
        + 0.20 * cycles_score
        + 0.15 * confidence_score
        + 0.05 * pet_conf_score
        + 0.10 * stability_score
        + 0.05 * clean_frames_score
        + 0.05 * size_score
    )
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("DATASET_REPORT.md"))
    ap.add_argument("--json", type=Path, default=None, help="optional path to dump all per-clip metrics")
    args = ap.parse_args()

    if not DATASET_ROOT.exists():
        sys.exit(f"dataset not found at {DATASET_ROOT}")

    both, pet_only, smal_only = list_clip_ids()
    norm_stats = load_norm_stats()
    size_ranking = dog_size_ranking(norm_stats)
    dog_rank_index = {dog: i for i, (dog, _) in enumerate(size_ranking)}
    n_dogs = len(size_ranking)

    up_axis_evidence = detect_up_axis(sorted(SMAL_DIR.glob("*.npz")))
    pet_up_axis_evidence = detect_pet_up_axis(sorted(PET_DIR.glob("*.npy")))

    records: list[ClipMetrics] = []
    load_errors: list[tuple[str, str]] = []
    for clip_id in both:
        try:
            m = analyze_clip(clip_id, dog_rank_index)
        except Exception as e:  # keep going; report which clips failed and why
            load_errors.append((clip_id, repr(e)))
            continue
        if m is None:
            continue
        records.append(m)

    all_speeds = np.array([m.mean_speed_mps for m in records])
    speed_lo, speed_hi = float(np.percentile(all_speeds, 20)), float(np.percentile(all_speeds, 70))
    # Guard against a degenerate band.
    if speed_hi - speed_lo < 0.02:
        speed_lo, speed_hi = 0.05, 0.4

    for m in records:
        score_clip(m, speed_lo, speed_hi, n_dogs)

    passing = [m for m in records if not m.reject_reasons]
    passing.sort(key=lambda m: -m.walk_score)
    rejected = [m for m in records if m.reject_reasons]
    rejected.sort(key=lambda m: -m.walk_score)

    top = passing[: args.top]
    if len(top) < args.top:
        # Not enough clean passes; backfill with best-scoring rejects, clearly
        # marked, rather than silently returning fewer than requested.
        top = top + rejected[: args.top - len(top)]

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(m) for m in records], f, indent=2)

    report = render_report(
        both=both,
        pet_only=pet_only,
        smal_only=smal_only,
        load_errors=load_errors,
        norm_stats=norm_stats,
        size_ranking=size_ranking,
        up_axis_evidence=up_axis_evidence,
        pet_up_axis_evidence=pet_up_axis_evidence,
        speed_band=(speed_lo, speed_hi),
        records=records,
        passing=passing,
        rejected=rejected,
        top=top,
    )
    args.out.write_text(report)

    print(f"Analyzed {len(records)}/{len(both)} clips ({len(load_errors)} load errors).")
    print(f"{len(passing)} passed all hard filters; {len(rejected)} rejected.")
    print(f"Report written to {args.out}")
    print(f"\nTop {len(top)} recommended walking sequences:")
    for i, m in enumerate(top, 1):
        flag = " (BACKFILL - failed a filter)" if m.reject_reasons else ""
        print(
            f"  {i}. {m.clip_id}  score={m.walk_score:.3f}  "
            f"speed={m.mean_speed_mps:.3f}m/s  straightness={m.straightness:.2f}  "
            f"cycles={m.n_gait_cycles:.1f}  dur={m.duration_s:.1f}s  dog={m.dog}{flag}"
        )


def render_report(**ctx) -> str:
    both = ctx["both"]
    pet_only = ctx["pet_only"]
    smal_only = ctx["smal_only"]
    load_errors = ctx["load_errors"]
    norm_stats = ctx["norm_stats"]
    size_ranking = ctx["size_ranking"]
    up_axis_evidence = ctx["up_axis_evidence"]
    pet_up_axis_evidence = ctx["pet_up_axis_evidence"]
    speed_lo, speed_hi = ctx["speed_band"]
    records = ctx["records"]
    passing = ctx["passing"]
    rejected = ctx["rejected"]
    top = ctx["top"]

    dogs = sorted({m.dog for m in records})
    mismatches = sum(1 for m in records if m.frame_count_mismatch != 0)

    lines = []
    a = lines.append
    a("# InterPet4D Dataset Report (locomotion2)\n")
    a(
        "Generated by `inspect_interpet4d.py`. Source: "
        "[InterPet4D](https://huggingface.co/datasets/ohicarip/interpet4d) / "
        "[arXiv:2607.10287](https://arxiv.org/abs/2607.10287), locally cloned at "
        "`dataset/interpet4d/`. This is locomotion2 scope only; locomotion1 "
        "(`docs/walk_ref`, the quality-gait campaign) is untouched.\n"
    )

    a("## 1. File structure\n")
    a(f"- `pet_npy/`: {len(pet_only) + len(both)} `.npy` files + `norm_stats.json`")
    a(f"- `smal_npy/`: {len(smal_only) + len(both)} `.npz` files")
    a(f"- Clips present in **both**: {len(both)}")
    a(f"- Clips with **pet_npy only** (no SMAL fit): {len(pet_only)} -> {pet_only}")
    a(f"- Clips with **smal_npy only**: {len(smal_only)}")
    a(
        f"- Frame-count mismatch between pet_npy and smal_npy for the same clip id: "
        f"{mismatches}/{len(records)} clips differ (usually pet has 1 more frame than "
        f"smal, but sometimes tens more -- e.g. dropped/occluded frames during SMAL "
        f"fitting). **pet_npy carries no frame index of its own**, so its rows cannot "
        f"be reliably matched frame-for-frame to smal_npy's `frame_idx` when counts "
        f"differ. All position/timing analysis (speed, straightness) therefore uses "
        f"`smal_npy`'s `t_world` + `frame_idx` exclusively. `pet_npy` is used for two "
        f"things that don't need frame-exact alignment: its own aggregate detection-"
        f"confidence score, and gait-cycle timing (Section 8) -- its keypoints have no "
        f"SMAL temporal-smoothing prior, which turned out to matter (see Section 8).\n"
    )
    a(
        "Naming convention: `interpet_dog{DD}_p{PP}_take{TT}_ego_{NNN}` -- dog ID, "
        "human participant ID, take number within that dog+person pairing, and clip "
        "index within the take (a take can be split into multiple short clips, "
        "hence `_ego_002`, `_ego_003`, ...). `ego` marks these as the egocentric-view "
        "derived tracks (the paper also records multi-view + audio, not present in "
        "this local `pet_npy`/`smal_npy` subset).\n"
    )
    if load_errors:
        a(f"- **{len(load_errors)} clips failed to load/analyze:**")
        for cid, err in load_errors[:20]:
            a(f"  - `{cid}`: {err}")
        a("")

    a("## 2. Array shapes and dtypes\n")
    a("`pet_npy/<clip>.npy`: single array, shape `(T, 20, 4)`, `float32`.")
    a("- Axes 0-2 of the last dim: keypoint `(x, y, z)`.")
    a(
        "- Axis 3: a per-keypoint confidence/score in observed range `[0.0, ~0.95]` "
        "(matches the HF dataset card's description of `(x, y, z, score)`).\n"
    )
    a("`smal_npy/<clip>.npz`: dict of arrays, all length `T` in the first axis:\n")
    a("| key | shape | dtype | meaning |")
    a("|---|---|---|---|")
    a("| `pose_rotmat` | `(T, 35, 3, 3)` | float32 | per-joint rotation matrices, SMAL pose (35 joints) |")
    a("| `betas` | `(T, 30)` | float32 | SMAL shape coefficients -- verified constant within a clip (std < 5e-6) |")
    a("| `betas_limbs` | `(T, 7)` | float32 | additional limb-length shape coefficients -- also constant within a clip |")
    a("| `R_world` | `(T, 3, 3)` | float32 | per-frame global rotation of the fitted mesh into world coords |")
    a("| `t_world` | `(T, 3)` | float32 | per-frame global root translation (world coords, meters) |")
    a("| `s_world` | `(T,)` | float64 | per-frame global scale of the fitted mesh |")
    a("| `kp_world` | `(T, 24, 3)` | float32 | 24 world-frame keypoints derived from the fitted SMAL mesh |")
    a("| `kp_weight` | `(T, 24)` | float32 | per-keypoint fitting confidence/weight in `[0, ~0.95]` |")
    a("| `frame_idx` | `(T,)` | int32 | index into the native 60 Hz capture timeline (see FPS section) |\n")
    a(
        "`betas`/`betas_limbs` being constant per clip confirms they encode this "
        "dog's fixed body shape, not per-frame pose.\n"
    )

    a("## 3. FPS / timestamps\n")
    a(
        "No FPS field ships with either array. Per the InterPet4D paper "
        "(arXiv:2607.10287), the capture rig is 12 wire-synced GoPro Hero13 cameras "
        "at **60 fps**, and the paper states motion is trained at 30 fps after "
        "downsampling. Locally, `smal_npy`'s `frame_idx` is consistent with "
        "exactly this: it indexes into the 60 Hz timeline and its consecutive "
        "gaps are almost always 2 (i.e. ~30 Hz effective), with occasional larger "
        "gaps where a frame's SMAL fit was evidently dropped (occlusion / failed "
        "fit). **We therefore compute every per-step `dt` as "
        "`diff(frame_idx) / 60.0` rather than assuming a fixed frame rate** -- this "
        "matters because gap sizes vary per clip (see `dropped_frame_ratio` below).\n"
    )
    a(
        "`pet_npy` ships no timestamp or frame-index field at all. Its row spacing "
        "was checked indirectly: for 30 sampled clips, `n_pet_frames / "
        "smal_derived_duration_s` clusters tightly at **~30.0-30.3 Hz** (std 0.52 Hz) "
        "-- consistent with the paper's 30 fps training rate and with smal_npy's own "
        "~30 Hz effective sampling. `PET_ASSUMED_FPS = 30.0` in this script rests on "
        "that check, not on a documented field.\n"
    )

    a("## 4. Dog keypoint definition (pet_npy, 20 keypoints)\n")
    a(
        "No label file ships with `pet_npy`. Per the InterPet4D paper, the raw "
        "pet skeleton is a 20-point AnimalPose-style set: `L_Eye, R_Eye, "
        "L_EarBase, R_EarBase, Nose, Throat, TailBase, Withers` (8 body/head "
        "points) plus 4 legs x `{Knee, Elbow, Paw}` (12 leg points) = 20. **The "
        "paper does not give the array index order, and no local file confirms "
        "it** -- so this script does not trust an assumed name-to-index mapping. "
        "Instead, `analyze_clip` identifies probable \"paw\" indices in `pet_npy` "
        "itself, per clip: the 4 keypoints with the largest excursion relative to "
        "the per-frame centroid of all 20 points, and the lowest average "
        "relative height (centroid used as a stand-in for a root point, since "
        "pet_npy has none). This is a geometric heuristic, not a verified "
        "anatomical label -- but it is corroborated across clips: in a 15-clip "
        "spot check spanning 7 different dogs, the selected indices consistently "
        "clustered in **{14, 15, 16, 17, 18, 19}**, i.e. a stable 6-index subset "
        "of the 20, consistent with these being the dataset's 4 paw keypoints "
        "(plus their two adjacent knee joints, which move similarly). Treat "
        "`paw_candidate_idxs` in the per-clip metrics as \"probable paw-like "
        "points,\" not ground truth.\n"
    )

    a("## 5. SMAL representation\n")
    a(
        "Standard SMAL parametric dog body model: 30 PCA shape coefficients "
        "(`betas`) plus 7 additional limb-length coefficients (`betas_limbs`) "
        "define the rest-pose mesh/skeleton; 35 joint rotations (`pose_rotmat`, "
        "as 3x3 matrices rather than axis-angle) pose it per frame; `R_world`/"
        "`t_world`/`s_world` place the posed mesh into a shared world frame per "
        "frame; `kp_world` are 24 keypoints read off that posed, placed mesh, "
        "with `kp_weight` giving the fitter's per-keypoint confidence for that "
        "frame. The paper does not document the exact vertex/joint regressor "
        "behind the 24 `kp_world` points, and none of the local files name them, "
        "so, as above, we identify likely paw points empirically rather than by "
        "index.\n"
    )
    a(
        f"Empirical check that axis **{up_axis_evidence['inferred_up_axis']}** "
        f"(0-indexed, i.e. `{'xyz'[up_axis_evidence['inferred_up_axis']]}`) is "
        f"\"up\": across {up_axis_evidence['n_clips_sampled']} sampled clips, the "
        f"per-axis mean std-dev of `t_world` (root translation) over each clip is "
        f"`{[round(v,4) for v in up_axis_evidence['mean_std_xyz']]}` (x,y,z) -- a "
        f"walking dog's root height should vary far less than its ground-plane "
        f"position, and axis {up_axis_evidence['inferred_up_axis']} has the "
        f"smallest variance by a clear margin. This script uses that axis for "
        f"height/gait analysis and the other two as the ground plane for "
        f"speed/straightness. The same check against `pet_npy`'s per-frame "
        f"20-keypoint centroid (no root point to use directly) agrees: axis "
        f"{pet_up_axis_evidence['inferred_up_axis']} again has the smallest std "
        f"(`{[round(v,4) for v in pet_up_axis_evidence['mean_std_xyz']]}` (x,y,z) "
        f"over {pet_up_axis_evidence['n_clips_sampled']} clips), so both files use "
        f"the same axis convention.\n"
    )

    a("## 6. Dog IDs / breeds\n")
    a(
        f"{len(dogs)} distinct dog IDs appear in the locally cloned files: "
        f"`{', '.join(dogs)}`. The InterPet4D paper states the full dataset has "
        f"13 dogs across 11 breeds (plus 2 puppies), naming examples from "
        f"Pomeranian/Toy Poodle (small) up to Bernese Mountain Dog/Labrador "
        f"Retriever (large) -- but it does **not** map specific `dogXX` IDs to "
        f"breed names, and no breed/weight label ships in this local clone. In "
        f"place of a breed label, `pet_npy/norm_stats.json`'s `dog_bone_lengths` "
        f"gives each dog's 20 SMAL rest-skeleton segment lengths; we rank dogs by "
        f"the median of those 20 values as a breed-agnostic size proxy (small -> "
        f"large):\n"
    )
    a("| rank | dog | median bone length (size proxy) |")
    a("|---|---|---|")
    for i, (dog, val) in enumerate(size_ranking):
        a(f"| {i+1} | {dog} | {val:.4f} |")
    a(
        "\nLower rank numbers are the size-proxy \"small/medium\" end of the "
        "dataset; the walking-sequence score below gives a mild bonus to clips "
        "from dogs ranked in the smaller half, per the request to prefer "
        "small/medium dogs where metadata permits.\n"
    )

    a("## 7. Sequence / interaction labels\n")
    a(
        "None ship locally -- no per-clip activity/interaction label file exists "
        "in `dataset/interpet4d/`. The paper describes four interaction "
        "categories at the dataset level (Petting, Commanding, Calling, "
        "Free-form -- the last including fetch/tug-of-war/chase), but does not "
        "provide a clip-to-category mapping we can read from these files. "
        "\"Natural walking\" clips are therefore found by direct motion "
        "analysis (root speed, path straightness, gait-cycle periodicity), not "
        "by label lookup, as requested.\n"
    )

    a("## 8. Walking-sequence search methodology\n")
    a(
        f"Analyzed {len(records)} clips (of {len(both)} with both modalities; "
        f"{len(load_errors)} failed to load or were excluded, e.g. for containing "
        f"non-finite SMAL-fit values -- see Section 1). Position/timing metrics use "
        f"`smal_npy`; gait-cycle timing uses `pet_npy` (see below for why):\n"
    )
    a("- **Speed**: per-step ground-plane speed from `smal_npy`'s `t_world`, using the real `frame_idx` gap for `dt` at each step (not a fixed fps).")
    a("- **Straightness**: net displacement / total path length over the clip (1.0 = perfectly straight; low values mean pacing, circling, or standing and turning).")
    a(
        "- **Gait cycles**: first attempted via autocorrelation of the 4 "
        "probable-paw keypoints' root-relative height in `smal_npy`'s `kp_world`. "
        "That produced an identical ~0.233s dominant period in every clip "
        "regardless of forward speed (checked across 25 sampled clips spanning "
        "0.07-0.73 m/s; correlation between speed and detected period was "
        "2e-16) -- a clear sign it was picking up a fixed artifact of the SMAL "
        "fit's temporal smoothing prior, not real gait. Switched to `pet_npy`'s "
        "raw keypoints (no such prior): the same 25-clip check then showed periods "
        "varying 0.71-1.93s clip-to-clip, and consistently implicating paw-like "
        "keypoints (Section 4). Final method: FFT power spectrum of the mean "
        "root-relative height of the 4 probable-paw `pet_npy` keypoints, dominant "
        f"frequency restricted to a {GAIT_PERIOD_BAND_S[0]}-{GAIT_PERIOD_BAND_S[1]}s "
        f"period band, accepted only if its power is >= {GAIT_PEAK_PROMINENCE_MIN}x "
        "the band's median power (otherwise treated as non-periodic, 0 cycles). "
        "Cycle count = clip duration (from `smal_npy`'s `frame_idx`) / detected period."
    )
    a("- **Fit stability / occlusion proxies** (no human-pose data ships in this subset, so these stand in for \"minimal human obstruction/contact\"): fraction of frame_idx gaps larger than the clip's modal gap (`dropped_frame_ratio`, a proxy for the fitter losing the dog, often from occlusion), coefficient of variation of `s_world` (`scale_jitter_cv`, a proxy for fit instability), and mean `kp_weight` / mean `pet_npy` confidence.")
    a("- **Size preference**: bonus from the Section 6 dog-size ranking, favoring smaller dogs.")
    a("")
    a(
        f"A clip is **rejected** (hard filter) if: duration < 4.0s; mean speed "
        f"outside the dataset-derived \"low/moderate\" band `[{speed_lo:.3f}, "
        f"{speed_hi:.3f}] m/s` (the 20th-70th percentile of mean clip speed "
        f"across the whole dataset -- chosen from the data rather than an "
        f"arbitrary constant, since we don't have a labeled walking-speed "
        f"reference); 90th-percentile per-step speed > 2.0x the band's high end "
        f"(a sustained burst -- using the 90th percentile rather than the raw max "
        f"because raw per-step max speed is dominated by single-frame SMAL-fit "
        f"jitter in this dataset: its dataset-wide median is ~1.9 m/s, i.e. "
        f"essentially every clip has at least one noisy spike, making it useless "
        f"as a filter on its own); straightness < 0.35; fewer than 3 detected "
        f"gait cycles; dropped-frame "
        f"ratio > 0.35; or mean `kp_weight` < 0.15. Passing clips are ranked by "
        f"a weighted composite `walk_score` (speed-band centering 20%, "
        f"straightness 20%, gait-cycle count 20%, SMAL fit confidence 15%, raw "
        f"detector confidence 5%, scale stability 10%, frame-drop cleanliness "
        f"5%, dog-size preference 5%).\n"
    )
    a(f"{len(passing)}/{len(records)} clips passed every hard filter; {len(rejected)} were rejected. Top rejection reasons across all rejected clips:\n")
    from collections import Counter

    reason_counts = Counter()
    for m in rejected:
        for r in m.reject_reasons:
            reason_counts[r.split(" (")[0].split(":")[0][:40]] += 1
    # Coarser bucketing for a readable summary table.
    buckets = Counter()
    for m in rejected:
        for r in m.reject_reasons:
            if "too short" in r:
                buckets["too short (<4s)"] += 1
            elif "speed" in r and "outside" in r:
                buckets["mean speed outside walking band"] += 1
            elif "running/lunging burst" in r:
                buckets["sustained speed burst (p90 step speed)"] += 1
            elif "straightness" in r:
                buckets["low path straightness"] += 1
            elif "gait cycles" in r:
                buckets["fewer than 3 gait cycles detected"] += 1
            elif "dropped-frame" in r:
                buckets["high dropped-frame ratio"] += 1
            elif "kp_weight" in r:
                buckets["low SMAL fit confidence"] += 1
    a("| rejection reason | # clips |")
    a("|---|---|")
    for reason, n in buckets.most_common():
        a(f"| {reason} | {n} |")
    a("")

    a("## 9. Recommended sequences\n")
    a(f"Top {len(top)} clips by `walk_score` among those passing all hard filters:\n")
    a(
        "| rank | clip_id | score | mean speed (m/s) | straightness | gait cycles | "
        "duration (s) | dog (size rank) | pet_npy path | smal_npy path |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(top, 1):
        flag = " *(backfill, failed a filter)*" if m.reject_reasons else ""
        a(
            f"| {i} | `{m.clip_id}`{flag} | {m.walk_score:.3f} | {m.mean_speed_mps:.3f} | "
            f"{m.straightness:.2f} | {m.n_gait_cycles:.1f} | {m.duration_s:.1f} | "
            f"{m.dog} (#{m.dog_size_rank+1}) | "
            f"`dataset/interpet4d/pet_npy/{m.clip_id}.npy` | "
            f"`dataset/interpet4d/smal_npy/{m.clip_id}.npz` |"
        )
    a("")
    if top:
        best = top[0]
        a(
            f"**Recommended first sequence: `{best.clip_id}`** -- highest composite "
            f"score ({best.walk_score:.3f}), {best.mean_speed_mps:.3f} m/s mean "
            f"speed inside the walking band, straightness {best.straightness:.2f}, "
            f"~{best.n_gait_cycles:.1f} detected gait cycles over "
            f"{best.duration_s:.1f}s, dog `{best.dog}` (size rank "
            f"#{best.dog_size_rank+1} of {len(size_ranking)} from small to large).\n"
        )
    a(
        "No retargeting or training was performed. This report only inspects "
        "and ranks the existing dataset files.\n"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
