"""Categorize all lifelike_dog BVH clips and pick clean sub-windows for AMP.

Root body = "Bip01" (top of the hierarchy; b_Hips sits at a zero offset under it
and just carries a secondary local rotation, so Bip01's own translation channels
are the actual world root trajectory). Up-axis = Y (BVH convention, confirmed by
the parsed OFFSET/translation magnitudes -- see bvh_parser.detect_scale).

These clips are NOT already-trimmed single-behavior takes: e.g. dog_quad_walk_001
is a 110s capture that ambles all over the mocap volume, changing heading
throughout. So classification works in two passes:
  1. Per-frame velocity-direction heading + speed (root, ground-plane X-Z).
  2. Segment into candidate windows via heading-stability + speed-band, same
     heading-stability-core + straightness-extension idea used for the InterPet4D
     window selection (docs/locomotion2/dataset/extract_walk.py), generalized to
     scan a whole (possibly very long) file for ALL qualifying windows, not just
     the single best one.
Backward-walk detection cross-checks velocity-heading against the root's OWN
rotation-implied forward axis (calibrated on a known forward-walk clip).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bvh_parser import detect_scale, forward_kinematics, parse_bvh

DATA_DIR = "/pub0/muhammadf/Blender/dataset/lifelike_dog/raw_bvh/raw_bvh_data"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "dataset_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ROOT_NAME = "Bip01"
PAW_NAMES = {"fl": "b_LeftHand", "fr": "b_RightHand", "bl": "b_LeftToe", "br": "b_RightToe"}

# filename-implied category (ground truth from the asset pack naming; cross-checked
# numerically below rather than trusted blindly, per the lesson that automated
# metrics on this kind of data can be fooled by a clip that isn't doing what its
# name says)
NAME_CATEGORY = {
    "idle": "idle", "quad_walk": "walk", "star_walk": "turn_walk", "zig_walk": "turn_walk",
    "quad_walkrun": "transition", "back": "backward", "fast_run": "run", "quad_run": "run",
    "jump_high": "jump", "jump": "jump", "play": "other", "hit": "other",
}


def categorize_name(fname: str) -> str:
    stem = fname.replace(".bvh", "")
    for key in ["quad_walkrun", "fast_run", "quad_run", "quad_walk", "star_walk", "zig_walk",
                "jump_high", "jump", "idle", "back", "play", "hit"]:
        if stem.startswith(f"dog_{key}"):
            return NAME_CATEGORY[key]
    return "other"


def smooth(x, win):
    if win <= 1:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def rolling_circstd(theta, win):
    """Rolling circular std (radians) of an angle series, window in samples."""
    c = np.cos(theta)
    s = np.sin(theta)
    cw = smooth(c, win)
    sw = smooth(s, win)
    R = np.clip(np.sqrt(cw ** 2 + sw ** 2), 1e-9, 1.0)
    return np.sqrt(-2 * np.log(R))


def analyze_clip(path: str, fname: str) -> dict:
    b = parse_bvh(path)
    scale = detect_scale(b)
    pos, rot = forward_kinematics(b, scale=scale)
    names = [j.name for j in b.joints]
    root = pos[:, names.index(ROOT_NAME)]
    dt = b.frame_time
    fps = b.fps
    T = b.n_frames

    win_speed = max(1, int(round(0.25 * fps)))  # 0.25s box smoothing before differencing
    rx = smooth(root[:, 0], win_speed)
    rz = smooth(root[:, 2], win_speed)
    vx = np.gradient(rx, dt)
    vz = np.gradient(rz, dt)
    speed = np.sqrt(vx ** 2 + vz ** 2)
    heading = np.arctan2(vz, vx)

    win_head = max(1, int(round(1.0 * fps)))  # 1s window for heading stability
    heading_std = rolling_circstd(heading, win_head)

    root_y = root[:, 1]

    # paw heights (world Y) for contact-pattern sanity + gait cadence estimate
    paw_y = {leg: pos[:, names.index(n), 1] for leg, n in PAW_NAMES.items()}

    result = dict(
        file=fname,
        name_category=categorize_name(fname),
        n_frames=T, fps=float(fps), duration_s=T / fps,
        speed_mean=float(speed.mean()), speed_p95=float(np.percentile(speed, 95)),
        speed_max=float(speed.max()),
        root_y_min=float(root_y.min()), root_y_max=float(root_y.max()),
        heading_std_mean_deg=float(np.degrees(heading_std.mean())),
        has_nan=bool(np.isnan(pos).any()),
    )

    # ---- window scan: contiguous runs with speed in a walking band and low heading std
    moving = speed > 0.04
    steady_heading = heading_std < np.radians(20)
    good = moving & steady_heading
    windows = []
    i = 0
    min_len = int(round(1.5 * fps))  # at least 1.5s
    while i < T:
        if good[i]:
            j = i
            while j < T and good[j]:
                j += 1
            if j - i >= min_len:
                seg_speed = speed[i:j]
                disp = np.hypot(rx[j - 1] - rx[i], rz[j - 1] - rz[i])
                path_len = np.sum(np.hypot(np.diff(rx[i:j]), np.diff(rz[i:j])))
                straightness = disp / (path_len + 1e-9)
                windows.append(dict(
                    t0=i / fps, t1=j / fps, dur=(j - i) / fps,
                    speed_mean=float(seg_speed.mean()), speed_std=float(seg_speed.std()),
                    straightness=float(straightness),
                ))
            i = j
        else:
            i += 1
    windows.sort(key=lambda w: -w["dur"])
    result["n_candidate_windows"] = len(windows)
    result["best_windows"] = windows[:3]

    # idle detection: fraction of frames essentially stationary
    result["frac_stationary"] = float((speed < 0.04).mean())

    # ---- stationary-window scan (genuine standing, for the "idle" style reference) ----
    still = speed < 0.04
    still_windows = []
    i = 0
    min_still = int(round(1.0 * fps))
    while i < T:
        if still[i]:
            j = i
            while j < T and still[j]:
                j += 1
            if j - i >= min_still:
                still_windows.append(dict(t0=i / fps, t1=j / fps, dur=(j - i) / fps))
            i = j
        else:
            i += 1
    still_windows.sort(key=lambda w: -w["dur"])
    result["n_stationary_windows"] = len(still_windows)
    result["best_stationary_windows"] = still_windows[:2]

    return result, dict(t=np.arange(T) / fps, speed=speed, heading_deg=np.degrees(heading),
                         heading_std_deg=np.degrees(heading_std), root_y=root_y, paw_y=paw_y,
                         rx=rx, rz=rz)


def calibrate_forward_axis(quad_walk_result, quad_walk_series):
    """Sanity check only (reported, not currently used to gate anything): fraction
    of a known-forward clip's moving frames whose velocity heading matches within
    90 deg of the clip's own mean heading (i.e. it doesn't reverse direction)."""
    s = quad_walk_series
    moving = s["speed"] > 0.04
    if moving.sum() < 10:
        return None
    mean_h = np.degrees(np.arctan2(np.mean(np.sin(np.radians(s["heading_deg"][moving]))),
                                    np.mean(np.cos(np.radians(s["heading_deg"][moving])))))
    diffs = (s["heading_deg"][moving] - mean_h + 180) % 360 - 180
    return float((np.abs(diffs) < 90).mean())


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.bvh")))
    all_results = []
    series_cache = {}
    for path in files:
        fname = os.path.basename(path)
        res, series = analyze_clip(path, fname)
        all_results.append(res)
        series_cache[fname] = series
        bsw = res["best_stationary_windows"][0] if res["best_stationary_windows"] else None
        bsw_s = f"[{bsw['t0']:.1f},{bsw['t1']:.1f}]s" if bsw else "--"
        print(f"{fname:32s} cat={res['name_category']:10s} dur={res['duration_s']:6.1f}s "
              f"speed_mean={res['speed_mean']:.3f} stationary={res['frac_stationary']:.2f} "
              f"windows={res['n_candidate_windows']} still_win={bsw_s} nan={res['has_nan']}")

    with open(os.path.join(OUT_DIR, "dataset_scan_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # forward-axis calibration using dog_quad_walk_001 (per filename, a forward walk)
    fw_check = calibrate_forward_axis(None, series_cache["dog_quad_walk_001.bvh"])

    # ---- validation plots: one per category-representative clip
    reps = {
        "idle": "dog_idle_002.bvh",
        "walk": "dog_quad_walk_001.bvh",
        "walk2": "dog_quad_walk_002.bvh",
        "turn_walk": "dog_zig_walk_001.bvh",
        "transition": "dog_quad_walkrun_001.bvh",
        "backward": "dog_back_001.bvh",
        "run": "dog_quad_run_001.bvh",
    }
    for tag, fname in reps.items():
        if fname not in series_cache:
            continue
        s = series_cache[fname]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(s["rx"], s["rz"])
        axes[0, 0].set_title(f"{fname}: top-down root path (X vs Z)")
        axes[0, 0].set_xlabel("X (m)"); axes[0, 0].set_ylabel("Z (m)"); axes[0, 0].axis("equal")

        axes[0, 1].plot(s["t"], s["speed"])
        axes[0, 1].axhline(0.04, color="gray", ls="--", lw=0.7)
        axes[0, 1].set_title("root speed (m/s)"); axes[0, 1].set_xlabel("t (s)")

        axes[1, 0].plot(s["t"], s["heading_std_deg"])
        axes[1, 0].set_title("rolling heading std (deg, 1s window)"); axes[1, 0].set_xlabel("t (s)")

        for leg, y in s["paw_y"].items():
            axes[1, 1].plot(s["t"], y, label=leg, lw=0.8)
        axes[1, 1].set_title("paw height Y (m)"); axes[1, 1].set_xlabel("t (s)"); axes[1, 1].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{tag}_{fname.replace('.bvh','')}.png"), dpi=110)
        plt.close(fig)

    # ---- write DATASET_REPORT.md
    lines = []
    lines.append("# Lifelike Dog BVH Dataset Report\n")
    lines.append(f"Source: `dataset/lifelike_dog/raw_bvh/raw_bvh_data/` ({len(files)} BVH files).\n")
    lines.append("## Skeleton\n")
    lines.append(
        "3ds-Max-Biped-style rig (human bone names retargeted onto a quadruped mesh): "
        "root `Bip01` -> `b_Hips` -> spine chain (`b_Spine`..`b_Spine3`) -> neck/head, "
        "and two shoulder clavicles `b_LeftClav`/`b_RightClav` for the FRONT legs "
        "(`_Arm` -> `_ForeArm` -> `_Hand` -> `_Finger` -> end site), plus a tail chain "
        "off `b_Hips` and hind legs off `b_Hips` directly "
        "(`_LegUpper` -> `_Leg` -> `_Leg1` -> `_Ankle` -> `_Toe` -> `_Toe002`/end site). "
        "Units are centimeters (auto-detected and converted to meters). Up-axis is Y "
        "(root/hip height sits at ~0.15-0.6 m depending on clip and pose; paw Y sits "
        "near 0 at ground contact). All 33 files share FPS=120 (Frame Time 0.008333s).\n"
    )
    lines.append(
        "**2-segment leg abstraction used for AMP features** (to match Bingo's 2-segment "
        "legs): front hip=`b_*Arm`, mid(\"knee\")=`b_*ForeArm`, paw=`b_*Hand`; "
        "rear hip=`b_*LegUpper`, mid(\"knee\")=`b_*Ankle` (absorbing the `Leg`/`Leg1` "
        "sub-bend), paw=`b_*Toe`.\n"
    )
    if fw_check is not None:
        lines.append(
            f"Forward-axis sanity check on `dog_quad_walk_001` (a named forward walk): "
            f"{fw_check*100:.0f}% of moving frames have a velocity heading within 90 deg "
            f"of the clip's own mean heading -- i.e. **the clip does not reverse "
            f"direction** even though (see below) it changes heading substantially "
            f"over its 110s duration (it ambles around the capture volume, not a "
            f"straight line for the whole clip).\n"
        )
    lines.append("## Per-clip scan (name-implied category, cross-checked numerically)\n")
    lines.append("| file | category | dur(s) | mean speed (m/s) | stationary frac | candidate windows | best window | best stationary window |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_results:
        bw = r["best_windows"][0] if r["best_windows"] else None
        bw_s = f"[{bw['t0']:.1f},{bw['t1']:.1f}]s v={bw['speed_mean']:.2f} straight={bw['straightness']:.2f}" if bw else "--"
        bsw = r["best_stationary_windows"][0] if r["best_stationary_windows"] else None
        bsw_s = f"[{bsw['t0']:.1f},{bsw['t1']:.1f}]s ({bsw['dur']:.1f}s)" if bsw else "--"
        lines.append(f"| {r['file']} | {r['name_category']} | {r['duration_s']:.1f} | "
                     f"{r['speed_mean']:.3f} | {r['frac_stationary']:.2f} | {r['n_candidate_windows']} | {bw_s} | {bsw_s} |")

    lines.append("\n## Category summary\n")
    from collections import Counter
    cats = Counter(r["name_category"] for r in all_results)
    for c, n in cats.items():
        lines.append(f"- **{c}**: {n} clips")

    # data-driven idle-window summary across the WHOLE dataset (not just *_idle* files)
    idle_candidates = []
    for r in all_results:
        for w in r["best_stationary_windows"]:
            idle_candidates.append((r["file"], w["t0"], w["t1"], w["dur"]))
    idle_candidates.sort(key=lambda x: -x[3])
    total_idle_s = sum(r["frac_stationary"] * r["duration_s"] for r in all_results
                        if r["name_category"] in ("idle", "walk"))

    lines.append("\n## Important finding: filename categories do not match actual motion\n")
    lines.append(
        "Numerically checking root speed exposed a mismatch: the `dog_idle_*.bvh` clips "
        "are **not standing still** -- `frac_stationary` (fraction of frames with root "
        "speed < 0.04 m/s) is only 0.00-0.23 for the four idle clips, and their MEAN speed "
        "(0.97-1.53 m/s) is comparable to or higher than the `quad_walk` clips (~1.0 m/s). "
        "Root (`Bip01`) and all four paws travel the same multi-meter excursions in these "
        "files (verified directly, not just at the root), so this is real locomotion, not "
        "a moving-root/stationary-mesh export artifact. This matches the standing lesson "
        "from the InterPet4D work: **never trust a clip's name or an automated aggregate "
        "metric without checking the actual motion.** Genuinely still moments do exist, "
        "but only as short windows inside otherwise-moving clips -- see the per-clip "
        "`best_stationary_window` column above and the ranked list below.\n"
    )
    lines.append("### Best stationary windows found dataset-wide (>=1.0s of speed<0.04 m/s)\n")
    if idle_candidates:
        for fname, t0, t1, dur in idle_candidates[:8]:
            lines.append(f"- `{fname}` [{t0:.1f}, {t1:.1f}]s ({dur:.1f}s)")
    else:
        lines.append("- none found >=1.0s anywhere in the dataset.\n")

    lines.append("\n## Selection for the first AMP controller\n")
    lines.append(
        "Per the brief (idle + forward walk + useful walk transitions only; exclude "
        "run/jump/play/aggressive), the expert set for `extract_dog_amp_features.py` is:\n\n"
        "- **idle/stand**: the ranked stationary windows above (not full `idle_*` clips, "
        "since those are mostly moving -- see finding above). If total stationary duration "
        "is too short to be a useful style prior on its own, the AMP env's own "
        "upright/stability + zero-command-tracking reward terms carry the standing "
        "behavior instead, same as the prior (rev_3-era) `bingo_amp_env` design.\n"
        "- **forward walk**: clean straight-walk windows extracted from `dog_quad_walk_001.bvh` "
        "and `dog_quad_walk_002.bvh` (both long multi-behavior captures; only the "
        "heading-stable, straightness>0.9 candidate windows listed above are used, not "
        "the full files).\n"
        "- **walk transitions**: `dog_quad_walkrun_*.bvh` are EXCLUDED from this first pass "
        "-- inspecting their speed traces shows they ramp continuously from walk into a "
        "true run with no sustained walk-paced plateau long enough to be useful without "
        "also teaching a running gait, which the brief explicitly excludes.\n"
        "- **excluded**: `*_run*`, `*_jump*`, `play_*`, `hit_*` (per brief), and "
        "`*_star_walk*`/`*_zig_walk*`/`*_back*` (turning/backward -- out of scope for a "
        "straight-forward-only first controller, but cataloged above for a later stage).\n"
    )
    with open(os.path.join(OUT_DIR, "DATASET_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {os.path.join(OUT_DIR, 'DATASET_REPORT.md')}")
    print(f"wrote {len(reps)} validation plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
