"""Task 2B data audit - what locomotion content do the personality clips ACTUALLY contain?

Before any style-conditioned locomotion policy can claim to reproduce "Cheeky" or
"Timid", the source has to contain enough *locomotion* to define a gait. This script
measures that, per clip, from the Stage-3 robot-space motions in ``motions/*_v4.npz``
- i.e. from the exact 21-DOF v4 trajectories, not from Ashley's rig and not from the
clip names.

Everything here is descriptive. Nothing is synthesised, and a clip that does not walk
is reported as not walking rather than being given an invented gait.

Measured per clip:
  gait cycles        touchdown events per foot; a *cycle* is one complete
                     stance -> swing -> stance for that foot
  root velocity      heading-aligned forward / lateral / yaw-rate ranges
  stance/swing       duty factor, mean stance and swing durations
  contact pattern    per-leg phase offsets vs the reference leg -> gait class
  stride             length (root travel per cycle) and frequency (Hz)
  body pose          height above the Stage-2 floor (z = 0), pitch, roll
  root oscillation   peak-to-peak and RMS vertical motion of the base
  expression         head / tail / ear joint ranges and activity

Run (no Isaac needed, system python + numpy):
  python3 rl/tools/analyze_style_motions.py
  python3 rl/tools/analyze_style_motions.py --motions motions/cheeky_v4.npz --detail
  python3 rl/tools/analyze_style_motions.py --json out.json --plot out.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

LEGS = ["fl", "fr", "bl", "br"]
# The six authored personality performances. The reaction clips are deliberately
# excluded from the default set: they are seated/gestural by construction.
STYLE_CLIPS = ["cheeky", "timid", "deadpan", "eccentric", "enthusiastic", "laidback"]

EXPR_JOINTS = {
    "head": ["head_pitch_joint", "head_yaw", "head_roll"],
    "tail": ["tail_pitch", "tail_yaw"],
    "ear_l": ["l_ear_pitch", "l_ear_roll"],
    "ear_r": ["r_ear_pitch", "r_ear_roll"],
}

# A foot must be off the ground for at least this long for the lift to count as a
# swing rather than contact-sensor chatter. 3 frames at 24 Hz = 125 ms.
MIN_SWING_FRAMES = 3
MIN_STANCE_FRAMES = 2


def quat_to_euler(q):
    """wxyz quaternion array (T,4) -> roll, pitch, yaw (T,) each, radians."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sp)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def runs(mask):
    """Contiguous True runs of a boolean array as (start, end_exclusive) pairs."""
    out = []
    i = 0
    T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def clean_contacts(c):
    """Drop runs shorter than the minimum so sensor chatter is not read as a step."""
    c = c.copy()
    for s, e in runs(~c):
        if (e - s) < MIN_SWING_FRAMES and s > 0 and e < len(c):
            c[s:e] = True
    for s, e in runs(c):
        if (e - s) < MIN_STANCE_FRAMES and s > 0 and e < len(c):
            c[s:e] = False
    return c


def analyse(path):
    m = np.load(path, allow_pickle=True)
    fps = float(m["fps"])
    dt = 1.0 / fps
    dof_names = [str(s) for s in m["dof_names"]]
    # Two schemas in circulation, same convention as stage2/bake_v4_motion.py: the
    # Stage-2 solver writes root_pos/root_quat directly, the Stage-3/4 whole-body
    # files carry the root as body 0. Accept both rather than converting files.
    if "root_pos" in m.files and "root_quat" in m.files:
        root = m["root_pos"].astype(float)
        quat = m["root_quat"].astype(float)
    else:
        root = m["body_positions"][:, 0].astype(float)
        quat = m["body_rotations"][:, 0].astype(float)
    dof = m["dof_positions"].astype(float)
    contacts = m["contacts"].astype(bool)
    T = len(root)

    roll, pitch, yaw = quat_to_euler(quat)

    # ---- root motion, expressed in the character's own heading frame -------------
    vel_w = np.gradient(root, dt, axis=0)
    ch, sh = np.cos(yaw), np.sin(yaw)
    v_fwd = vel_w[:, 0] * ch + vel_w[:, 1] * sh
    v_lat = -vel_w[:, 0] * sh + vel_w[:, 1] * ch
    yaw_un = np.unwrap(yaw)
    yaw_rate = np.gradient(yaw_un, dt)

    horiz = np.linalg.norm(root[:, :2] - root[0, :2], axis=1)
    net_travel = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    path_len = float(np.sum(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)))

    # ---- per-foot gait structure -------------------------------------------------
    legs = {}
    cycles_per_leg = []
    for k, leg in enumerate(LEGS):
        c = clean_contacts(contacts[:, k])
        st = runs(c)
        sw = runs(~c)
        # A cycle is a touchdown-to-touchdown interval, so it needs >= 2 touchdowns.
        touchdowns = [s for s, _ in st if s > 0]
        periods = np.diff(touchdowns) * dt if len(touchdowns) >= 2 else np.array([])
        n_cycles = max(0, len(touchdowns) - 1)
        cycles_per_leg.append(n_cycles)
        legs[leg] = {
            "duty_factor": float(c.mean()),
            "n_stance_runs": len(st),
            "n_swing_runs": len(sw),
            "n_cycles": n_cycles,
            "mean_stance_s": float(np.mean([(e - s) * dt for s, e in st])) if st else 0.0,
            "mean_swing_s": float(np.mean([(e - s) * dt for s, e in sw])) if sw else 0.0,
            "touchdown_frames": [int(t) for t in touchdowns],
            "mean_period_s": float(periods.mean()) if len(periods) else 0.0,
            "stride_freq_hz": float(1.0 / periods.mean()) if len(periods) and periods.mean() > 0 else 0.0,
        }

    # Stride length: root travel between successive touchdowns of the same foot,
    # averaged over every foot that actually completes a cycle.
    strides = []
    for k, leg in enumerate(LEGS):
        td = legs[leg]["touchdown_frames"]
        for a, b in zip(td[:-1], td[1:]):
            strides.append(float(np.linalg.norm(root[b, :2] - root[a, :2])))
    stride_len = float(np.mean(strides)) if strides else 0.0

    n_cycles_min = int(min(cycles_per_leg))
    n_cycles_total = int(sum(cycles_per_leg))

    # ---- contact pattern / gait classification ----------------------------------
    # Phase of each leg's first touchdown relative to fl, normalised by the mean cycle.
    ref_td = legs["fl"]["touchdown_frames"]
    ref_period = legs["fl"]["mean_period_s"]
    phases = {}
    for leg in LEGS:
        td = legs[leg]["touchdown_frames"]
        if ref_td and td and ref_period > 0:
            d = (td[0] - ref_td[0]) * dt
            phases[leg] = float((d / ref_period) % 1.0)
        else:
            phases[leg] = float("nan")

    n_down = contacts.sum(1)
    gait = classify_gait(phases, n_cycles_min, float(n_down.mean()))

    # ---- expression --------------------------------------------------------------
    expr = {}
    for group, names in EXPR_JOINTS.items():
        idx = [dof_names.index(n) for n in names if n in dof_names]
        if not idx:
            continue
        block = dof[:, idx]
        expr[group] = {
            "joints": [dof_names[i] for i in idx],
            "range_rad": [float(block.min()), float(block.max())],
            "peak_to_peak_rad": [float(v) for v in (block.max(0) - block.min(0))],
            "rms_rate_rad_s": float(np.sqrt(np.mean(np.gradient(block, dt, axis=0) ** 2))),
            "std_rad": [float(v) for v in block.std(0)],
        }

    return {
        "clip": os.path.basename(path).replace("_v4.npz", ""),
        "path": os.path.relpath(path, ROOT),
        "frames": int(T),
        "fps": fps,
        "duration_s": float(T * dt),
        "gait": {
            "class": gait,
            "cycles_min_over_legs": n_cycles_min,
            "cycles_total": n_cycles_total,
            "mean_feet_down": float(n_down.mean()),
            "phases_vs_fl": phases,
            "stride_length_m": stride_len,
            "stride_freq_hz": float(np.mean([legs[l]["stride_freq_hz"] for l in LEGS
                                             if legs[l]["stride_freq_hz"] > 0]))
            if any(legs[l]["stride_freq_hz"] > 0 for l in LEGS) else 0.0,
        },
        "legs": legs,
        "root": {
            "net_travel_m": net_travel,
            "path_length_m": path_len,
            "max_displacement_m": float(horiz.max()),
            "v_fwd_range": [float(v_fwd.min()), float(v_fwd.max())],
            "v_fwd_mean": float(v_fwd.mean()),
            "v_lat_range": [float(v_lat.min()), float(v_lat.max())],
            "yaw_rate_range": [float(yaw_rate.min()), float(yaw_rate.max())],
            "net_yaw_deg": float(np.degrees(yaw_un[-1] - yaw_un[0])),
            "height_mean_m": float(root[:, 2].mean()),
            "height_range_m": [float(root[:, 2].min()), float(root[:, 2].max())],
            "height_p2p_mm": float(1000 * (root[:, 2].max() - root[:, 2].min())),
            "height_rms_mm": float(1000 * root[:, 2].std()),
            "pitch_deg_range": [float(np.degrees(pitch.min())), float(np.degrees(pitch.max()))],
            "roll_deg_range": [float(np.degrees(roll.min())), float(np.degrees(roll.max()))],
            "pitch_deg_mean": float(np.degrees(pitch.mean())),
            "roll_deg_mean": float(np.degrees(roll.mean())),
        },
        "expression": expr,
    }


def classify_gait(phases, n_cycles, mean_feet_down):
    """Name the contact pattern, or say plainly that there is no gait to name."""
    if n_cycles < 1:
        return "NO GAIT (no foot completes a touchdown-to-touchdown cycle)"
    if n_cycles < 2:
        return "SUB-GAIT (< 2 cycles on the least-active foot; not a limit cycle)"
    p = {k: v for k, v in phases.items() if not np.isnan(v)}
    if len(p) < 4:
        return "PARTIAL (a foot never touches down)"

    def near(a, b, tol=0.15):
        d = abs((a - b) % 1.0)
        return min(d, 1.0 - d) < tol

    # trot = diagonal pairs in phase, fl with br
    if near(p["fl"], p["br"]) and near(p["fr"], p["bl"]) and not near(p["fl"], p["fr"]):
        return "trot (diagonal pairs)"
    # pace = lateral pairs in phase
    if near(p["fl"], p["bl"]) and near(p["fr"], p["br"]) and not near(p["fl"], p["fr"]):
        return "pace (lateral pairs)"
    # bound = front pair together, back pair together
    if near(p["fl"], p["fr"]) and near(p["bl"], p["br"]) and not near(p["fl"], p["bl"]):
        return "bound (front/back pairs)"
    if mean_feet_down > 3.0:
        return "walk (>3 feet down on average)"
    return "irregular / unclassified"


# Froude number Fr = v^2 / (g*h) with h the standing hip height. Animals trot below
# Fr ~ 2-3 and gallop above it; a value far past that means the authored root is
# moving faster than any legged gait at this size, i.e. the spike is animation, not
# locomotion. Bingo stands ~0.18 m at the hip, so Fr = 3 is ~2.3 m/s.
FROUDE_MAX = 3.0
GRAVITY = 9.81
STAND_HIP_H = 0.18

# A trot keeps ~2 feet down, a walk ~3. Below this the clip is ballistic: the body is
# airborne more than it is supported, which no position-controlled policy can track.
MIN_MEAN_FEET_DOWN = 1.5


def verdict(a):
    """Is there enough locomotion here to define a personality-specific gait?

    Four independent criteria, all reported. A single label hides which one failed,
    and *which* one failed decides what to do about the clip.
    """
    g, r = a["gait"], a["root"]
    v_peak = max(abs(r["v_fwd_range"][0]), abs(r["v_fwd_range"][1]))
    froude = v_peak ** 2 / (GRAVITY * STAND_HIP_H)

    checks = {
        "cycles>=3": g["cycles_min_over_legs"] >= 3,
        "travel>0.3m": r["net_travel_m"] > 0.3,
        "support>=1.5": g["mean_feet_down"] >= MIN_MEAN_FEET_DOWN,
        f"Froude<{FROUDE_MAX:g}": froude < FROUDE_MAX,
    }
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return "USABLE", "steady-state, supported, physically plausible gait", checks, froude
    if len(failed) == 1 and failed[0] == "travel>0.3m":
        return "IN-PLACE", "cycles a gait but does not travel; usable for style, not for velocity", checks, froude
    if len(failed) == 1:
        return "MARGINAL", f"fails {failed[0]}", checks, froude
    return "NOT LOCOMOTION", "fails " + ", ".join(failed), checks, froude


def fmt_report(rows):
    out = []
    w = out.append
    w("=" * 108)
    w("TASK 2B DATA AUDIT - locomotion content of the Stage-3 personality motions")
    w("=" * 108)
    w("")
    w(f"{'clip':14s} {'dur':>6s} {'cyc':>4s} {'travel':>8s} {'v_fwd mean':>11s} "
      f"{'stride':>8s} {'freq':>7s} {'height':>8s} {'gait':>26s}")
    w(f"{'':14s} {'(s)':>6s} {'min':>4s} {'(m)':>8s} {'(m/s)':>11s} "
      f"{'(m)':>8s} {'(Hz)':>7s} {'(mm)':>8s}")
    w("-" * 108)
    for a in rows:
        g, r = a["gait"], a["root"]
        w(f"{a['clip']:14s} {a['duration_s']:6.2f} {g['cycles_min_over_legs']:4d} "
          f"{r['net_travel_m']:8.3f} {r['v_fwd_mean']:11.3f} "
          f"{g['stride_length_m']:8.3f} {g['stride_freq_hz']:7.2f} "
          f"{1000*r['height_mean_m']:8.1f} {g['class']:>26s}")
    w("")
    w("VERDICT - can this clip define a personality gait?")
    w("-" * 108)
    w(f"  {'clip':14s} {'verdict':16s} {'feet':>5s} {'v_peak':>7s} {'Froude':>7s}  reason")
    for a in rows:
        v, why, checks, froude = verdict(a)
        vp = max(abs(a["root"]["v_fwd_range"][0]), abs(a["root"]["v_fwd_range"][1]))
        w(f"  {a['clip']:14s} {v:16s} {a['gait']['mean_feet_down']:5.2f} {vp:7.2f} "
          f"{froude:7.2f}  {why}")
    w("")
    w("  criteria: cycles>=3 (a limit cycle exists) | travel>0.3 m (it goes somewhere) |")
    w("            support>=1.5 feet down on average (not ballistic) |")
    w(f"            Froude < {FROUDE_MAX:g} (peak root speed is achievable by a {STAND_HIP_H:.2f} m quadruped)")
    w("")
    w("PER-CLIP DETAIL")
    w("-" * 108)
    for a in rows:
        g, r = a["gait"], a["root"]
        w(f"\n### {a['clip']}  ({a['frames']} frames @ {a['fps']:.0f} Hz = {a['duration_s']:.2f} s)")
        w(f"  gait          {g['class']}   cycles/leg min {g['cycles_min_over_legs']} "
          f"total {g['cycles_total']}   mean feet down {g['mean_feet_down']:.2f}/4")
        w(f"  phase vs fl   " + "  ".join(
            f"{l}={g['phases_vs_fl'][l]:.2f}" if not np.isnan(g['phases_vs_fl'][l]) else f"{l}=n/a"
            for l in LEGS))
        w(f"  duty factor   " + "  ".join(f"{l}={a['legs'][l]['duty_factor']*100:4.0f}%" for l in LEGS))
        w(f"  stance/swing  " + "  ".join(
            f"{l}={a['legs'][l]['mean_stance_s']:.2f}/{a['legs'][l]['mean_swing_s']:.2f}s" for l in LEGS))
        w(f"  root travel   net {r['net_travel_m']:.3f} m  path {r['path_length_m']:.3f} m  "
          f"net yaw {r['net_yaw_deg']:+.1f} deg")
        w(f"  root vel      fwd [{r['v_fwd_range'][0]:+.3f},{r['v_fwd_range'][1]:+.3f}] mean {r['v_fwd_mean']:+.3f} m/s"
          f"   lat [{r['v_lat_range'][0]:+.3f},{r['v_lat_range'][1]:+.3f}]"
          f"   yaw [{r['yaw_rate_range'][0]:+.2f},{r['yaw_rate_range'][1]:+.2f}] rad/s")
        w(f"  body height   mean {1000*r['height_mean_m']:.1f} mm  "
          f"range [{1000*r['height_range_m'][0]:.1f},{1000*r['height_range_m'][1]:.1f}]  "
          f"p2p {r['height_p2p_mm']:.1f} mm  rms {r['height_rms_mm']:.1f} mm")
        w(f"  body attitude pitch [{r['pitch_deg_range'][0]:+.1f},{r['pitch_deg_range'][1]:+.1f}] mean {r['pitch_deg_mean']:+.1f} deg"
          f"   roll [{r['roll_deg_range'][0]:+.1f},{r['roll_deg_range'][1]:+.1f}] mean {r['roll_deg_mean']:+.1f} deg")
        for grp, e in a["expression"].items():
            w(f"  {grp:12s}  p2p " + " ".join(f"{v:.2f}" for v in e["peak_to_peak_rad"])
              + f" rad   rms rate {e['rms_rate_rad_s']:.2f} rad/s")
    return "\n".join(out)


def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(4, n, figsize=(3.2 * n, 11), squeeze=False)
    for j, a in enumerate(rows):
        m = np.load(os.path.join(ROOT, a["path"]), allow_pickle=True)
        fps = float(m["fps"])
        t = np.arange(a["frames"]) / fps
        if "root_pos" in m.files and "root_quat" in m.files:
            root = m["root_pos"].astype(float)
            quat = m["root_quat"].astype(float)
        else:
            root = m["body_positions"][:, 0].astype(float)
            quat = m["body_rotations"][:, 0].astype(float)
        contacts = m["contacts"].astype(bool)
        roll, pitch, yaw = quat_to_euler(quat)
        dt = 1.0 / fps
        vel = np.gradient(root, dt, axis=0)
        v_fwd = vel[:, 0] * np.cos(yaw) + vel[:, 1] * np.sin(yaw)

        ax = axes[0][j]
        for k, leg in enumerate(LEGS):
            c = clean_contacts(contacts[:, k])
            for s, e in runs(c):
                ax.broken_barh([(t[s], t[min(e, len(t) - 1)] - t[s])], (k - 0.4, 0.8),
                               color="#3b6ea5")
        ax.set_yticks(range(4)); ax.set_yticklabels(LEGS, fontsize=8)
        ax.set_ylim(-0.6, 3.6); ax.invert_yaxis()
        ax.set_title(f"{a['clip']}\n{a['gait']['class']}", fontsize=9)
        ax.set_xlabel("s", fontsize=7); ax.tick_params(labelsize=7)

        ax = axes[1][j]
        ax.plot(t, v_fwd, lw=1.0, color="#2a7f62")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_ylabel("v_fwd (m/s)", fontsize=8) if j == 0 else None
        ax.set_xlabel("s", fontsize=7); ax.tick_params(labelsize=7)

        ax = axes[2][j]
        ax.plot(t, 1000 * root[:, 2], lw=1.0, color="#a5563b")
        ax.set_ylabel("base z (mm)", fontsize=8) if j == 0 else None
        ax.set_xlabel("s", fontsize=7); ax.tick_params(labelsize=7)

        ax = axes[3][j]
        ax.plot(root[:, 0], root[:, 1], lw=1.0, color="#5a3ba5")
        ax.plot(root[0, 0], root[0, 1], "o", ms=4, color="#2a7f62")
        ax.plot(root[-1, 0], root[-1, 1], "s", ms=4, color="#a53b3b")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_ylabel("y (m)", fontsize=8) if j == 0 else None
        ax.set_xlabel("x (m)", fontsize=7); ax.tick_params(labelsize=7)

    fig.suptitle("Bingo personality clips - contact schedule, forward velocity, base height, ground track",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    print(f"[[ wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", nargs="*", default=None,
                    help="explicit .npz paths; default = the six personality clips")
    ap.add_argument("--json", default=None, help="write the full metric dict here")
    ap.add_argument("--report", default=None, help="write the text report here")
    ap.add_argument("--plot", default=None, help="write a summary figure here")
    a = ap.parse_args()

    if a.motions:
        paths = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in a.motions]
    else:
        paths = []
        for name in STYLE_CLIPS:
            p = os.path.join(ROOT, "motions", f"{name}_v4.npz")
            if os.path.exists(p):
                paths.append(p)
        paths += sorted(glob.glob(os.path.join(ROOT, "motions", "reaction*_v4.npz")))

    rows = [analyse(p) for p in paths]
    text = fmt_report(rows)
    print(text)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[[ wrote {a.json}")
    if a.report:
        with open(a.report, "w") as f:
            f.write(text + "\n")
        print(f"[[ wrote {a.report}")
    if a.plot:
        make_plot([r for r in rows if r["clip"] in STYLE_CLIPS], a.plot)


if __name__ == "__main__":
    main()
