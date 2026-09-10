"""Visual check of the two Task-1 design decisions that were derived, not inherited.

Both are claims in docs/locomotion/TASK1_DESIGN.md that are easy to state and easy
to get wrong, so this draws them from the real geometry:

  1. The stance. BINGO_V4_CFG.init_state still carries the rev_3 pose. Plotted
     against the actual collision hulls next to STAND_SOLVED, the difference is
     not subtle - one paw ends up 207 mm in front of the base.

  2. The action scale. Shows each leg joint's hard limit, soft limit, stance value
     and the reach of the chosen per-joint scale at |action| = 3, against what a
     single scale of 0.3 (Stage 5's residual scale) would ask for.

Runs on system python with numpy + matplotlib. No Isaac Sim, no Blender.

    python3 rl/tools/plot_stance_and_action_scale.py --out docs/locomotion/stance_and_action_scale.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "stage2"))
sys.path.insert(0, os.path.join(ROOT, "stage4"))

from v4_kinematics import V4Kin, LEGS, DOF_ORDER, axis_rot  # noqa: E402

URDF = os.path.join(ROOT, "URDF/bingo_urdf v4_w_ear_joints/urdf/"
                          "bingo_urdf_w_ear_joints_physics.urdf")
HULLS = os.path.join(ROOT, "stage4/out/collision_hulls.npz")

STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}
LEGACY_REV3 = {
    "fl_SP_J": -0.3, "bl_SP_J": -0.3, "fr_SP_J": +0.3, "br_SP_J": -0.3,
    "fl_knee": +0.6, "bl_knee": +0.6, "fr_knee": -0.6, "br_knee": -0.6,
}
ACTION_SCALE = {".*_SY_J": 0.125, ".*_SP_J": 0.195, ".*_knee": 0.170}
STAGE5_SCALE = 0.30
SOFT = 0.9

kin = V4Kin(URDF)
hull = {k: v for k, v in np.load(HULLS).items()}


def leg_frames(pose, leg):
    """(SP pivot, knee pivot, knee-link rotation) in the base frame."""
    R, p = np.eye(3), np.zeros(3)
    pts = {}
    q = [pose.get(f"{leg}_SY_J", 0.0), pose.get(f"{leg}_SP_J", 0.0),
         pose.get(f"{leg}_knee", 0.0)]
    for i, (name, qi) in enumerate(zip(kin.leg_chain(leg), q)):
        J = kin.j[name]
        p = p + R @ J["xyz"]
        if i == 1:
            pts["sp"] = p.copy()
        elif i == 2:
            pts["knee"] = p.copy()
        R = R @ J["R"] @ axis_rot(J["axis"], qi)
    return pts["sp"], pts["knee"], R, p


def paw_world(pose, leg):
    _, _, R, p = leg_frames(pose, leg)
    return hull[f"{leg}_knee"] @ R.T + p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/locomotion/stance_and_action_scale.png"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.26)

    # ---------------- row 1: the two stances, side view and top view ------------
    for col, (title, pose) in enumerate((
            ("STAND_SOLVED  (used by the locomotion env)", STAND_SOLVED),
            ("rev_3 legacy pose  (still in bingo_v4.py init_state)", LEGACY_REV3))):
        ax = fig.add_subplot(gs[0, col])
        lows = []
        for leg in LEGS:
            sp, knee, R, p = leg_frames(pose, leg)
            w = paw_world(pose, leg)
            lows.append(w[:, 2].min())
            colr = "#3b6ea5" if leg.startswith("f") else "#a5563b"
            ax.plot([0, sp[0]], [0, sp[2]], color="0.75", lw=1.0)
            ax.plot([sp[0], knee[0]], [sp[2], knee[2]], color=colr, lw=2.2)
            ax.plot(w[:, 0], w[:, 2], ".", ms=1.2, color=colr, alpha=0.5)
            i = int(np.argmin(w[:, 2]))
            ax.plot(w[i, 0], w[i, 2], "v", ms=7, color=colr)
            ax.annotate(leg, (knee[0], knee[2]), fontsize=7, color=colr,
                        xytext=(3, 3), textcoords="offset points")
        lows = np.array(lows)
        floor = lows.min()
        ax.axhline(floor, color="#2a7f62", lw=1.2, ls="--")
        ax.plot(0, 0, "ks", ms=7)
        ax.annotate("base origin", (0, 0), fontsize=7, xytext=(4, 6),
                    textcoords="offset points")
        ax.set_title(f"{title}\npaw z spread {1000*(lows.max()-lows.min()):.2f} mm   "
                     f"base height {-floor:.3f} m", fontsize=9)
        ax.set_xlabel("x  forward (m)", fontsize=8)
        ax.set_ylabel("z  up (m)", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # top view: support polygons compared
    ax = fig.add_subplot(gs[0, 2])
    for tag, pose, colr in (("STAND_SOLVED", STAND_SOLVED, "#2a7f62"),
                            ("rev_3 legacy", LEGACY_REV3, "#a53b3b")):
        pts = []
        for leg in LEGS:
            w = paw_world(pose, leg)
            pts.append(w[int(np.argmin(w[:, 2])), :2])
        pts = np.array(pts)
        order = [0, 1, 3, 2]           # fl, fr, br, bl -> a convex ring
        ring = pts[order]
        ax.add_patch(Polygon(ring, closed=True, fill=True, alpha=0.16, color=colr))
        ax.plot(np.r_[ring[:, 0], ring[0, 0]], np.r_[ring[:, 1], ring[0, 1]],
                "-o", ms=5, color=colr, lw=1.6, label=tag)
        span = pts[:, 0].max() - pts[:, 0].min()
        ax.annotate(f"{tag}: x span {1000*span:.0f} mm",
                    (0.03, 0.95 if colr == "#2a7f62" else 0.89),
                    xycoords="axes fraction", fontsize=8, color=colr)
    ax.plot(0, 0, "ks", ms=7)
    ax.set_title("support polygon, top view\n(the legacy pose is not a stance)",
                 fontsize=9)
    ax.set_xlabel("x forward (m)", fontsize=8)
    ax.set_ylabel("y left (m)", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

    # ---------------- row 2: action scale vs joint limits -----------------------
    ax = fig.add_subplot(gs[1, :])
    leg_names = DOF_ORDER[:12]
    lims = kin.all_limits()
    y = np.arange(len(leg_names))[::-1]
    for i, jn in enumerate(leg_names):
        lo, hi = lims[i]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * SOFT
        slo, shi = mid - half, mid + half
        q = STAND_SOLVED[jn]
        scale = next(s for pat, s in ACTION_SCALE.items() if re.fullmatch(pat, jn))
        yy = y[i]
        ax.plot([lo, hi], [yy, yy], color="0.82", lw=9, solid_capstyle="butt",
                zorder=1, label="hard limit" if i == 0 else None)
        ax.plot([slo, shi], [yy, yy], color="0.62", lw=9, solid_capstyle="butt",
                zorder=2, label="soft limit (x0.9)" if i == 0 else None)
        ax.plot([q - 3 * STAGE5_SCALE, q + 3 * STAGE5_SCALE], [yy, yy],
                color="#a53b3b", lw=3.5, solid_capstyle="butt", zorder=3,
                label="Stage-5 scale 0.30 at |a|=3" if i == 0 else None)
        ax.plot([q - 3 * scale, q + 3 * scale], [yy, yy], color="#2a7f62", lw=3.5,
                solid_capstyle="butt", zorder=4,
                label="chosen per-joint scale at |a|=3" if i == 0 else None)
        ax.plot(q, yy, "k|", ms=13, zorder=5,
                label="stance pose" if i == 0 else None)
        over = max(0.0, (q + 3 * STAGE5_SCALE) - shi, slo - (q - 3 * STAGE5_SCALE))
        if over > 1e-6:
            # fixed column clear of the widest bar (knee soft limit 1.404 rad),
            # otherwise the SY labels land on top of their own bars
            ax.annotate(f"0.30 overruns by {over:.2f} rad", (1.62, yy),
                        fontsize=7, color="#a53b3b", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(leg_names, fontsize=8)
    ax.set_xlabel("joint angle (rad)", fontsize=9)
    ax.set_title("Per-joint action scale vs the real v4 limits. The chosen scale reaches "
                 "the soft limit at |action| = 3; a single 0.30 does not fit SY.",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.set_xlim(-2.0, 2.4)

    fig.suptitle("Bingo Task 1 - derived design decisions, drawn from the real URDF "
                 "and collision hulls", fontsize=12)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"[[ wrote {a.out}")

    # numbers alongside the picture
    for tag, pose in (("STAND_SOLVED", STAND_SOLVED), ("rev_3 legacy", LEGACY_REV3)):
        pts = np.array([paw_world(pose, l)[np.argmin(paw_world(pose, l)[:, 2])]
                        for l in LEGS])
        print(f"[[ {tag:14s} paw z {np.round(pts[:,2],4)}  spread "
              f"{1000*(pts[:,2].max()-pts[:,2].min()):5.2f} mm  x span "
              f"{1000*(pts[:,0].max()-pts[:,0].min()):6.1f} mm")


if __name__ == "__main__":
    main()
