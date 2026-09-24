"""Export the Stage-3/4 validated robot-space motions for the browser.

Only clips whose Stage-4 status is an actual pass are exported. MEMORY.md is
authoritative here and a Stage-3 animation existing is explicitly NOT sufficient:

    Yes  reaction_yes_v4_dynamic   55 frames, completes, 4/4 contacts   -> ready
    No   reaction_no_v4_dynamic    19 frames, completes, 4/4 contacts   -> ready
    What reaction_what_v4_dynamic  71 frames, completes, 4/4 contacts   -> ready

    Cheeky / Timid  complete physics playback but their Stage-4 world-space fidelity
                    is NOT a holistic pass (1.405 m / 77.8 deg and 0.611 m / 11.3 deg
                    respectively). MEMORY.md: "Survival alone does not make these
                    clips complete." Not exported.
    DeadPan, Eccentric, Enthusiastic, Laidback
                    no Stage-4 pass recorded. Not exported.

What is exported is JOINT ANGLES ONLY - the 21 DOF, in canonical order. The floating
root is deliberately left out: the browser tracks these as PD targets under gravity
and contact, exactly as Stage 4 does, and never teleports the root.

    python3 bingo-simulator/tools/export_motions.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(SIM_ROOT, ".."))

DOF_ORDER = [
    "fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
    "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
    "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
    "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
]

# (skill name, source npz, the Stage-4 evidence that justifies exposing it)
READY = [
    ("Yes", "stage4/out/reaction_yes_v4_dynamic.npz",
     "Stage 4 pass: 55 frames, completes, 4/4 contacts (MEMORY.md). "
     "Simulator gate: holds, 3/3 runs, <=2.6 deg tilt, 4/4 contacts."),
    ("No", "stage4/out/reaction_no_v4_dynamic.npz",
     "Stage 4 pass: 19 frames, completes, 4/4 contacts (MEMORY.md). "
     "Simulator gate: holds, 3/3 runs, <=6.6 deg tilt, 4/4 contacts."),
    ("What", "stage4/out/reaction_what_v4_dynamic.npz",
     "Stage 4 pass: 71 frames, completes, 4/4 contacts (MEMORY.md). "
     "Simulator gate: holds, 3/3 runs, <=0.6 deg tilt, 4/4 contacts."),
]

# Clips with a Stage-4 run on record but NOT a holistic Isaac world-space pass.
# Isaac world-space drift is not the same question the browser asks: the browser
# never teleports the root, it tracks these as PD targets under gravity and contact.
# So they are exported with --include-candidates and gated by an actual measurement
# in the simulator (tools/probe_skills.mjs), not promoted on faith.
CANDIDATES = [
    ("Cheeky",       "motions/cheeky_v4.npz",       "stage4/out/cheeky_v4_stage4.npz"),
    ("Timid",        "motions/timid_v4.npz",        "stage4/out/timid_v4_stage4.npz"),
    ("DeadPan",      "motions/deadpan_v4.npz",      "stage4/out/deadpan_v4_stage4.npz"),
    ("Eccentric",    "motions/eccentric_v4.npz",    "stage4/out/eccentric_v4_stage4.npz"),
    ("Enthusiastic", "motions/enthusiastic_v4.npz", "stage4/out/enthusiastic_v4_stage4.npz"),
    ("Laidback",     "motions/laidback_v4.npz",     "stage4/out/laidback_v4_stage4.npz"),
]


def stage4_metrics(rel):
    """Contact fraction, mean joint tracking error and final root drift."""
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        return None
    s = np.load(p, allow_pickle=True)
    return {
        "contact_pct": round(100.0 * float(np.mean(s["contacts"])), 1),
        "q_err_rad": round(float(np.mean(np.abs(s["q_act"] - s["q_ref"]))), 4),
        "root_drift_m": round(
            float(np.linalg.norm(s["root_pos"][-1] - s["root_pos_ref"][-1])), 4),
    }


def write_motion(name, rel, out_dir, evidence, extra=None):
    """Export one clip's joint angles to JSON. Returns the index entry or None."""
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        return None
    m = np.load(p, allow_pickle=True)
    names = [str(s) for s in m["dof_names"]]
    if names != DOF_ORDER:
        raise SystemExit(f"{rel}: dof_names is not the canonical order")
    dof = m["dof_positions"].astype(float)
    fps = float(m["fps"])
    # Several reaction clips author ONLY the 9 head/tail/ear channels and leave the
    # 12 leg channels at exactly zero. Driving those zeros would command the legs
    # straight and topple the robot, so the runtime must leave the legs alone for
    # these and let stand/locomotion keep holding the body up.
    expression_only = bool(np.all(np.abs(dof[:, :12]) < 1e-6))
    out = {
        "name": name,
        "fps": fps,
        "expression_only": expression_only,
        "frames": int(dof.shape[0]),
        "duration_s": float(dof.shape[0] / fps),
        "joint_order": DOF_ORDER,
        "dof_positions": [[round(float(v), 6) for v in row] for row in dof],
        "source": rel,
        "stage4_evidence": evidence,
        "note": ("Joint targets only. The floating root is intentionally absent: "
                 "the runtime tracks these under gravity and contact and never "
                 "teleports the root."),
    }
    fn = os.path.join(out_dir, f"{name.lower()}.json")
    with open(fn, "w") as f:
        json.dump(out, f)
    entry = {"name": name, "file": f"motions/{name.lower()}.json",
             "frames": out["frames"], "duration_s": round(out["duration_s"], 3),
             "expression_only": expression_only, "evidence": evidence}
    if extra:
        entry.update(extra)
    print(f"[[ {name:13s} {out['frames']:4d} frames @ {fps:.0f} Hz "
          f"= {out['duration_s']:.2f} s  "
          f"{'expression-only' if expression_only else 'full-body':15s} "
          f"-> {os.path.basename(fn)}")
    return entry


# Exposed in the UI as unavailable, with the reason, rather than hidden.
#
# These reasons are now the SIMULATOR's own measurement, not just Isaac's. Each clip
# was exported and played here via tools/probe_skills.mjs: the runtime tracks the
# clip's joint angles as PD targets under gravity and never teleports the root, and
# all six topple. They rise on straightening legs during the blend-in and then go
# over - the authored pose is not statically stable for Bingo, which is consistent
# with their poor Stage-4 world-space record. Falling is the honest outcome; the
# brief forbids faking it.
NOT_READY = [
    ("Cheeky", "Falls in the simulator: 114.6 deg tilt, 0/4 contacts "
               "(tools/probe_skills.mjs). Stage 4 also not a holistic pass "
               "(1.405 m root drift)."),
    ("Timid", "Falls in the simulator: 106.4 deg tilt, 0/4 contacts. Stage 4 not a "
              "holistic pass (0.611 m root drift)."),
    ("DeadPan", "Falls in the simulator: 101.7 deg tilt, 0/4 contacts. No Stage-4 "
                "pass recorded."),
    ("Eccentric", "Falls in the simulator: 119.3 deg tilt, 0/4 contacts. No Stage-4 "
                  "pass recorded; the clip is an authored sit."),
    ("Enthusiastic", "Falls in the simulator: 114.5 deg tilt, 0/4 contacts. No "
                     "Stage-4 pass recorded."),
    ("Laidback", "Falls in the simulator: 69.7 deg tilt, 2/4 contacts. No Stage-4 "
                 "pass recorded."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(SIM_ROOT, "app/public/motions"))
    ap.add_argument("--include-candidates", action="store_true",
                    help="also export the six personality clips so they can be "
                         "measured in the simulator by tools/probe_skills.mjs")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    withheld = {n: r for n, r in NOT_READY}
    index = {"ready": [], "not_ready": []}

    for name, rel, evidence in READY:
        e = write_motion(name, rel, a.out_dir, evidence)
        if e is None:
            print(f"[[ MISSING {rel} - skipping {name}")
            withheld[name] = f"source motion not on disk: {rel}"
            continue
        index["ready"].append(e)

    if a.include_candidates:
        for name, rel, s4 in CANDIDATES:
            met = stage4_metrics(s4)
            ev = ("Stage-4 run on record (not an Isaac world-space pass): "
                  f"{met['contact_pct']}% contact, {met['q_err_rad']} rad mean joint "
                  f"error, {met['root_drift_m']} m root drift. UNGATED in the "
                  "simulator." if met else "Stage-4 result file not on disk.")
            e = write_motion(name, rel, a.out_dir, ev,
                             extra={"candidate": True, "stage4": met})
            if e is None:
                print(f"[[ MISSING {rel} - skipping {name}")
                continue
            index["ready"].append(e)
            withheld.pop(name, None)

    index["not_ready"] = [{"name": n, "reason": r} for n, r in withheld.items()]

    ip = os.path.join(a.out_dir, "index.json")
    with open(ip, "w") as f:
        json.dump(index, f, indent=1)
    print(f"[[ wrote {ip}: {len(index['ready'])} ready, "
          f"{len(index['not_ready'])} withheld")


if __name__ == "__main__":
    main()
