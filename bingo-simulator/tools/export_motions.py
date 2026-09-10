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
     "Stage 4 pass: 55 frames, completes, 4/4 contacts (MEMORY.md)"),
    ("No", "stage4/out/reaction_no_v4_dynamic.npz",
     "Stage 4 pass: 19 frames, completes, 4/4 contacts (MEMORY.md)"),
    ("What", "stage4/out/reaction_what_v4_dynamic.npz",
     "Stage 4 pass: 71 frames, completes, 4/4 contacts (MEMORY.md)"),
]

# Exposed in the UI as unavailable, with the reason, rather than hidden.
NOT_READY = [
    ("Cheeky", "Stage 4 completes but world-space fidelity is not a holistic pass "
               "(1.405 m / 77.8 deg error). MEMORY.md: survival alone is not complete."),
    ("Timid", "Stage 4 completes but world-space fidelity is not a holistic pass "
              "(0.611 m / 11.3 deg error)."),
    ("DeadPan", "No Stage-4 pass recorded."),
    ("Eccentric", "No Stage-4 pass recorded; the clip is an authored sit."),
    ("Enthusiastic", "No Stage-4 pass recorded."),
    ("Laidback", "No Stage-4 pass recorded."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(SIM_ROOT, "app/public/motions"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    index = {"ready": [], "not_ready": [
        {"name": n, "reason": r} for n, r in NOT_READY]}

    for name, rel, evidence in READY:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            print(f"[[ MISSING {rel} - skipping {name}")
            index["not_ready"].append(
                {"name": name, "reason": f"source motion not on disk: {rel}"})
            continue
        m = np.load(p, allow_pickle=True)
        names = [str(s) for s in m["dof_names"]]
        if names != DOF_ORDER:
            raise SystemExit(f"{rel}: dof_names is not the canonical order")
        dof = m["dof_positions"].astype(float)
        fps = float(m["fps"])
        out = {
            "name": name,
            "fps": fps,
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
        fn = os.path.join(a.out_dir, f"{name.lower()}.json")
        with open(fn, "w") as f:
            json.dump(out, f)
        index["ready"].append({
            "name": name, "file": f"motions/{name.lower()}.json",
            "frames": out["frames"], "duration_s": round(out["duration_s"], 3),
            "evidence": evidence,
        })
        print(f"[[ {name:6s} {out['frames']:4d} frames @ {fps:.0f} Hz "
              f"= {out['duration_s']:.2f} s -> {os.path.basename(fn)}")

    ip = os.path.join(a.out_dir, "index.json")
    with open(ip, "w") as f:
        json.dump(index, f, indent=1)
    print(f"[[ wrote {ip}: {len(index['ready'])} ready, "
          f"{len(index['not_ready'])} withheld")


if __name__ == "__main__":
    main()
