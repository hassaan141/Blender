"""Stage 1 baker: Blender rig -> raw Cartesian motion JSON.

Exports world-space joint positions in RAW BLENDER UNITS. No scaling, no axis
conversion, no IK. Stage 2 owns all of that -- this script only reads what the
rig evaluates to, so it can be verified against the viewport.

Usage:
    blender -b blend/legacy_art_rig/Bingo_DeadPan.blend -P scripts/bake_motion.py -- --out raw/deadpan_raw.json
    blender -b blend/legacy_art_rig/Bingo_DeadPan.blend -P scripts/bake_motion.py -- --out raw/x.json --hz 120 --start 132 --end 192
"""

import bpy, json, sys, argparse, math

# --- leg map -------------------------------------------------------------
# The rig's .L bones sit on the character's RIGHT (character faces -Y, so its
# left is +X, but .L bones are at -X). Names are mirrored; this table is the
# single place that is corrected.
LEGS = {
    "fl": {"sy": "def_Shoulder.R", "sp": "def_Arm.R",  "kn": "def_ForeArm.R", "tip": "def_Hand.R"},
    "fr": {"sy": "def_Shoulder.L", "sp": "def_Arm.L",  "kn": "def_ForeArm.L", "tip": "def_Hand.L"},
    "bl": {"sy": "def_Hip.R",      "sp": "def_Leg.R",  "kn": "def_Shin.R",    "tip": "def_Foot.R"},
    "br": {"sy": "def_Hip.L",      "sp": "def_Leg.L",  "kn": "def_Shin.L",    "tip": "def_Foot.L"},
}
BODY = {"spine_front": "def_Spine.001", "spine_back": "def_Pelvis",
        "neck": "def_Neck", "head": "def_Head",
        "tail1": "def_Tail.001", "tail2": "def_Tail.002"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rig", default="Bingo_Rig")
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args(argv)


def main():
    a = parse_args()
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]

    f0 = a.start if a.start is not None else sc.frame_start
    f1 = a.end if a.end is not None else sc.frame_end
    step = sc.render.fps / a.hz          # frames per sample; 24fps @120Hz -> 0.2
    n = int(round((f1 - f0) / step)) + 1

    # Rest-pose segment lengths, so Stage 2 computes its scale factor instead of
    # guessing it (spec B.1.5).
    rest = {}
    for leg, m in LEGS.items():
        rest[leg] = {k: round(rig.data.bones[bn].length, 6) for k, bn in m.items()}

    frames = []
    for i in range(n):
        t = f0 + i * step
        fi = int(math.floor(t))
        sc.frame_set(fi, subframe=float(t - fi))
        ev = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        M = ev.matrix_world

        def head(bn):
            p = M @ ev.pose.bones[bn].head
            return [p.x, p.y, p.z]

        def quat(bn):
            q = (M @ ev.pose.bones[bn].matrix).to_quaternion()
            return [q.w, q.x, q.y, q.z]

        rec = {"t": round((t - f0) / sc.render.fps, 6)}
        for leg, m in LEGS.items():
            rec[leg] = {"sy": head(m["sy"]), "sp": head(m["sp"]),
                        "kn": head(m["kn"]), "tip": head(m["tip"])}
        for k, bn in BODY.items():
            rec[k] = {"p": head(bn), "q": quat(bn)}
        frames.append(rec)

    # Ground estimate + contact heuristic: 5th percentile of all tip heights.
    tz = sorted(f[l]["tip"][2] for f in frames for l in LEGS)
    ground = tz[int(0.05 * len(tz))]
    thr = ground + 0.008 / 0.006866      # ~8mm expressed in rig units
    for f in frames:
        f["contact"] = {l: bool(f[l]["tip"][2] <= thr) for l in LEGS}

    out = {
        "source": bpy.path.basename(bpy.data.filepath),
        "rig": a.rig,
        "units": "raw blender units, unscaled",
        "axis_note": "rig is -Y forward, +Z up; character LEFT is +X. Stage 2 must convert to +X forward.",
        "leg_map": {k: v for k, v in LEGS.items()},
        "authored_fps": sc.render.fps,
        "sample_hz": a.hz,
        "frame_range": [f0, f1],
        "n_frames": len(frames),
        "rest_bone_lengths": rest,
        "ground_z": ground,
        "contact_threshold_z": thr,
        "frames": frames,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"[[ wrote {a.out}: {len(frames)} samples @ {a.hz}Hz "
          f"from frames {f0}-{f1} of {out['source']}")


main()
