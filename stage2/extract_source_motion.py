"""Stage 2.1 - extract Ashley's EVALUATED world-space motion from Bingo_Cheeky.blend.

We never read raw control values. Every keypoint is the evaluated world-space
position/orientation of a DEFORM bone (the def_* skeleton that the controls drive
through IK/constraints), sampled through the dependency graph frame by frame.

Ashley's rig is a biped-style quadruped:
  front legs = Arm/ForeArm/Hand      back legs = Leg/Shin/Foot
  .L / .R are the rig's mirror labels (anatomy is resolved later by Kabsch).

For each Ashley leg we capture the complete semantic chain separately:
SY -> SP -> knee -> ankle/paw-bone HEAD -> toe/contact.  The ankle is the
primary v4 shank-tip IK target; the toe is used only as a secondary rigid-paw
contact target.  Body/head/tail/ear transforms are saved alongside it.

Run:
  blender -b blend_sources/Bingo_Cheeky.blend -P stage2/extract_source_motion.py -- \
      --out stage2/out/cheeky_source_keypoints.npz
"""
import bpy, sys, argparse
import numpy as np

# Ashley leg -> (SY bone, SP bone, knee bone, paw bone)
ALEGS = ["aFL", "aFR", "aBL", "aBR"]
LEGDEF = {
    "aFL": ("def_Shoulder.L", "def_Arm.L", "def_ForeArm.L", "def_Hand.L"),
    "aFR": ("def_Shoulder.R", "def_Arm.R", "def_ForeArm.R", "def_Hand.R"),
    "aBL": ("def_Hip.L",      "def_Leg.L", "def_Shin.L",    "def_Foot.L"),
    "aBR": ("def_Hip.R",      "def_Leg.R", "def_Shin.R",    "def_Foot.R"),
}
BODY = "def_Pelvis"
HEAD = "def_Head"
TAIL = "def_Tail.001"
# The visible ear meshes are bone-parented to the Anim_Ear children, not to the
# def_Ear parents.  Sampling def_Ear silently drops the large animator-authored
# local ear rotation.  The artist mesh object names are crossed, but the bone
# anatomy remains .L -> left and .R -> right after the single reflection in the
# solver; do not swap these a second time based on mesh names.
EARL = "Anim_Ear.L"
EARR = "Anim_Ear.R"
EARL_DEF = "def_Ear.L"
EARR_DEF = "def_Ear.R"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default="Bingo_Rig")
    a = ap.parse_args(argv)

    sc = bpy.context.scene
    ob = bpy.data.objects[a.rig]
    fps = sc.render.fps / sc.render.fps_base
    f0, f1 = sc.frame_start, sc.frame_end
    T = f1 - f0 + 1
    frames = np.arange(f0, f1 + 1)

    def qof(M):
        q = M.to_quaternion()
        return np.array([q.w, q.x, q.y, q.z])

    sy = {l: np.zeros((T, 3)) for l in ALEGS}
    sp = {l: np.zeros((T, 3)) for l in ALEGS}
    knee = {l: np.zeros((T, 3)) for l in ALEGS}
    ankle = {l: np.zeros((T, 3)) for l in ALEGS}
    toe = {l: np.zeros((T, 3)) for l in ALEGS}
    body_pos = np.zeros((T, 3)); body_quat = np.zeros((T, 4))
    head_q = np.zeros((T, 4)); tail_q = np.zeros((T, 4))
    earl_q = np.zeros((T, 4)); earr_q = np.zeros((T, 4))
    earl_def_q = np.zeros((T, 4)); earr_def_q = np.zeros((T, 4))
    head_p = np.zeros((T, 3)); tail_p = np.zeros((T, 3))

    for i, fr in enumerate(frames):
        sc.frame_set(int(fr))
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        M = ob.matrix_world
        pb = ev.pose.bones
        for l in ALEGS:
            syn, spn, kn, pn = LEGDEF[l]
            sy[l][i] = M @ pb[syn].head
            sp[l][i] = M @ pb[spn].head
            knee[l][i] = M @ pb[kn].head
            ankle[l][i] = M @ pb[pn].head
            toe[l][i] = M @ pb[pn].tail
        body_pos[i] = M @ pb[BODY].head
        body_quat[i] = qof(M @ pb[BODY].matrix)
        head_p[i] = M @ pb[HEAD].head
        head_q[i] = qof(M @ pb[HEAD].matrix)
        tail_p[i] = M @ pb[TAIL].head
        tail_q[i] = qof(M @ pb[TAIL].matrix)
        earl_q[i] = qof(M @ pb[EARL].matrix)
        earr_q[i] = qof(M @ pb[EARR].matrix)
        earl_def_q[i] = qof(M @ pb[EARL_DEF].matrix)
        earr_def_q[i] = qof(M @ pb[EARR_DEF].matrix)

    # Bone-data lengths are pose independent.  Do not infer morphology scale
    # from frame 1: Ashley's paw is animated and frame 1 is not an armature rest.
    rest = {}
    sy_sp_len = {}
    paw_len = {}
    for l in ALEGS:
        syn, spn, kn, pn = LEGDEF[l]
        sy_sp_len[l] = float(ob.data.bones[syn].length)
        rest[l] = (float(ob.data.bones[spn].length), float(ob.data.bones[kn].length))
        paw_len[l] = float(ob.data.bones[pn].length)

    out = dict(
        fps=np.array(fps), frames=frames.astype(np.int32),
        body_pos=body_pos.astype(np.float32), body_quat=body_quat.astype(np.float32),
        head_pos=head_p.astype(np.float32), head_quat=head_q.astype(np.float32),
        tail_pos=tail_p.astype(np.float32), tail_quat=tail_q.astype(np.float32),
        earl_quat=earl_q.astype(np.float32), earr_quat=earr_q.astype(np.float32),
        earl_def_quat=earl_def_q.astype(np.float32),
        earr_def_quat=earr_def_q.astype(np.float32),
        ear_source_bones=np.array([EARL, EARR]),
        aleg_order=np.array(ALEGS),
        rest_lengths=np.array([rest[l] for l in ALEGS], np.float32),
        sy_sp_lengths=np.array([sy_sp_len[l] for l in ALEGS], np.float32),
        paw_lengths=np.array([paw_len[l] for l in ALEGS], np.float32),
        source=np.array(bpy.path.basename(bpy.data.filepath)),
    )
    for l in ALEGS:
        out[f"sy_{l}"] = sy[l].astype(np.float32)
        out[f"sp_{l}"] = sp[l].astype(np.float32)
        out[f"knee_{l}"] = knee[l].astype(np.float32)
        out[f"ankle_{l}"] = ankle[l].astype(np.float32)
        out[f"toe_{l}"] = toe[l].astype(np.float32)
        # Compatibility aliases for diagnostic consumers not yet migrated.
        out[f"hip_{l}"] = sp[l].astype(np.float32)
        out[f"paw_{l}"] = toe[l].astype(np.float32)
    np.savez(a.out, **out)

    # report
    alltoe = np.concatenate([toe[l] for l in ALEGS], 0)
    print(f"[[ extracted {T} frames @ {fps:.1f} fps  ({T/fps:.2f} s)  from {out['source']}")
    print(f"[[ body height (pelvis z) mean {body_pos[:,2].mean():.2f}  range "
          f"[{body_pos[:,2].min():.2f}, {body_pos[:,2].max():.2f}] (source units)")
    print(f"[[ toe/contact z floor (min over clip) {alltoe[:,2].min():.3f}  units")
    for l in ALEGS:
        thigh, shank = rest[l]
        print(f"[[ {l}: SY-SP {sy_sp_len[l]:.2f}  thigh {thigh:.2f}  "
              f"shank {shank:.2f}  paw {paw_len[l]:.2f} (units)  "
              f"toe z [{toe[l][:,2].min():.2f},{toe[l][:,2].max():.2f}]")
    print(f"[[ visible ears sampled from {EARL}/{EARR} (def_Ear parents also saved for diagnostics)")
    print(f"[[ wrote {a.out}")
    sys.stdout.flush()
    import os; os._exit(0)


main()
