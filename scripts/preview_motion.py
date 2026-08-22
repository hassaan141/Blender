"""Play a motion .npz back ON the conform rig, in Blender, and render it.

This is the missing visual check between `retarget.py` / `bake_conform.py` and
Isaac Sim: it shows what the robot has actually been asked to do, on the real
robot geometry, without needing a GPU or a sim install.

    blender -b blend/conform/Bingo_ConformRig_v4.blend -P scripts/preview_motion.py -- \
        --motion motions/clip.npz --out output/video/clip --render

Without --render it just applies the motion and saves a .blend you can scrub.

IK is MUTED on purpose. The rig drives its legs from foot controls, but a motion
file stores joint angles, so replaying it has to be pure FK -- otherwise the IK
solver overrides the very numbers we are trying to inspect.
"""
import bpy, sys, os, argparse
import numpy as np
from mathutils import Matrix, Quaternion, Vector

LEGS = ("fl", "fr", "bl", "br")
HEAD_TAIL = ["head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw"]
EARS = ["l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--out", required=True, help="output basename (no extension)")
    p.add_argument("--rig", default="Bingo_Robot")
    p.add_argument("--render", action="store_true", help="render an mp4 as well")
    p.add_argument("--fps", type=int, default=30, help="playback/render fps")
    p.add_argument("--max_seconds", type=float, default=None)
    p.add_argument("--res", type=int, default=720)
    return p.parse_args(argv)


def main():
    a = parse_args()
    d = np.load(a.motion, allow_pickle=True)
    names = [str(x) for x in d["dof_names"]]
    q = d["dof_positions"]
    src_fps = float(d["fps"])
    bpos, brot = d["body_positions"], d["body_rotations"]

    # Expression channels live either in dof (21-DOF files) or in their own keys.
    extra = {}
    for i, n in enumerate(names[12:], start=12):
        extra[n] = q[:, i]
    if "head_tail_positions" in d.files:
        for k, n in enumerate(HEAD_TAIL):
            extra.setdefault(n, d["head_tail_positions"][:, k])
    if "ear_positions" in d.files:
        for k, n in enumerate(EARS):
            extra.setdefault(n, d["ear_positions"][:, k])

    T = len(q)
    if a.max_seconds:
        T = min(T, int(a.max_seconds * src_fps))
    step = max(1, int(round(src_fps / a.fps)))     # decimate to playback rate
    frames = list(range(0, T, step))

    rig = bpy.data.objects[a.rig]
    sc = bpy.context.scene
    sc.render.fps = a.fps
    sc.frame_start, sc.frame_end = 1, len(frames)

    # Pure FK playback: the solver must not touch what we are inspecting.
    n_muted = 0
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type in ('IK', 'LIMIT_ROTATION'):
                c.mute = True
                n_muted += 1
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.lock_rotation = (False, False, False)
        pb.lock_location = (False, False, False)
    print(f"[[ muted {n_muted} IK/limit constraints for FK playback")

    root = rig.pose.bones["root"]
    rest3 = root.bone.matrix_local.to_3x3()
    M = rig.matrix_world
    Minv3 = M.to_3x3().inverted()

    present = [n for n in names[:12] + list(extra) if n in rig.pose.bones]
    missing = [n for n in names[:12] + list(extra) if n not in rig.pose.bones]
    if missing:
        print(f"[[ note: rig has no {missing} -- not driven")

    # Camera is keyframed to follow the root. The clips translate (DeadPan moves
    # 0.42 m), so a fixed camera loses the robot; a follow cam keeps it framed
    # while the ground grid still shows that it is travelling.
    cam_d = bpy.data.cameras.new("preview_cam")
    cam = bpy.data.objects.new("preview_cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    CAM_OFF = Vector((0.42, -0.62, 0.30))     # three-quarter front-left, robot ~0.3 m long

    def look_at(eye, tgt):
        d = (eye - tgt).normalized()
        right = Vector((0, 0, 1)).cross(d)
        if right.length < 1e-6:
            right = Vector((1, 0, 0))
        right.normalize()
        up = d.cross(right)
        return Matrix((right, up, d)).transposed().to_euler('XYZ')

    for out_f, i in enumerate(frames, start=1):
        sc.frame_set(out_f)
        # legs (and any expression channel the rig actually has)
        for k, n in enumerate(names[:12]):
            if n in rig.pose.bones:
                pb = rig.pose.bones[n]
                pb.rotation_euler = (0.0, 0.0, float(q[i, k]))
                pb.keyframe_insert("rotation_euler", frame=out_f)
        for n, series in extra.items():
            if n in rig.pose.bones:
                pb = rig.pose.bones[n]
                pb.rotation_euler = (0.0, 0.0, float(series[i]))
                pb.keyframe_insert("rotation_euler", frame=out_f)
        # root: invert exactly what bake_conform.py recorded
        w = Quaternion([float(x) for x in brot[i, 0]]).to_matrix()
        obj_R = Minv3 @ w @ rest3
        obj_p = M.inverted() @ Vector([float(x) for x in bpos[i, 0]])
        root.matrix = Matrix.LocRotScale(obj_p, obj_R.to_quaternion(), Vector((1, 1, 1)))
        root.keyframe_insert("location", frame=out_f)
        root.rotation_euler = obj_R.to_euler('XYZ')
        root.keyframe_insert("rotation_euler", frame=out_f)

        tgt = Vector([float(x) for x in bpos[i, 0]])
        cam.location = tgt + CAM_OFF
        cam.rotation_euler = look_at(cam.location, tgt)
        cam.keyframe_insert("location", frame=out_f)
        cam.keyframe_insert("rotation_euler", frame=out_f)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    blend_out = a.out + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend_out))
    print(f"[[ {len(frames)} frames applied @ {a.fps} fps -> {blend_out}")

    if a.render:
        sc.render.engine = 'BLENDER_WORKBENCH'
        sc.render.resolution_x = int(a.res * 16 / 9)
        sc.render.resolution_y = a.res
        sc.render.resolution_percentage = 100
        # PNG frames, stitched by ffmpeg afterwards. Blender's built-in FFMPEG
        # writer is not offered by every build (5.2 dropped it from this enum),
        # and MEMORY.md section 7 already documents encoder trouble on the RL box --
        # frames + an explicit ffmpeg call is the portable path.
        sc.render.image_settings.file_format = 'PNG'
        sc.render.filepath = os.path.abspath(a.out) + "_f"
        try:                       # show the floor so ground contact is visible
            sc.display.shading.show_shadows = True
            sc.display.shading.show_cavity = True
        except Exception:
            pass
        bpy.ops.render.render(animation=True)
        print(f"[[ rendered frames {sc.render.filepath}####.png")
        print(f"[[ stitch with: ffmpeg -framerate {a.fps} -i "
              f"'{sc.render.filepath}%04d.png' -c:v libx264 -profile:v baseline "
              f"-pix_fmt yuv420p -movflags +faststart '{os.path.abspath(a.out)}.mp4'")


main()
