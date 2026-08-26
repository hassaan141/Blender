"""Stage 2.10 - bake the solved root + 21 joint trajectories onto the EXACT v4
physical skeleton, and save Bingo_Cheeky_V4_Retargeted.blend.

The result depends only on the physical bones (root + 21 joints), not on Ashley's
rig and not on the animator controls. Controls are left at rest with IK off, so the
physical FK joints drive the meshes directly. Angles are written about each joint
bone's local Z (its motor axis, per check_rig) - the exact inverse of the reader in
bake_conform.joint_angle, so a re-bake round-trips.

Run:
  blender -b blend_sources/Bingo_V4_AnimatorRig.blend -P stage2/bake_v4_motion.py -- \
      --motion stage2/out/cheeky_v4_retarget.npz \
      --out blend_sources/Bingo_Cheeky_V4_Retargeted.blend
"""
import bpy, sys, argparse
import numpy as np
from mathutils import Matrix, Quaternion, Vector

DOF_ORDER = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
             "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
             "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
LEGS = ("fl", "fr", "bl", "br")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default="Bingo_Robot")
    a = ap.parse_args(argv)

    m = np.load(a.motion, allow_pickle=True)
    fps = int(round(float(m["fps"])))
    dof = m["dof_positions"].astype(float)
    root_pos = m["root_pos"].astype(float)
    root_quat = m["root_quat"].astype(float)   # wxyz
    T = dof.shape[0]

    arm = bpy.data.objects[a.rig]
    bpy.context.view_layer.objects.active = arm
    sc = bpy.context.scene
    sc.render.fps = fps; sc.render.fps_base = 1.0
    sc.frame_start = 1; sc.frame_end = T

    bpy.ops.object.mode_set(mode='POSE')
    pbs = arm.pose.bones

    # controls at rest, IK off, feet in root space (defaults) so FK drives meshes
    root_ctrl = pbs.get("ctrl_Root")
    if root_ctrl is not None:
        for leg in LEGS:
            if f"ik_{leg}" in root_ctrl:
                root_ctrl[f"ik_{leg}"] = 0.0
            if f"footroot_{leg}" in root_ctrl:
                root_ctrl[f"footroot_{leg}"] = 1.0
    for cn in ("ctrl_Root", "ctrl_Body", *(f"ctrl_{l}_foot" for l in LEGS),
               "ctrl_head", "ctrl_tail", "ctrl_l_ear", "ctrl_r_ear"):
        if cn in pbs:
            pbs[cn].location = (0, 0, 0)
            pbs[cn].rotation_quaternion = (1, 0, 0, 0)
            pbs[cn].rotation_euler = (0, 0, 0)

    # Mute the COPY_ROTATION constraints that slave the head/tail/ear joints to
    # their controls; otherwise the (at-rest) controls force those joints to zero
    # and overwrite the baked expression. After muting, the physical joints carry
    # the motion directly (the spec's requirement). Leg LIMIT_ROTATION/IK are kept
    # (IK is already off via ik_<leg>=0).
    muted = 0
    for jn in DOF_ORDER:
        pbs[jn].rotation_mode = 'XYZ'
        for con in pbs[jn].constraints:
            if con.type == 'COPY_ROTATION':
                con.mute = True; muted += 1
    print(f"[[ muted {muted} COPY_ROTATION constraints so physical joints carry expression")
    root = pbs["root"]
    root.rotation_mode = 'QUATERNION'

    M = arm.matrix_world.copy()
    M3 = M.to_3x3()
    M3inv = M3.inverted()
    Minv = M.inverted()
    root_ml3 = root.bone.matrix_local.to_3x3()

    def quat_to_R(q):
        return Quaternion((q[0], q[1], q[2], q[3])).to_matrix()

    for i in range(T):
        f = i + 1
        sc.frame_set(f)
        # joints: rotation about local Z = motor axis
        for k, jn in enumerate(DOF_ORDER):
            pb = pbs[jn]
            pb.rotation_euler = (0.0, 0.0, float(dof[i, k]))
            pb.keyframe_insert("rotation_euler", frame=f)
        # root: build the pose-bone armature-space matrix that yields the desired
        # world origin pose, inverting bake_conform's reader exactly.
        Rw = quat_to_R(root_quat[i])                    # desired origin orient (world)
        R_arm = M3inv @ Rw @ root_ml3                   # -> pose bone 3x3 (armature)
        t_arm = Minv @ Vector(root_pos[i])              # head in armature space
        mat = R_arm.to_4x4()
        mat.translation = t_arm
        root.matrix = mat
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_quaternion", frame=f)

    bpy.ops.object.mode_set(mode='OBJECT')
    # Final Stage 2 playback is a robot-only deliverable.  The animation is now
    # fully keyed on root + 21 physical joints, so remove every animator control,
    # IK end-effector, and foot-tip marker instead of merely hiding them.  This
    # prevents the rest-position helpers from obscuring the moving robot.
    physical = {"root", *DOF_ORDER}
    # Preserve each contact-point offset as armature metadata so export tools can
    # reconstruct paw positions without visible foot-tip bones.
    for leg in LEGS:
        knee = arm.data.bones[f"{leg}_knee"]
        tip_name = f"{leg}_foot_tip"
        key = f"{leg}_foot_tip_local"
        if tip_name in arm.data.bones:
            tip = arm.data.bones[tip_name]
            local = knee.matrix_local.inverted() @ tip.head_local
            arm[key] = list(local)
        elif key not in arm:
            raise KeyError(f"missing both {tip_name} and armature metadata {key}")

    # Muted expressive-control constraints are dead after baking. Remove them
    # before their target bones so Blender's dependency graph has no stale refs.
    for jn in DOF_ORDER:
        pb = arm.pose.bones[jn]
        for con in list(pb.constraints):
            if con.type == 'COPY_ROTATION':
                pb.constraints.remove(con)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    removed = [b.name for b in arm.data.edit_bones if b.name not in physical]
    for name in removed:
        arm.data.edit_bones.remove(arm.data.edit_bones[name])
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.show_in_front = False
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print(f"[[ baked {T} frames @ {fps} fps onto physical skeleton of '{a.rig}'")
    print(f"[[ removed {len(removed)} nonphysical helper/control bones: {removed}")
    print(f"[[ wrote {a.out}")
    sys.stdout.flush()
    import os; os._exit(0)


main()
