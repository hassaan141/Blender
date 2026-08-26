"""Bake an animation authored on the conform rig straight to a Bingo motion .npz.

Because the rig IS the robot, there is no retargeting step: each bone's rotation
about its own local Z is already that joint's angle in the URDF's convention.
This reads them directly, so nothing is approximated.

    blender -b blend/conform/<animated.blend> -P scripts/bake_conform.py -- \
        --out motions/clip.npz [--hz 120]

Works with IK: joint angles are recovered from the EVALUATED pose matrices, not
from rotation_euler (which IK does not write to).
"""
import bpy, sys, os, argparse, math
import numpy as np
from mathutils import Vector, Matrix

LEGS = ("fl", "fr", "bl", "br")
DOF_ORDER = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee"]
HEAD_TAIL = ["head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw"]
EARS = ["l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]

# The 21-DOF schema is a strict EXTENSION of the 12-DOF one: indices 0-11 are
# unchanged, so a 12-DOF consumer that slices the first 12 entries stays correct
# against a 21-DOF file. Expression lives entirely in indices 12-20 -- 9 DOF the
# 12-DOF schema discards (spec B.5.6).
DOF_ORDER_21 = DOF_ORDER + HEAD_TAIL + EARS

BODY_NAMES = ["origin", "fl_knee", "fr_knee", "bl_knee", "br_knee"]
PAW_DROP = 0.0288


def joint_angle(ev, name):
    """Rotation of a bone about its own local Z, in radians.

    Valid under IK: derived from evaluated pose matrices rather than
    rotation_euler, which IK leaves untouched.
    """
    pb = ev.pose.bones[name]
    rest = pb.bone.matrix_local
    if pb.parent:
        base = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ rest
    else:
        base = rest
    L = base.inverted() @ pb.matrix
    e = L.to_euler('XYZ')
    return e.z


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default="Bingo_Robot")
    ap.add_argument("--hz", type=float, default=120.0)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--dof", type=int, choices=(12, 21), default=12,
                    help="12 = legs only (legacy, default). 21 = legs + head + "
                         "tail + ears, the expression schema (spec B.5.6).")
    a = ap.parse_args(argv)

    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    f0 = a.start if a.start is not None else sc.frame_start
    f1 = a.end if a.end is not None else sc.frame_end
    step = sc.render.fps / a.hz
    n = int(round((f1 - f0) / step)) + 1

    # Which expression joints this rig actually has. A conform rig built from
    # rev_3 has no ear bones; bake what exists rather than failing, so old files
    # (Ashley's walk cycle, for one) still bake.
    have = {jn for jn in HEAD_TAIL + EARS if jn in rig.pose.bones}
    absent = [jn for jn in HEAD_TAIL + EARS if jn not in have]
    if absent:
        print(f"[[ note: rig has no {absent} -- those channels bake as zeros")
    names = DOF_ORDER_21 if a.dof == 21 else DOF_ORDER
    ndof = len(names)

    dof = np.zeros((n, ndof)); ht = np.zeros((n, 5)); ears = np.zeros((n, 4))
    body_pos = np.zeros((n, 5, 3)); body_rot = np.zeros((n, 5, 4))
    tips = np.zeros((n, 4, 3))

    for i in range(n):
        t = f0 + i * step
        fi = int(math.floor(t))
        sc.frame_set(fi, subframe=float(t - fi))
        ev = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        M = rig.matrix_world

        for k, jn in enumerate(DOF_ORDER):
            dof[i, k] = joint_angle(ev, jn)
        for k, jn in enumerate(HEAD_TAIL):
            ht[i, k] = joint_angle(ev, jn) if jn in have else 0.0
        for k, jn in enumerate(EARS):
            ears[i, k] = joint_angle(ev, jn) if jn in have else 0.0
        if a.dof == 21:
            dof[i, 12:17] = ht[i]
            dof[i, 17:21] = ears[i]

        # Root/base pose. A Blender bone's local Y runs ALONG the bone, so the
        # root bone's own frame is not the robot's `origin` frame -- strip the
        # rest orientation out to recover it, or every rotation is ~90 deg off.
        rb = ev.pose.bones["root"]
        body_pos[i, 0] = M @ rb.head
        R_origin = (M.to_3x3() @ rb.matrix.to_3x3()
                    @ rb.bone.matrix_local.to_3x3().inverted())
        q = R_origin.to_quaternion()
        body_rot[i, 0] = [q.w, q.x, q.y, q.z]

        for k, l in enumerate(LEGS):
            tip_name = f"{l}_foot_tip"
            if tip_name in ev.pose.bones:
                tips[i, k] = M @ ev.pose.bones[tip_name].head
            else:
                # Final baked Stage 2 files intentionally remove helper bones.
                # Reconstruct the same contact point from metadata stored in the
                # armature and the evaluated physical knee transform.
                key = f"{l}_foot_tip_local"
                if key not in rig:
                    raise KeyError(f"missing both {tip_name} and armature metadata {key}")
                tips[i, k] = M @ (ev.pose.bones[f"{l}_knee"].matrix @ Vector(rig[key]))
            body_pos[i, k + 1] = tips[i, k]
            qk = (M @ ev.pose.bones[f"{l}_knee"].matrix).to_quaternion()
            body_rot[i, k + 1] = [qk.w, qk.x, qk.y, qk.z]

    dt = 1.0 / a.hz
    dof_vel = np.gradient(dof, dt, axis=0)
    body_lin = np.gradient(body_pos, dt, axis=0)
    body_ang = np.zeros((n, 5, 3))

    # contact: paw surface within 5 mm of the floor and not moving much
    paw_z = tips[:, :, 2] - PAW_DROP
    speed = np.linalg.norm(np.gradient(tips, dt, axis=0), axis=2)
    contacts = (paw_z < 0.005) & (speed < 0.05)

    np.savez(a.out,
             fps=np.array(a.hz),
             dof_names=np.array(names), body_names=np.array(BODY_NAMES),
             dof_positions=dof.astype(np.float32),
             dof_velocities=dof_vel.astype(np.float32),
             body_positions=body_pos.astype(np.float32),
             body_rotations=body_rot.astype(np.float32),
             body_linear_velocities=body_lin.astype(np.float32),
             body_angular_velocities=body_ang.astype(np.float32),
             contacts=contacts, head_tail_positions=ht.astype(np.float32),
             ear_positions=ears.astype(np.float32),
             source=np.array(bpy.path.basename(bpy.data.filepath)))

    print(f"[[ frames {n} @ {a.hz}Hz ({n/a.hz:.2f}s)  schema {ndof} DOF")
    # Legs and expression channels have DIFFERENT velocity limits (10 vs 8 rad/s,
    # spec A.3), so reporting one number against one limit hides violations.
    leg_v = np.abs(dof_vel[:, :12])
    print(f"[[ legs: range rad min {dof[:, :12].min():+.3f} max {dof[:, :12].max():+.3f}"
          f" | max |vel| {leg_v.max():.2f} rad/s (limit 10)"
          f" frames over: {(leg_v.max(1) > 10).sum()}")
    if a.dof == 21:
        ex_v = np.abs(dof_vel[:, 12:])
        print(f"[[ expression (head/tail/ears): range rad min {dof[:, 12:].min():+.3f}"
              f" max {dof[:, 12:].max():+.3f}"
              f" | max |vel| {ex_v.max():.2f} rad/s (limit 8)"
              f" frames over: {(ex_v.max(1) > 8).sum()}")
        live = [names[12 + k] for k in range(ndof - 12)
                if np.abs(dof[:, 12 + k]).max() > 1e-4]
        print(f"[[ expression channels carrying motion: {live if live else 'NONE'}")
    print(f"[[ root height mean {body_pos[:,0,2].mean():.4f} m  "
          f"min paw {paw_z.min()*1000:+.1f} mm")
    print(f"[[ duty factor {dict(zip(LEGS, contacts.mean(0).round(2)))}")
    # The RL policy scales SY actions by 0.3, so it can only realise about
    # +/-0.126 rad of abduction however the reference is authored.
    sy = np.abs(dof[:, [0, 3, 6, 9]]).max()
    if sy > 0.126:
        print(f"[[ WARNING max |SY| {sy:.3f} rad exceeds the policy's effective "
              f"authority (~0.126 rad). This clip may be untrackable laterally; "
              f"consider straighter foot paths or a lower stance.")
    print(f"[[ wrote {a.out}")


main()
