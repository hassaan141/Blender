"""Generate a Blender rig that is a kinematic twin of the Bingo robot, from the URDF.

Everything the animator needs to get right is baked in by construction:
  - bones sit at the exact joint positions, at true real-world scale (metres)
  - each bone's local Z axis IS that motor's rotation axis
  - the other two rotation channels are locked, so a 1-DOF hinge cannot be
    animated as a ball joint
  - hard Limit Rotation constraints at the robot's real limits (with margin), so
    an unreachable pose is not authorable
  - the robot's own STL meshes are attached, so she is animating Bingo, not a proxy

Run:
    blender -b --factory-startup -P scripts/build_rig.py -- \
        --urdf bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf \
        --out blend/conform/Bingo_ConformRig.blend
"""

import bpy, sys, os, argparse, math
import xml.etree.ElementTree as ET
from mathutils import Vector, Matrix

SHANK_LEN = 0.120
BONE_LEN = 0.030          # cosmetic length for the hinge bones

# Author-time limits: the robot's real limits pulled in slightly, per spec B.1.6.
# Every value is additionally clamped by the URDF's own limit below, so a margin
# that is looser than the robot never widens the authorable range.
MARGIN = {"SY": 0.38, "SP": 1.40, "knee": 1.40,
          "head_pitch_joint": (-0.58, 0.36), "head_yaw": 0.54, "head_roll": 0.70,
          "tail_pitch": 0.54, "tail_yaw": 0.54,
          # Ears (v4+). The *_ear_pitch joints are declared `continuous` in the
          # URDF yet still carry a -3..3 limit; we honour the limit. The roll
          # joints are ASYMMETRIC and mirrored (left -1.5..0, right 0..1.5), so
          # they must be given as explicit tuples, not a symmetric scalar.
          "l_ear_pitch": 2.70, "r_ear_pitch": 2.70,
          "l_ear_roll": (-1.35, 0.0), "r_ear_roll": (0.0, 1.35)}

# Default standing pose. NOTE: the spec's A.3 crouch (SP -0.30 / knee +0.60)
# gives a base height of 0.1859 m, which is BELOW the 0.19-0.20 m stance height
# A.1 states -- the two numbers in the doc disagree. -0.25/+0.50 gives 0.1989 m,
# satisfying the stated stance height, and reads far less squat.
CROUCH = {"fl": (0.0, -0.25, 0.50), "fr": (0.0, 0.25, -0.50),
          "bl": (0.0, -0.25, 0.50), "br": (0.0, -0.25, -0.50)}


def rpy_to_mat(r, p, y):
    return (Matrix.Rotation(y, 3, 'Z') @ Matrix.Rotation(p, 3, 'Y') @ Matrix.Rotation(r, 3, 'X'))


def load_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin"); ax = j.find("axis"); lim = j.find("limit")
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
        joints[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=Vector(xyz), R=rpy_to_mat(*rpy),
            axis=Vector([float(v) for v in ax.get("xyz").split()]) if ax is not None else Vector((0, 0, 1)),
            lo=float(lim.get("lower")), hi=float(lim.get("upper")))
    meshes = {}
    for ln in root.findall("link"):
        v = ln.find("visual")
        m = v.find("geometry/mesh") if v is not None else None
        if m is not None:
            meshes[ln.get("name")] = m.get("filename")
    return joints, meshes


def resolve_mesh(rel, urdf_dir):
    """Locate a URDF mesh reference on disk.

    rev_3 writes plain relative paths ("../meshes/x.STL"); v4 writes ROS package
    URIs ("package://<pkg>/meshes/x.STL"). Both have to work, and the package
    name does not necessarily match the folder name on disk, so resolve a package
    URI against the real package root -- the nearest ancestor holding package.xml.
    """
    if not rel.startswith("package://"):
        return os.path.normpath(os.path.join(urdf_dir, rel))
    inner = rel[len("package://"):]
    sub = inner.split("/", 1)[1] if "/" in inner else inner   # drop the pkg name
    d = os.path.abspath(urdf_dir)
    while True:
        if os.path.exists(os.path.join(d, "package.xml")):
            return os.path.normpath(os.path.join(d, sub))
        parent = os.path.dirname(d)
        if parent == d:
            # no package.xml anywhere above: fall back to the URDF's parent dir
            return os.path.normpath(os.path.join(urdf_dir, "..", sub))
        d = parent


def link_frames(joints):
    """World transform of every link at the zero pose."""
    frames = {"origin": (Matrix.Identity(3), Vector((0, 0, 0)))}
    changed = True
    while changed:
        changed = False
        for name, j in joints.items():
            if j["child"] in frames or j["parent"] not in frames:
                continue
            R, p = frames[j["parent"]]
            frames[j["child"]] = (R @ j["R"], p + R @ j["xyz"])
            changed = True
    return frames


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-meshes", action="store_true")
    a = ap.parse_args(argv)

    urdf_dir = os.path.dirname(os.path.abspath(a.urdf))
    joints, meshes = load_urdf(a.urdf)
    frames = link_frames(joints)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.length_unit = 'CENTIMETERS'   # Bingo is small; cm reads better
    sc.render.fps = 60

    arm_data = bpy.data.armatures.new("Bingo_Robot")
    arm = bpy.data.objects.new("Bingo_Robot", arm_data)
    sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    # root carries the world transform of the base (spec B.1.4)
    root = eb.new("root")
    root.head, root.tail = Vector((0, 0, 0)), Vector((0.08, 0, 0))

    # Canonical joint order. The first 12 are the legs and MUST keep this order
    # and these indices -- the RL side addresses them positionally. Head, tail
    # and ears extend the list, so a 12-DOF consumer reading the first 12 entries
    # stays correct against a 21-DOF file.
    order = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
             "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
             "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
    # Ears only exist from URDF v4 onward; drop silently on older revisions so
    # this script still builds a correct rig from rev_3.
    missing = [jn for jn in order if jn not in joints]
    if missing:
        print(f"[[ note: URDF has no {missing} -- building without them")
        order = [jn for jn in order if jn in joints]
    bone_of_link = {"origin": "root"}
    misalign = []

    for jn in order:
        j = joints[jn]
        Rp, pp = frames[j["parent"]]
        head = pp + Rp @ j["xyz"]
        axis_w = (Rp @ j["axis"]).normalized()

        # Point the bone at its child joint, but PERPENDICULAR to the rotation
        # axis -- that is what lets align_roll put local Z exactly on the axis.
        kids = [k for k in joints.values() if k["parent"] == j["child"]]
        Rc, pc = frames[j["child"]]
        if kids:
            aim = (pc + Rc @ kids[0]["xyz"]) - head
        elif jn.endswith("knee"):
            aim = Rc @ Vector((0, 0, -SHANK_LEN))
        else:
            aim = Rc @ Vector((0.05, 0, 0))
        perp = aim - aim.dot(axis_w) * axis_w
        if perp.length < 1e-4:
            perp = axis_w.orthogonal()
        perp = (perp - perp.dot(axis_w) * axis_w).normalized()   # re-orthogonalise

        # Build the bone's orientation explicitly: local Y along the bone, local Z
        # ON the motor axis. Neither align_roll() nor setting .roll is reliable
        # here -- align_roll fails silently when the URDF offset is zero, and
        # EditBone.z_axis reads stale straight after .roll is assigned, so a
        # verify-then-fix loop reports success while leaving the bone wrong.
        y_ax = perp
        z_ax = axis_w.normalized()
        x_ax = y_ax.cross(z_ax).normalized()
        M = Matrix((x_ax, y_ax, z_ax)).transposed().to_4x4()
        M.translation = head

        b = eb.new(jn)
        b.head = head
        b.tail = head + perp * BONE_LEN
        b.matrix = M
        b.length = BONE_LEN
        b.parent = eb[bone_of_link[j["parent"]]]
        b.use_connect = False
        misalign.append((jn, axis_w))            # verified after leaving edit mode
        bone_of_link[j["child"]] = jn

    # foot tip markers -- zero DOF, they exist so contact/foot position is
    # readable without re-deriving FK (spec B.1.1)
    for leg in ("fl", "fr", "bl", "br"):
        Rk, pk = frames[f"{leg}_knee"]
        tip = pk + Rk @ Vector((0, 0, -SHANK_LEN))
        t = eb.new(f"{leg}_foot_tip")
        t.head, t.tail = tip, tip + Vector((0, 0, -0.02))
        t.parent = eb[f"{leg}_knee"]
        t.use_connect = False

    # Force every joint bone's direction perpendicular to its motor axis. Roll
    # only spins Z within the plane normal to the bone, so if the bone is not
    # perpendicular to the axis then NO roll can align them -- which is exactly
    # what went wrong on the two joints whose URDF offset is zero.
    for jn, axis_w in misalign:
        e = eb[jn]
        d = (e.tail - e.head)
        d = d - d.dot(axis_w) * axis_w
        if d.length < 1e-4:
            d = axis_w.orthogonal()
        e.tail = e.head + d.normalized() * BONE_LEN

    # ---- IK layer -----------------------------------------------------------
    # The joint bones are short stubs aligned to their motor axes, so they cannot
    # be IK end-effectors directly. Give each leg a zero-DOF "ik_end" bone whose
    # tail sits on the foot tip, and drive that. IK then rotates only SY/SP/knee,
    # and per-bone IK locks keep each of them a true 1-DOF hinge.
    for leg in ("fl", "fr", "bl", "br"):
        Rk, pk = frames[f"{leg}_knee"]
        tip = pk + Rk @ Vector((0, 0, -SHANK_LEN))
        e = eb.new(f"{leg}_ik_end")
        e.head, e.tail = pk, tip
        e.parent = eb[f"{leg}_knee"]
        e.use_connect = False

        c = eb.new(f"ctrl_{leg}_foot")
        c.head, c.tail = tip, tip + Vector((0, 0.05, 0))
        # Deliberately NOT parented to root. Foot controls must live in world
        # space: a planted paw has to stay put while the body travels over it.
        # Parenting them to root drags planted feet along and bakes in foot
        # sliding, which is exactly what the RL reference must not contain.
        c.parent = None
        c.use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # Edit-mode roll/matrix writes are not reliable (align_roll fails silently on
    # zero-offset joints, EditBone.z_axis reads stale, and EditBone.matrix does
    # not stick). Object-mode bone.matrix_local IS trustworthy, so correct the
    # roll across mode switches until every bone's Z sits on its motor axis.
    for _ in range(6):
        bad = {}
        for jn, axis_w in misalign:
            b = arm.data.bones[jn]
            R = b.matrix_local.to_3x3()
            z = (R @ Vector((0, 0, 1))).normalized()
            y = (R @ Vector((0, 1, 0))).normalized()
            ax_n = axis_w.normalized()
            d = math.atan2(y.dot(z.cross(ax_n)), z.dot(ax_n))
            if abs(d) > 1e-9:
                bad[jn] = d
        if not bad:
            break
        bpy.ops.object.mode_set(mode='EDIT')
        for jn, d in bad.items():
            arm_data.edit_bones[jn].roll -= d
        bpy.ops.object.mode_set(mode='OBJECT')

    # ---- lock to one DOF and clamp to the real limits -----------------------
    for jn in order:
        pb = arm.pose.bones[jn]
        pb.rotation_mode = 'XYZ'
        pb.lock_rotation = (True, True, False)     # local Z is the motor axis
        pb.lock_location = (True, True, True)
        pb.lock_scale = (True, True, True)
        c = pb.constraints.new('LIMIT_ROTATION')
        c.owner_space = 'LOCAL'
        c.use_limit_x = c.use_limit_y = True
        c.min_x = c.max_x = c.min_y = c.max_y = 0.0
        c.use_limit_z = True
        key = "SY" if "SY" in jn else ("knee" if "knee" in jn else ("SP" if "SP" in jn else jn))
        m = MARGIN[key]
        lo, hi = m if isinstance(m, tuple) else (-m, m)
        # never exceed what the URDF actually allows
        c.min_z, c.max_z = max(lo, joints[jn]["lo"]), min(hi, joints[jn]["hi"])
    for leg in ("fl", "fr", "bl", "br"):
        pb = arm.pose.bones[f"{leg}_foot_tip"]
        pb.lock_rotation = (True, True, True)
        pb.lock_location = (True, True, True)

    # ---- IK constraints and hinge locks -------------------------------------
    for leg in ("fl", "fr", "bl", "br"):
        # Every joint stays a true 1-DOF hinge under IK. That gives the chain
        # exactly 3 DOF (SY+SP+knee) for a 3-DOF target -- solvable. Dropping to
        # 2 DOF (excluding SY) is unsolvable and the solver simply gives up;
        # unlocking the axes "works" but produces poses off the hinge axis that
        # the robot physically cannot reach, which is worse than not solving.
        for jn in (f"{leg}_SY_J", f"{leg}_SP_J", f"{leg}_knee"):
            pb = arm.pose.bones[jn]
            pb.lock_ik_x = pb.lock_ik_y = True
            lim = next(c for c in pb.constraints if c.type == 'LIMIT_ROTATION')
            pb.use_ik_limit_z = True
            pb.ik_min_z, pb.ik_max_z = lim.min_z, lim.max_z
        # SY has only +/-24 deg and is not a locomotion joint, but the solver
        # will burn all of it on straight-line motion unless told to resist.
        # Also cap what the SOLVER may spend on SY at roughly the authority the
        # RL policy actually has (it scales SY actions by 0.3, so ~+/-0.126 rad).
        # Without this, IK spends 0.3 rad of SY on a straight-line walk and the
        # resulting reference is one the policy cannot track. Deliberate lateral
        # posing via FK still has the full +/-0.38.
        # Stiffness discourages the solver from spending SY, but do NOT hard-cap
        # it: SY is what lets IK hold a foot planted, and clamping it to the
        # policy's authority made foot drift jump from 3 mm to 29 mm. Better to
        # allow it and have the bake WARN when a clip leans on SY too hard.
        arm.pose.bones[f"{leg}_SY_J"].ik_stiffness_z = 0.95

        pb = arm.pose.bones[f"{leg}_ik_end"]
        pb.lock_ik_x = pb.lock_ik_y = pb.lock_ik_z = True   # carries no DOF itself
        ik = pb.constraints.new('IK')
        ik.target = arm
        ik.subtarget = f"ctrl_{leg}_foot"
        # ik_end + knee + SP only. SY is deliberately OUT of the IK chain: it is
        # the sideways joint with just +/-24 deg of range, and the solver will
        # happily burn all of it on a straight-line walk, pinning it at its limit
        # for a third of the clip. SY stays FK so lateral motion is intentional.
        ik.chain_count = 4
        ik.use_tail = True

        cb = arm.pose.bones[f"ctrl_{leg}_foot"]
        cb.lock_rotation = (True, True, True)
        cb.lock_scale = (True, True, True)

    # ---- attach the robot's own meshes --------------------------------------
    if not a.no_meshes:
        for link, rel in meshes.items():
            path = resolve_mesh(rel, urdf_dir)
            if path is None or not os.path.exists(path):
                print(f"[[ missing mesh {rel}"); continue
            before = set(bpy.data.objects)
            try:
                bpy.ops.wm.stl_import(filepath=path)
            except AttributeError:
                bpy.ops.import_mesh.stl(filepath=path)
            new = list(set(bpy.data.objects) - before)
            if not new:
                continue
            ob = new[0]; ob.name = f"mesh_{link}"
            R, p = frames[link]
            ob.matrix_world = Matrix.Translation(p) @ R.to_4x4()
            ob.parent = arm
            ob.parent_type = 'BONE'
            ob.parent_bone = bone_of_link[link]
            pbone = arm.pose.bones[bone_of_link[link]]
            # bone parenting is relative to the bone TAIL; undo that so the mesh
            # lands where the URDF says it does
            ob.matrix_parent_inverse = (
                arm.matrix_world @ Matrix.Translation(pbone.tail - pbone.head)
                @ pbone.matrix.to_4x4()).inverted()

    # ---- put it in the standing pose ----------------------------------------
    # With IK active the joint bones are driven by the foot controls, so the pose
    # has to be expressed by MOVING THE CONTROLS -- keying the joints would just
    # be overridden by the solver.
    for leg, (sy, sp, kn) in CROUCH.items():
        R0, p0 = frames[f"{leg}_knee"]
        rest_tip = p0 + R0 @ Vector((0, 0, -SHANK_LEN))
        # where that foot ends up in the standing pose, from URDF FK
        T_R, T_p = Matrix.Identity(3), Vector((0, 0, 0))
        for jn, val in ((f"{leg}_SY_J", sy), (f"{leg}_SP_J", sp), (f"{leg}_knee", kn)):
            J = joints[jn]
            T_p = T_p + T_R @ J["xyz"]
            ang = val
            ax = J["axis"].normalized()
            T_R = T_R @ J["R"] @ Matrix.Rotation(ang, 3, ax)
        want_tip = T_p + T_R @ Vector((0, 0, -SHANK_LEN))
        cb = arm.pose.bones[f"ctrl_{leg}_foot"]
        cb.location = cb.bone.matrix_local.to_3x3().inverted() @ (want_tip - rest_tip)

    # Stand it on the floor. Bones are built in the URDF's own frame, where the
    # base is the origin, so without this the robot is buried to its belly.
    # Offset the OBJECT (not the bones) so joint values stay exactly URDF values.
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = arm.evaluated_get(dg)
    tip_z = min((arm.matrix_world @ ev.pose.bones[f"{l}_foot_tip"].head).z
                for l in ("fl", "fr", "bl", "br"))
    PAW_DROP = 0.0288          # paw mesh hangs this far below the shank tip
    arm.location.z = -(tip_z - PAW_DROP)
    bpy.context.view_layer.update()
    print(f"[[ stance: base raised to {arm.location.z:.4f} m so the paws rest on z=0")

    # Verify from the FINAL bone matrices, not from edit-mode values, which are
    # stale and previously reported success on bones that were 30 deg wrong.
    worst, wname = 0.0, ""
    for jn, axis_w in misalign:
        z = arm.data.bones[jn].matrix_local.to_3x3() @ Vector((0, 0, 1))
        d = math.degrees(z.normalized().angle(axis_w.normalized()))
        if d > worst:
            worst, wname = d, jn
    print(f"[[ axis alignment: worst {wname} off by {worst:.4f} deg "
          f"(0 = bone Z is exactly the motor axis)")
    if worst > 0.01:
        raise SystemExit(f"ABORT: {wname} axis is {worst:.3f} deg off; rig not written")
    print(f"[[ bones: {len(arm_data.bones)}  meshes: {len([o for o in bpy.data.objects if o.type=='MESH'])}")

    for jn, axis_w in misalign:
        z = arm.data.bones[jn].matrix_local.to_3x3() @ Vector((0, 0, 1))
        d = math.degrees(z.normalized().angle(axis_w.normalized()))
        if d > 0.01:
            print(f"[[ PRESAVE-BAD {jn} {d:.3f} z={tuple(round(v,3) for v in z)} axis={tuple(axis_w)}")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    print(f"[[ wrote {a.out}")


main()
