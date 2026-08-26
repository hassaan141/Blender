"""Add an animator control layer on top of the exact v4 physical robot rig.

Two clearly separated layers (spec):
    ANIMATOR CONTROLS  ->  constraints / IK  ->  V4 PHYSICAL BONES  ->  meshes  ->  bake -> Isaac

The physical v4 skeleton (fl_SY_J ... ears, root, meshes) is NOT modified: same names,
origins, axes, zero pose, limits, link lengths, 21 DOF. Only a control layer is added and
wired to drive the physical bones through constraints, so the final physical joint angles can
be baked and exported unchanged.

Controls added:
  ctrl_Root   master  - moves the whole dog (torso + feet-in-root + head/tail/ears)
  ctrl_Body   torso   - moves the base relative to planted feet (crouch/lean)
  ctrl_<leg>_foot      - 4 foot IK targets (already present; rewired to be switchable)
  ctrl_head/ctrl_tail/ctrl_l_ear/ctrl_r_ear - expressive controls

Per-leg switches (custom props on ctrl_Root, wired by drivers):
  ik_<leg>      0 = FK (physical joints animate directly)   1 = IK (foot control drives leg)
  footroot_<leg> 1 = foot follows ctrl_Root (ROOT/character space)  0 = foot fixed in WORLD

Run:
    blender -b <physical_rig>.blend -P scripts/add_animator_controls.py -- --out <animator_rig>.blend
"""
import bpy, sys, argparse
from mathutils import Vector, Matrix

LEGS = ("fl", "fr", "bl", "br")
HEAD_CTRL = {"ctrl_head": ["head_pitch_joint", "head_yaw", "head_roll"],
             "ctrl_tail": ["tail_pitch", "tail_yaw"],
             "ctrl_l_ear": ["l_ear_pitch", "l_ear_roll"],
             "ctrl_r_ear": ["r_ear_pitch", "r_ear_roll"]}


def eb_new(eb, name, head, tail, parent=None):
    b = eb.new(name); b.head, b.tail = Vector(head), Vector(tail)
    b.use_connect = False
    if parent: b.parent = parent
    return b


def add_prop(pb, name, val):
    pb[name] = float(val)
    # ensure UI range
    try:
        ui = pb.id_properties_ui(name); ui.update(min=0.0, max=1.0)
    except Exception:
        pass


def drive_influence(owner_obj, constraint, arm_name, prop_bone, prop_name):
    """Drive constraint.influence from ctrl bone custom property (0..1)."""
    fc = constraint.driver_add("influence"); d = fc.driver; d.type = 'SCRIPTED'
    v = d.variables.new(); v.name = "v"; v.type = 'SINGLE_PROP'
    tg = v.targets[0]; tg.id = owner_obj
    tg.data_path = f'pose.bones["{prop_bone}"]["{prop_name}"]'
    d.expression = "v"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)

    arm = bpy.data.objects["Bingo_Robot"]
    bpy.context.view_layer.objects.active = arm

    # world-space rest positions we need (object mode, from bone matrices)
    def bpos(name):  # bone head world
        b = arm.data.bones[name]
        return (arm.matrix_world @ b.matrix_local).translation.copy()
    base_head = bpos("root")

    # ---------------------------------------------------------------- edit bones
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones

    # master + body
    ctrl_root = eb_new(eb, "ctrl_Root", base_head + Vector((0, 0, -0.02)), base_head + Vector((0.12, 0, -0.02)))
    ctrl_body = eb_new(eb, "ctrl_Body", base_head, base_head + Vector((0.10, 0, 0)), parent=ctrl_root)

    # reparent the PHYSICAL base 'root' bone under ctrl_Body (so Body moves the torso,
    # Root moves everything). Bone parenting keeps the physical joint values intact.
    eb["root"].parent = eb["ctrl_Body"]
    eb["root"].use_connect = False

    # foot controls already exist (parent=None, world). Keep them world; a Child Of to
    # ctrl_Root (added in pose mode) gives ROOT-space following. Nothing to do in edit mode
    # except make sure they are not bone-parented to anything.
    for leg in LEGS:
        if f"ctrl_{leg}_foot" in eb:
            eb[f"ctrl_{leg}_foot"].parent = None

    # expressive controls near their joints, parented to ctrl_Body (ride the torso)
    for cname, jnames in HEAD_CTRL.items():
        j0 = jnames[0]
        if j0 not in eb:   # rev_3 has no ears
            continue
        h = eb[j0].head.copy()
        eb_new(eb, cname, h, h + Vector((0, 0, 0.04)), parent=eb["ctrl_Body"])

    bpy.ops.object.mode_set(mode='OBJECT')

    # ---------------------------------------------------------------- pose wiring
    bpy.ops.object.mode_set(mode='POSE')
    pbs = arm.pose.bones
    root_ctrl = pbs["ctrl_Root"]
    for leg in LEGS:
        add_prop(root_ctrl, f"ik_{leg}", 0.0)        # default FK so physical joints animate directly
        add_prop(root_ctrl, f"footroot_{leg}", 1.0)  # default follow root (character space)

    # make the existing leg IK switchable (default off -> FK works, Test 1)
    for leg in LEGS:
        ik_end = pbs.get(f"{leg}_ik_end")
        ik = next((c for c in ik_end.constraints if c.type == 'IK'), None) if ik_end else None
        if ik is not None:
            drive_influence(arm, ik, arm.name, "ctrl_Root", f"ik_{leg}")

    # foot space switch: Child Of ctrl_Root, influence = footroot_<leg>
    for leg in LEGS:
        cf = pbs.get(f"ctrl_{leg}_foot")
        if cf is None:
            continue
        co = cf.constraints.new('CHILD_OF'); co.name = "space_root"
        co.target = arm; co.subtarget = "ctrl_Root"
        # set inverse so influence=1 keeps the foot where it is at rest
        bpy.context.view_layer.update()
        drive_influence(arm, co, arm.name, "ctrl_Root", f"footroot_{leg}")

    # expressive controls -> physical joints via Copy Rotation (local)
    for cname, jnames in HEAD_CTRL.items():
        if cname not in pbs:
            continue
        for jn in jnames:
            pb = pbs[jn]
            cr = pb.constraints.new('COPY_ROTATION'); cr.name = f"from_{cname}"
            cr.target = arm; cr.subtarget = cname
            cr.target_space = cr.owner_space = 'LOCAL'
            cr.use_x = cr.use_y = False; cr.use_z = True   # 1-DOF hinge about local Z

    bpy.ops.object.mode_set(mode='OBJECT')

    # child-of inverse matrices must be baked so influence=1 is a no-op at rest
    bpy.ops.object.mode_set(mode='POSE')
    for leg in LEGS:
        cf = pbs.get(f"ctrl_{leg}_foot")
        if cf is None: continue
        for c in cf.constraints:
            if c.type == 'CHILD_OF':
                arm.data.bones.active = arm.data.bones[cf.name]
                try:
                    with bpy.context.temp_override(object=arm, active_object=arm, active_pose_bone=cf,
                                                   pose_bone=cf, constraint=c):
                        bpy.ops.constraint.childof_set_inverse(constraint=c.name, owner='BONE')
                except Exception as e:
                    print(f"[[ childof inverse op failed for {cf.name}: {e}; computing manually")
                    # inverse = (parent bone pose matrix in world) at bind, so influence=1 is a no-op now
                    c.inverse_matrix = (arm.matrix_world @ pbs["ctrl_Root"].matrix).inverted()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print(f"[[ animator rig written: {a.out}")
    print(f"[[ bones now: {len(arm.data.bones)}")


main()
