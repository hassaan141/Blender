"""IK acceptance test for the conform rig.

Drives every foot control around its workspace and checks three things that all
have to hold at once, because fixing one by breaking another is the failure mode
this rig hit repeatedly during development:

  1. TRACKING  - the paw actually goes where the control is put
  2. HINGE     - each joint still rotates about ONE axis (its motor axis).
                 IK is perfectly happy to produce off-axis poses that look fine
                 in Blender and are impossible on the robot.
  3. LIMITS    - no joint is driven past what the URDF allows

    blender -b blend/conform/Bingo_ConformRig.blend -P scripts/test_ik.py
"""
import bpy, math
from mathutils import Vector

LEGS = ("fl", "fr", "bl", "br")
JOINTS = lambda l: (f"{l}_SY_J", f"{l}_SP_J", f"{l}_knee")
# Offsets in metres: fore/aft, lateral, lift.
#
# The reachable envelope is STRONGLY ASYMMETRIC and this test used to get it
# wrong. The default stance already puts each paw ahead of its own hip (front
# +103 mm, back +44 mm), so at the 0.199 m stance only about **35 mm of forward
# travel** remains, against ~188 mm backward (measured, MEMORY.md section 3).
# The old "forward" case asked for 60 mm -- outside the workspace -- so all four
# legs "failed" by 11.2 mm every run. That was the robot's real reach limit being
# reported as a test failure, not a rig defect: off-axis stayed 0.00000 deg and
# the joints stayed inside their limits, i.e. the IK clamped exactly as intended.
# 30 mm forward sits inside the real envelope and actually tests tracking.
OFFSETS = [("neutral", (0, 0, 0)), ("forward", (0.03, 0, 0)), ("back", (-0.06, 0, 0)),
           ("lift", (0, 0, 0.04)), ("lift+fwd", (0.05, 0, 0.03)),
           ("out", (0, 0.03, 0)), ("in", (0, -0.03, 0)), ("down", (0, 0, -0.015))]

arm = bpy.data.objects["Bingo_Robot"]
fails = []
rows = []

lims = {}
for l in LEGS:
    for jn in JOINTS(l):
        c = next(c for c in arm.pose.bones[jn].constraints if c.type == 'LIMIT_ROTATION')
        lims[jn] = (c.min_z, c.max_z)


def local_euler(ev, name):
    pb = ev.pose.bones[name]
    rest = pb.bone.matrix_local
    base = (pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ rest) if pb.parent else rest
    return (base.inverted() @ pb.matrix).to_euler('XYZ')


def evaluate():
    bpy.context.view_layer.update()
    return arm.evaluated_get(bpy.context.evaluated_depsgraph_get())


for l in LEGS:
    ctrl = arm.pose.bones[f"ctrl_{l}_foot"]
    home = ctrl.location.copy()
    for label, off in OFFSETS:
        # control bones are unparented, so a world offset maps through the
        # bone's own rest orientation
        R = ctrl.bone.matrix_local.to_3x3()
        ctrl.location = home + R.inverted() @ Vector(off)
        ev = evaluate()

        want = arm.matrix_world @ ev.pose.bones[f"ctrl_{l}_foot"].head
        got = arm.matrix_world @ ev.pose.bones[f"{l}_foot_tip"].head
        track = (want - got).length

        off_axis, over = 0.0, []
        for jn in JOINTS(l):
            e = local_euler(ev, jn)
            off_axis = max(off_axis, abs(e.x), abs(e.y))
            lo, hi = lims[jn]
            if e.z < lo - 1e-4 or e.z > hi + 1e-4:
                over.append(f"{jn}={e.z:+.3f}")

        ok = track < 0.002 and off_axis < 1e-4 and not over
        rows.append((l, label, track * 1000, math.degrees(off_axis), over, ok))
        if not ok:
            fails.append(f"{l}/{label}")
    ctrl.location = home
evaluate()

print(f"[[ {'leg':4s} {'target':10s} {'tracking':>10s} {'off-axis':>10s}  limits")
for l, label, track, offdeg, over, ok in rows:
    print(f"[[ {'PASS' if ok else 'FAIL'} {l:4s} {label:10s} {track:7.3f} mm "
          f"{offdeg:8.5f} deg  {'ok' if not over else ','.join(over)}")

# unreachable target must clamp gracefully, not explode or go off-hinge
ctrl = arm.pose.bones["ctrl_fl_foot"]
home = ctrl.location.copy()
ctrl.location = home + ctrl.bone.matrix_local.to_3x3().inverted() @ Vector((0.40, 0, 0))
ev = evaluate()
off_axis = max(max(abs(local_euler(ev, jn).x), abs(local_euler(ev, jn).y)) for jn in JOINTS("fl"))
inside = all(lims[jn][0] - 1e-4 <= local_euler(ev, jn).z <= lims[jn][1] + 1e-4 for jn in JOINTS("fl"))
ok = off_axis < 1e-4 and inside
print(f"[[ {'PASS' if ok else 'FAIL'} unreachable target (400 mm) clamps and stays legal "
      f"— off-axis {math.degrees(off_axis):.5f} deg, within limits: {inside}")
if not ok:
    fails.append("unreachable")
ctrl.location = home
evaluate()

print(f"\n[[ {'IK TEST PASSED' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
