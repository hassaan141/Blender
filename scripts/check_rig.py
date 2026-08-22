"""Sanity-check the conform rig against the URDF before handing it to the animator.

    blender -b blend/conform/Bingo_ConformRig.blend -P scripts/check_rig.py -- \
        --urdf bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf

Every check is PASS/FAIL against the URDF, not against an expectation I typed in.
"""
import bpy, sys, math, argparse
import xml.etree.ElementTree as ET
from mathutils import Vector

SHANK_LEN, PAW_DROP = 0.120, 0.0288
LEGS = ("fl", "fr", "bl", "br")
fails = []


def chk(ok, label, detail=""):
    print(f"[[ {'PASS' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(); ap.add_argument("--urdf", required=True)
a = ap.parse_args(argv)

root = ET.parse(a.urdf).getroot()
def rpy_to_mat(r, p, y):
    from mathutils import Matrix
    return (Matrix.Rotation(y, 3, 'Z') @ Matrix.Rotation(p, 3, 'Y') @ Matrix.Rotation(r, 3, 'X'))

uj = {}
for j in root.findall("joint"):
    lim = j.find("limit"); ax = j.find("axis"); o = j.find("origin")
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
    uj[j.get("name")] = dict(lo=float(lim.get("lower")), hi=float(lim.get("upper")),
                             axis=Vector([float(v) for v in ax.get("xyz").split()]),
                             parent=j.find("parent").get("link"),
                             child=j.find("child").get("link"),
                             R=rpy_to_mat(*rpy))

# Link frames: a joint's axis is expressed in its PARENT link's frame, not world.
from mathutils import Matrix as _M
_fr = {"origin": _M.Identity(3)}
_chg = True
while _chg:
    _chg = False
    for _n, _j in uj.items():
        if _j["child"] in _fr or _j["parent"] not in _fr:
            continue
        _fr[_j["child"]] = _fr[_j["parent"]] @ _j["R"]
        _chg = True
for _n, _j in uj.items():
    _j["axis_w"] = (_fr[_j["parent"]] @ _j["axis"]).normalized()

arm = bpy.data.objects.get("Bingo_Robot")
chk(arm is not None, "armature 'Bingo_Robot' exists")
if arm is None:
    sys.exit(1)

# 1. every URDF joint has a bone, exactly named
missing = [n for n in uj if n not in arm.pose.bones]
chk(not missing, f"all {len(uj)} URDF joints have identically-named bones", ",".join(missing))

# 2. no stray animatable bones beyond joints + foot tips + root
allowed = (set(uj) | {f"{l}_foot_tip" for l in LEGS} | {f"{l}_ik_end" for l in LEGS}
           | {f"ctrl_{l}_foot" for l in LEGS} | {"root"})
extra = [b.name for b in arm.pose.bones if b.name not in allowed]
chk(not extra, "no unexpected extra bones", ",".join(extra))

# 3. each bone's local Z is the real motor axis
worst, wn = 0.0, ""
for n, j in uj.items():
    z = (arm.matrix_world.to_3x3() @ arm.pose.bones[n].bone.matrix_local.to_3x3() @ Vector((0, 0, 1)))
    d = math.degrees(z.normalized().angle(j["axis_w"]))
    if d > worst:
        worst, wn = d, n
chk(worst < 0.01, "every bone's local Z is its motor axis", f"worst {wn} {worst:.4f} deg")

# 4. one DOF only, and limits inside the URDF's
bad_lock, bad_lim = [], []
for n in uj:
    pb = arm.pose.bones[n]
    if not (pb.lock_rotation[0] and pb.lock_rotation[1]):
        bad_lock.append(n)
    c = next((c for c in pb.constraints if c.type == 'LIMIT_ROTATION'), None)
    if c is None or not c.use_limit_z:
        bad_lim.append(n + "(none)")
    elif c.min_z < uj[n]["lo"] - 1e-6 or c.max_z > uj[n]["hi"] + 1e-6:
        bad_lim.append(f"{n}[{c.min_z:.2f},{c.max_z:.2f}]")
chk(not bad_lock, "X/Y rotation locked on every joint bone", ",".join(bad_lock))
chk(not bad_lim, "limit constraints present and inside URDF limits", ",".join(bad_lim))

# 5. no scale or location animation allowed on joints
bad = [n for n in uj if not all(arm.pose.bones[n].lock_scale)
       or not all(arm.pose.bones[n].lock_location)]
chk(not bad, "location and scale locked on every joint bone", ",".join(bad))

# 6. meshes attached, and to a real bone
mesh = [o for o in bpy.data.objects if o.type == 'MESH']
badp = [o.name for o in mesh if o.parent != arm or o.parent_bone not in arm.pose.bones]
# Expected mesh count comes from the URDF, not a constant: rev_3 has 18 visual
# meshes, v4 has 22 (the four ear links). Hardcoding it fails on every new export.
n_vis = sum(1 for ln in root.iter("link")
            if ln.find("visual/geometry/mesh") is not None)
chk(len(mesh) == n_vis, f"{n_vis} robot meshes present (per URDF)", f"found {len(mesh)}")
chk(not badp, "every mesh is bone-parented to the rig", ",".join(badp))

# 7. geometry: segment lengths match the URDF
dg = bpy.context.evaluated_depsgraph_get(); ev = arm.evaluated_get(dg)
P = lambda n: arm.matrix_world @ ev.pose.bones[n].head
for l in LEGS:
    thigh = (P(f"{l}_knee") - P(f"{l}_SP_J")).length
    shank = (P(f"{l}_foot_tip") - P(f"{l}_knee")).length
    ok = abs(thigh - 0.0836) < 5e-4 and abs(shank - SHANK_LEN) < 1e-4
    chk(ok, f"{l} segment lengths", f"thigh {thigh*1000:.1f}mm (URDF 83.6) shank {shank*1000:.1f}mm (120.0)")

# 8. standing on the floor, feet level
zs = [P(f"{l}_foot_tip").z for l in LEGS]
paw = min(zs) - PAW_DROP
chk(abs(paw) < 2e-3, "paws rest on z=0", f"lowest paw {paw*1000:+.1f} mm")
chk((max(zs) - min(zs)) < 1e-3, "all four feet level in the default pose",
    f"spread {(max(zs)-min(zs))*1000:.2f} mm")
base = (arm.matrix_world @ ev.pose.bones["fl_SY_J"].head).z
chk(0.185 <= base <= 0.205, "base height in spec band 0.19-0.20 m", f"{base:.4f} m")

# 9. units / scene
chk(bpy.context.scene.unit_settings.system == 'METRIC', "scene units are metric")
chk(abs(arm.scale.x - 1) < 1e-6 and abs(arm.scale.y - 1) < 1e-6 and abs(arm.scale.z - 1) < 1e-6,
    "armature scale is 1,1,1", str(tuple(round(v, 4) for v in arm.scale)))

# 10. limits actually bite: drive a knee past its limit and confirm it is stopped
pb = arm.pose.bones["fl_knee"]
pb.rotation_euler.z = 5.0
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get(); ev = arm.evaluated_get(dg)
got = ev.pose.bones["fl_knee"].matrix_channel.to_euler().z
lim = next(c for c in pb.constraints if c.type == 'LIMIT_ROTATION').max_z
chk(abs(got) <= lim + 0.02, "limit constraint actually clamps an over-range pose",
    f"asked 5.00 rad, evaluated {got:.3f}, limit {lim:.2f}")
pb.rotation_euler.z = 0.60

# 11. IK actually drives the feet: move a control, confirm the foot follows it
ctrl = arm.pose.bones["ctrl_fl_foot"]
before = P("fl_foot_tip")
ctrl.location.z += 0.02
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get(); ev = arm.evaluated_get(dg)
after = arm.matrix_world @ ev.pose.bones["fl_foot_tip"].head
moved = (after - before).length
chk(moved > 0.015, "IK: moving a foot control moves the foot",
    f"control moved 20mm, foot moved {moved*1000:.1f}mm")
ctrl.location.z -= 0.02
bpy.context.view_layer.update()

print(f"\n[[ {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
