"""Headless validation of the v4 animator rig (spec Tests 1-5)."""
import bpy
from mathutils import Vector
arm = bpy.data.objects["Bingo_Robot"]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pbs = arm.pose.bones
root = pbs["ctrl_Root"]


def upd():
    for _ in range(3): bpy.context.view_layer.update()   # let IK/drivers settle
def dg(): return bpy.context.evaluated_depsgraph_get()
def mpos(mesh):
    o = bpy.data.objects[mesh].evaluated_get(dg()); return o.matrix_world.translation.copy()
def tip(leg):  # physical shank tip = foot_tip bone head, evaluated
    ev = arm.evaluated_get(dg()); return (arm.matrix_world @ ev.pose.bones[f"{leg}_foot_tip"].head).copy()
def bhead(bn):  # evaluated bone head (pivot) in world
    ev = arm.evaluated_get(dg()); return (arm.matrix_world @ ev.pose.bones[bn].head).copy()
def reset():
    for pb in pbs:
        pb.rotation_euler = (0, 0, 0); pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0)
    for leg in ("fl", "fr", "bl", "br"):
        root[f"ik_{leg}"] = 0.0; root[f"footroot_{leg}"] = 1.0
    upd()


# ---- Test 1: physical FK moves the downstream leg (measure foot tip + pivots) ----
# foot tip moves for every joint; the joint's OWN pivot and all UPSTREAM pivots stay.
reset()
t0 = tip("fl"); sp_piv0 = bhead("fl_SP_J"); kn_piv0 = bhead("fl_knee")
pbs["fl_SY_J"].rotation_euler[2] = 0.20; upd()
sy_tip = (tip("fl") - t0).length * 1000; sy_movesSP = (bhead("fl_SP_J") - sp_piv0).length * 1000
reset()
t0 = tip("fl"); sy_piv0 = bhead("fl_SY_J"); kn_piv0 = bhead("fl_knee")
pbs["fl_SP_J"].rotation_euler[2] = 0.30; upd()
sp_tip = (tip("fl") - t0).length * 1000; sp_keepsSY = (bhead("fl_SY_J") - sy_piv0).length * 1000
sp_movesKN = (bhead("fl_knee") - kn_piv0).length * 1000
reset()
t0 = tip("fl"); kn_piv0 = bhead("fl_knee"); sp_piv0 = bhead("fl_SP_J")
pbs["fl_knee"].rotation_euler[2] = 0.50; upd()
kn_tip = (tip("fl") - t0).length * 1000; kn_keepsKNpiv = (bhead("fl_knee") - kn_piv0).length * 1000
kn_keepsSP = (bhead("fl_SP_J") - sp_piv0).length * 1000
print("TEST1 fl_SY_J+0.2 -> foot_tip %.0fmm, moves SP pivot %.0fmm" % (sy_tip, sy_movesSP))
print("TEST1 fl_SP_J+0.3 -> foot_tip %.0fmm, SY pivot stays %.1fmm, moves knee pivot %.0fmm" % (sp_tip, sp_keepsSY, sp_movesKN))
print("TEST1 fl_knee+0.5 -> foot_tip %.0fmm, knee pivot stays %.1fmm, SP pivot stays %.1fmm" % (kn_tip, kn_keepsKNpiv, kn_keepsSP))
t1 = (sy_tip > 5 and sy_movesSP > 5 and sp_tip > 5 and sp_keepsSY < 1 and sp_movesKN > 5
      and kn_tip > 5 and kn_keepsKNpiv < 1 and kn_keepsSP < 1)
print("TEST1", "PASS" if t1 else "FAIL")

# ---- Test 2: ctrl_Root moves whole dog, feet included, no stretch ----------
reset()
b0 = mpos("mesh_origin"); f0 = {l: tip(l) for l in ("fl", "fr", "bl", "br")}
seg0 = {l: (tip(l) - mpos(f"mesh_{l}_knee")).length for l in ("fl", "fr", "bl", "br")}
root.location = (0, 0, 0.05); upd()
db = (mpos("mesh_origin") - b0).z * 1000
dfeet = {l: (tip(l) - f0[l]).z * 1000 for l in ("fl", "fr", "bl", "br")}
seg1 = {l: (tip(l) - mpos(f"mesh_{l}_knee")).length for l in ("fl", "fr", "bl", "br")}
stretch = max(abs(seg1[l] - seg0[l]) * 1000 for l in seg0)
print("TEST2 ctrl_Root+50mm -> torso %.0f  feet %s mm  max-stretch %.2f mm" %
      (db, {l: round(dfeet[l]) for l in dfeet}, stretch))
t2 = abs(db - 50) < 3 and all(abs(dfeet[l] - 50) < 3 for l in dfeet) and stretch < 1
print("TEST2", "PASS" if t2 else "FAIL")

# ---- Test 3: crouch - feet planted (WORLD), body down, knees bend ----------
reset()
for l in ("fl", "fr", "bl", "br"):
    root[f"ik_{l}"] = 1.0; root[f"footroot_{l}"] = 0.0   # IK on, feet world-fixed
upd()
def knee_dir(l):  # evaluated knee bone direction (captures IK bend)
    ev = arm.evaluated_get(dg()); pb = ev.pose.bones[f"{l}_knee"]
    return (pb.tail - pb.head).normalized()
f0 = {l: tip(l) for l in ("fl", "fr", "bl", "br")}
kd0 = {l: knee_dir(l) for l in ("fl", "fr", "bl", "br")}
b0 = mpos("mesh_origin")
pbs["ctrl_Body"].location = (0, 0, -0.03); upd()
foot_move = max((tip(l) - f0[l]).length * 1000 for l in ("fl", "fr", "bl", "br"))
db = (mpos("mesh_origin") - b0).z * 1000
import math
knee_bend = max(math.degrees(kd0[l].angle(knee_dir(l))) for l in ("fl", "fr", "bl", "br"))
print("TEST3 ctrl_Body-30mm(feet WORLD,IK) -> feet moved %.1f mm  body %.0f mm  max knee-bend %.1f deg" %
      (foot_move, db, knee_bend))
t3 = foot_move < 5 and db < -10 and knee_bend > 3
print("TEST3", "PASS" if t3 else "FAIL")

# ---- Test 4: plant one foot WORLD, move torso; then ROOT, move root --------
reset()
root["ik_fl"] = 1.0; root["footroot_fl"] = 0.0; upd()
f0 = tip("fl")
pbs["ctrl_Body"].location = (-0.015, 0, 0); upd()   # within the leg's reach envelope
foot_stay = (tip("fl") - f0).length * 1000
# switch to ROOT space, move root
reset()
root["ik_fl"] = 1.0; root["footroot_fl"] = 1.0; upd()
f0 = tip("fl")
root.location = (0.03, 0, 0); upd()
foot_follow = (tip("fl") - f0).length * 1000
print("TEST4 WORLD+body30mm -> foot moved %.1f mm (want ~0) | ROOT+root30mm -> foot moved %.1f mm (want ~30)" %
      (foot_stay, foot_follow))
t4 = foot_stay < 5 and abs(foot_follow - 30) < 6
print("TEST4", "PASS" if t4 else "FAIL")

reset()
print("ALL_TESTS", "PASS" if (t1 and t2 and t3 and t4) else "SOME_FAIL")
