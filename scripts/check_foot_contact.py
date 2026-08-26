"""Does the IK foot point coincide with the bottom of the paw mesh?

If the foot control / ik_end bone is ABOVE the lowest paw vertex, then planting
that control on the floor (z=0) sinks the visible paw below the floor by the gap.
This is the usual cause of 'paws clip through the floor' in an animator rig.
"""
import bpy, sys
from mathutils import Vector
OUT = open(sys.argv[-1], "w") if sys.argv[-1].endswith(".txt") else sys.stdout
def P(*a): print(*a, file=OUT)

arm = bpy.data.objects["Bingo_Robot"]
dg = bpy.context.evaluated_depsgraph_get()

def bone_head_z(bn):
    ev = arm.evaluated_get(dg)
    return (arm.matrix_world @ ev.pose.bones[bn].head).z

def bone_tail_z(bn):
    ev = arm.evaluated_get(dg)
    return (arm.matrix_world @ ev.pose.bones[bn].tail).z

def paw_bottom_z(leg):
    ob = bpy.data.objects[f"mesh_{leg}_knee"].evaluated_get(dg)
    m = ob.to_mesh(); mw = bpy.data.objects[f"mesh_{leg}_knee"].matrix_world
    z = min((mw @ v.co).z for v in m.vertices); ob.to_mesh_clear(); return z

P("=== foot IK point vs paw-mesh bottom (rest pose) ===")
P(f"{'leg':4s} {'ik_end_z':>10s} {'ctrl_foot_z':>12s} {'paw_bottom_z':>13s} {'gap(ik-bottom)':>15s}")
worst = 0.0
for leg in ("fl", "fr", "bl", "br"):
    ik_z = bone_tail_z(f"{leg}_ik_end")   # kinematic foot point = shank tip
    ctrl = f"ctrl_{leg}_foot"
    cz = bone_head_z(ctrl) if ctrl in arm.pose.bones else float('nan')
    pb = paw_bottom_z(leg)
    gap = (ik_z - pb) * 1000
    worst = max(worst, abs(gap))
    P(f"{leg:4s} {ik_z:+10.4f} {cz:+12.4f} {pb:+13.4f} {gap:+13.1f}mm")

P(f"\nWORST_GAP_MM {worst:.2f}")
P("PAW_CLIPS_WHEN_PLANTED" if worst > 3 else "FOOT_POINT_AT_PAW_BOTTOM_OK")
OUT.flush()
if OUT is not sys.stdout: OUT.close()
import os; os._exit(0)
