"""Check paw/foot clipping through the floor (z=0) at rest pose.

Reports, for every mesh, its lowest world-Z vertex. The lowest point overall is
where the robot would first touch a ground plane at z=0. Negative = below floor
(clipping); we also report the standing clearance if the base 'root' is placed so
the lowest paw sits on z=0.
"""
import bpy
from mathutils import Vector

import sys
OUT = open(sys.argv[-1], "w") if sys.argv[-1].endswith(".txt") else sys.stdout
def P(*a): print(*a, file=OUT)

dg = bpy.context.evaluated_depsgraph_get()
rows = []
for ob in bpy.data.objects:
    if ob.type != 'MESH':
        continue
    ev = ob.evaluated_get(dg)
    m = ev.to_mesh()
    if not m.vertices:
        ev.to_mesh_clear(); continue
    mw = ob.matrix_world
    zs = [(mw @ v.co).z for v in m.vertices]
    rows.append((ob.name, min(zs), max(zs)))
    ev.to_mesh_clear()

rows.sort(key=lambda r: r[1])
P("=== per-mesh world-Z range at rest pose (m) ===")
for name, zmin, zmax in rows:
    tag = "  <-- LOWEST" if name == rows[0][0] else ""
    P(f"  {name:22s} zmin {zmin:+.4f}  zmax {zmax:+.4f}{tag}")

overall_min = rows[0][1]
# paw/foot meshes only
paw = [r for r in rows if any(k in r[0].lower() for k in ("knee", "foot", "paw", "shank", "toe"))]
P("\n=== foot/lower-leg meshes ===")
for name, zmin, zmax in sorted(paw, key=lambda r: r[1]):
    P(f"  {name:22s} zmin {zmin:+.4f}")

P(f"\nOVERALL_LOWEST_Z {overall_min:+.5f}  ({rows[0][0]})")
P("BELOW_FLOOR" if overall_min < -1e-4 else "ABOVE_OR_ON_FLOOR")
OUT.flush()
if OUT is not sys.stdout: OUT.close()
import os; os._exit(0)
