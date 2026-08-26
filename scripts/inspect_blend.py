"""Dump armatures, bones, anim range, fps, units for a blend. Write to a .txt."""
import bpy, sys
OUT = open(sys.argv[-1], "w") if sys.argv[-1].endswith(".txt") else sys.stdout
def P(*a): print(*a, file=OUT)

sc = bpy.context.scene
P(f"FILE {bpy.data.filepath}")
P(f"frame_start {sc.frame_start}  frame_end {sc.frame_end}  fps {sc.render.fps} / {sc.render.fps_base}")
P(f"unit_system {sc.unit_settings.system}  scale_length {sc.unit_settings.scale_length}")

for ob in bpy.data.objects:
    if ob.type == 'ARMATURE':
        P(f"\n=== ARMATURE object '{ob.name}'  data '{ob.data.name}' ===")
        P(f"  world matrix loc {tuple(round(v,4) for v in ob.matrix_world.translation)}")
        P(f"  scale {tuple(round(v,4) for v in ob.scale)}  rot_euler {tuple(round(v,4) for v in ob.rotation_euler)}")
        ad = ob.animation_data
        if ad and ad.action:
            act = ad.action
            fr = act.frame_range
            P(f"  action '{act.name}'  frame_range {tuple(round(v,1) for v in fr)}  fcurves {len(act.fcurves)}")
            # bones that are actually animated
            animated = set()
            for fc in act.fcurves:
                dp = fc.data_path
                if dp.startswith('pose.bones["'):
                    animated.add(dp.split('"')[1])
            P(f"  animated bones ({len(animated)}): {sorted(animated)}")
        else:
            P("  (no action)")
        P(f"  --- all bones ({len(ob.pose.bones)}) ---")
        for pb in ob.pose.bones:
            par = pb.parent.name if pb.parent else "-"
            cons = ",".join(c.type for c in pb.constraints)
            P(f"    {pb.name:24s} parent={par:20s} cons=[{cons}]")

# meshes
P(f"\n=== MESHES ({sum(1 for o in bpy.data.objects if o.type=='MESH')}) ===")
for ob in bpy.data.objects:
    if ob.type == 'MESH':
        pinfo = f"{ob.parent.name}/{ob.parent_bone}" if ob.parent else "-"
        P(f"    {ob.name:28s} parent={pinfo}")

OUT.flush()
if OUT is not sys.stdout: OUT.close()
import os; os._exit(0)
