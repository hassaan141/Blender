"""Stage 2.12 - render a blend's animation to PNG frames with EEVEE, auto-framed.

Used to build the side-by-side Cheeky comparison. A matched 3/4 front view is set
up from each character's own forward/up axes so the performances are comparable.

  blender -b <file.blend> -P stage2/render_compare.py -- --target robot|ashley \
      --outdir stage2/out/frames_x [--every 2]
"""
import bpy, sys, argparse
import numpy as np
from mathutils import Vector

FRAMES = {  # forward, anatomical left, up, root-bone, armature
    "robot":  (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)), "root", "Bingo_Robot"),
    # Ashley is left-handed before the solver reflection: forward +Y, left +X.
    "ashley": (Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1)), "def_Pelvis", "Bingo_Rig"),
}


def is_body_mesh(ob, target):
    """Only the character's render meshes - not control shapes, stages, empties."""
    # Linked source libraries can contain a second object with the same name as
    # the scene instance.  Membership-by-name accepts both and makes framing span
    # the animated dog plus an invisible rest-pose duplicate near the origin.
    if (ob.type != 'MESH' or bpy.context.scene.objects.get(ob.name) is not ob
            or ob.hide_render):
        return False
    n = ob.name.lower()
    if any(k in n for k in ("cntrl", "control", "ctrl", "stage", "whatisthis", "box_")):
        return False
    if len(ob.data.vertices) < 40:
        return False
    if target == "ashley":
        # Ashley's render meshes live in the 'FinalMesh' collection
        return any("finalmesh" in c.name.lower() for c in ob.users_collection)
    return True


def char_bounds(target):
    """Evaluated core-character bounds, excluding travel and expressive extremities.

    Ashley's ears have very large authored excursions; including them in framing
    makes the dog tiny exactly when the ear gesture is strongest.  They remain
    rendered, but torso/head/legs determine camera center and scale.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e9,)*3); hi = -lo; any_ = False
    for ob in bpy.data.objects:
        if not is_body_mesh(ob, target):
            continue
        n = ob.name.lower()
        if "ear" in n or "tail" in n:
            continue
        ev = ob.evaluated_get(dg)
        for c in ev.bound_box:
            w = ev.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w)); any_ = True
    return ((lo + hi) * 0.5, (hi - lo).length) if any_ else (root_world(target), 1.0)


def char_size(frame_samples, target):
    """Median evaluated core-body diagonal (character size, travel excluded)."""
    diags = []
    for f in frame_samples:
        bpy.context.scene.frame_set(f)
        _, diag = char_bounds(target)
        diags.append(diag)
    diags.sort()
    for ob in bpy.data.objects:                      # hide non-body meshes
        if ob.type == 'MESH' and not is_body_mesh(ob, target):
            ob.hide_render = True
    return diags[len(diags) // 2] if diags else 1.0


def root_world(target):
    _, _, _, bone, armn = FRAMES[target]
    arm = bpy.data.objects[armn]
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return arm.matrix_world @ ev.pose.bones[bone].head


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(FRAMES))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--res", type=int, default=540)
    ap.add_argument("--engine", choices=("workbench", "eevee"), default="workbench")
    ap.add_argument("--decimate", type=float, default=0.12,
                    help="temporary robot mesh ratio for fast workbench comparison")
    ap.add_argument("--frames", help="comma-separated exact frames (overrides --every)")
    a = ap.parse_args(argv)

    sc = bpy.context.scene
    if a.engine == "workbench":
        # Comparison is about synchronized motion, not surface tessellation.
        # Ashley's linked meshes otherwise spend minutes evaluating Subdivision
        # in software headless rendering.
        sc.render.use_simplify = True
        sc.render.simplify_subdivision = 0
        for ob in bpy.data.objects:
            for mod in getattr(ob, "modifiers", ()):
                if mod.type == 'SUBSURF':
                    mod.show_render = False
                    mod.show_viewport = False
        if a.target == "robot" and 0 < a.decimate < 1:
            bpy.ops.object.select_all(action='DESELECT')
            for ob in list(bpy.context.scene.objects):
                if ob.type != 'MESH' or len(ob.data.vertices) < 5000:
                    continue
                bpy.context.view_layer.objects.active = ob; ob.select_set(True)
                mod = ob.modifiers.new("comparison_decimate", 'DECIMATE')
                mod.ratio = a.decimate
                bpy.ops.object.modifier_apply(modifier=mod.name)
                ob.select_set(False)
    fwd, left, up = FRAMES[a.target][0], FRAMES[a.target][1], FRAMES[a.target][2]

    f0, f1 = sc.frame_start, sc.frame_end
    samples = list(range(f0, f1 + 1, max(1, (f1 - f0) // 8)))
    csize = char_size(samples, a.target)             # character size (no travel)
    view = (0.62 * fwd - 0.82 * left + 0.26 * up).normalized()
    dist = (2.35 if a.target == "robot" else 2.6) * csize

    cam = bpy.data.cameras.new("cmp"); co = bpy.data.objects.new("cmp", cam)
    sc.collection.objects.link(co); sc.camera = co
    cam.lens = 50

    # sun + fill
    for name, e, energy in (("key", (0.5, -0.4, 0.9), 4.0), ("fill", (-0.6, 0.3, 0.6), 1.5)):
        lt = bpy.data.lights.new(name, "SUN"); lo_ = bpy.data.objects.new(name, lt)
        sc.collection.objects.link(lo_); lo_.rotation_euler = e; lt.energy = energy

    # big ground plane at z=0 (covers travel)
    ground_z = -2.93 if a.target == "ashley" else 0.0
    bpy.ops.mesh.primitive_plane_add(size=csize * 40, location=(0, 0, ground_z))

    if a.engine == "workbench":
        sc.render.engine = 'BLENDER_WORKBENCH'
        sc.display.shading.light = 'STUDIO'
        sc.display.shading.color_type = 'MATERIAL'
        sc.display.shading.show_shadows = True
        sc.display.shading.show_cavity = True
        sc.display.shading.cavity_type = 'WORLD'
    else:
        try:
            sc.render.engine = 'BLENDER_EEVEE_NEXT'
        except Exception:
            sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x = a.res; sc.render.resolution_y = int(a.res * 0.75)
    try:                                              # Blender 5.x: IMAGE vs VIDEO
        sc.render.image_settings.media_type = 'IMAGE'
    except Exception:
        pass
    sc.render.image_settings.file_format = 'PNG'      # some source files default to video
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Standard'
    import os
    os.makedirs(a.outdir, exist_ok=True)

    up_off = up * (0.08 * csize)                     # keep torso/head near center
    render_frames = ([int(x) for x in a.frames.split(',')] if a.frames
                     else list(range(f0, f1 + 1, a.every)))
    for f in render_frames:
        sc.frame_set(f)
        center, _ = char_bounds(a.target)            # follow evaluated visible body
        aim = center + up_off
        co.location = aim + view * dist
        co.rotation_euler = (aim - co.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        sc.render.filepath = os.path.join(a.outdir, f"f{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    schedule = a.frames if a.frames else f"{f0}-{f1} every {a.every}"
    print(f"[[ rendered {a.target} frames {schedule}; core size={csize:.3f} -> {a.outdir}")
    sys.stdout.flush(); os._exit(0)


main()
