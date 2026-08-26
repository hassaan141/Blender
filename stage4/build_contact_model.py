"""Stage 4 - build an orientation-aware contact model from the REAL collision meshes.

PAW_CONTACT_LOCAL was a single fixed point (the centroid of the lowest 1 mm surface
patch of each knee STL). That is only valid for the one shank orientation it was
measured at: as the shank rotates, a different part of the paw becomes the lowest
point. Measured bias against the true mesh extent was +9 mm on the front legs and
up to +67 mm on the hind legs.

Fix: store the CONVEX HULL vertices of each collision mesh (the URDF importer runs
with convex_decomp=False, so each collision shape *is* the convex hull of its STL).
At runtime the lowest point is then min over hull vertices of (R @ v + p).z, which
is exact for any orientation and matches what PhysX actually collides.

    python3 stage4/build_contact_model.py        # writes stage4/out/collision_hulls.npz
"""
import os, struct
import numpy as np

MESH_DIR = ("/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/meshes")
OUT = "/home/hassaan/Bingo/Blender/stage4/out/collision_hulls.npz"
# every link that can plausibly reach the floor
LINKS = ["fl_knee", "fr_knee", "bl_knee", "br_knee",
         "fl_shoulder_pitch", "fr_shoulder_pitch", "bl_shoulder_pitch", "br_shoulder_pitch",
         "origin", "head_roll", "tail_yaw",
         "fl_shoulder_yaw", "fr_shoulder_yaw", "bl_shoulder_yaw", "br_shoulder_yaw"]


def read_stl(path):
    with open(path, "rb") as f:
        head = f.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(n * 3, 3).astype(np.float64)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {}
    for link in LINKS:
        p = os.path.join(MESH_DIR, link + ".STL")
        if not os.path.exists(p):
            print(f"[[ skip {link} (no mesh)"); continue
        v = read_stl(p)
        if len(v) < 4:
            print(f"[[ skip {link} (degenerate stub, {len(v)} verts)"); continue
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(v)
            hv = v[np.unique(hull.vertices)]
        except Exception as e:
            hv = np.unique(np.round(v, 6), axis=0)
            print(f"[[ {link}: ConvexHull unavailable ({e}); using unique verts")
        out[link] = hv
        print(f"[[ {link:20s} {len(v):6d} tri-verts -> {len(hv):5d} hull verts | "
              f"local z {hv[:,2].min():+.4f}..{hv[:,2].max():+.4f} m")
    np.savez(OUT, **out)
    print(f"[[ wrote {OUT}  ({len(out)} links)")


main()
