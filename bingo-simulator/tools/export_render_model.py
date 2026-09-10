"""Produce browser-sized geometry and the render rig description for Bingo.

Two separate outputs, because physics geometry and visual geometry are NOT the same
thing (brief §5 - never render the raw collision model):

  collision/   convex hulls, for MuJoCo in the browser
  render/      decimated visual meshes, for Three.js
  kinematics.json  the body tree + joint axes/origins + mesh per body

Why convex hulls for collision
------------------------------
The raw v4 STLs are 45.4 MB / 907,484 triangles - CAD tessellation, with 90k+
triangles on a 15 g shoulder_yaw bracket. That is unusable over a network and
pointless for physics, because BOTH engines convexify collision meshes anyway.
stage4/contact_model.py states it for the Isaac side: "the URDF importer uses
convex_decomp=False, so each collision shape is the convex hull of its STL". So
shipping hulls is not a simplification of the physics - it is the physics, made
explicit. Measured reduction: bl_shoulder_yaw 93,480 -> 342 triangles.

Why decimation (not hulls) for rendering
----------------------------------------
A convex hull looks wrong: it fills every concavity. Visual meshes are decimated
instead, preserving shape, to a triangle budget.

    python3 bingo-simulator/tools/export_render_model.py
    python3 bingo-simulator/tools/export_render_model.py --render-budget 120000
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(SIM_ROOT, ".."))
URDF = os.path.join(REPO, "URDF/bingo_urdf v4_w_ear_joints/urdf/"
                          "bingo_urdf_w_ear_joints_physics.urdf")
MESH_SRC = os.path.join(REPO, "URDF/bingo_urdf v4_w_ear_joints/meshes")

DOF_ORDER = [
    "fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
    "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
    "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
    "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
]


def read_stl(path):
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    v = np.zeros((n * 3, 3), np.float64)
    off = 84
    for i in range(n):
        f = struct.unpack("<12f", d[off:off + 48])
        off += 50
        v[i * 3:i * 3 + 3] = np.array(f[3:12]).reshape(3, 3)
    faces = np.arange(n * 3).reshape(n, 3)
    return v, faces


def write_stl(path, verts, faces):
    tri = len(faces)
    buf = bytearray(84 + tri * 50)
    struct.pack_into("<I", buf, 80, tri)
    off = 84
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        nrm = nrm / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
        struct.pack_into("<12fH", buf, off,
                         *nrm, *a, *b, *c, 0)
        off += 50
    with open(path, "wb") as fh:
        fh.write(bytes(buf))
    return tri


def weld(verts, faces, tol=9):
    """Merge duplicate vertices so decimation and hulls behave."""
    key = np.round(verts, tol)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nf = inv[faces]
    good = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    return uniq, nf[good]


def parse_links():
    root = ET.parse(URDF).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        e = {"name": l.get("name"), "collision": None, "visual": None}
        for tag in ("collision", "visual"):
            el = l.find(tag)
            if el is not None and el.find(".//mesh") is not None:
                o = el.find("origin")
                e[tag] = {
                    "mesh": os.path.basename(el.find(".//mesh").get("filename")),
                    "xyz": [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()],
                    "rpy": [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()],
                }
        links[l.get("name")] = e
    for j in root.findall("joint"):
        o, ax = j.find("origin"), j.find("axis")
        lim = j.find("limit")
        joints[j.get("name")] = {
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()],
            "rpy": [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()],
            "axis": [float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()],
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
        }
    return links, joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(SIM_ROOT, "app/public/robot"))
    ap.add_argument("--render-budget", type=int, default=90000,
                    help="total triangle budget for the visual meshes")
    ap.add_argument("--collision-max-faces", type=int, default=256,
                    help="cap per-mesh collision hull faces. A raw hull of a dense CAD "
                         "part is still thousands of triangles (shoulder_pitch: 8446) "
                         "and contributes nothing physically - the shape is already "
                         "convex, so coarsening it changes the surface by microns "
                         "while cutting the download.")
    ap.add_argument("--paw-max-faces", type=int, default=4096,
                    help="separate, much looser cap for the four *_knee links. Those "
                         "carry the paw contact surface - the one piece of geometry the "
                         "whole robot balances on - and coarsening it to 256 faces "
                         "MEASURABLY changed the physics: the stance rose 2.3 mm and one "
                         "paw stopped touching the floor (3/4 contacts). Brackets can be "
                         "crude; the paws cannot.")
    a = ap.parse_args()

    col_dir = os.path.join(a.out_dir, "collision")
    ren_dir = os.path.join(a.out_dir, "render")
    os.makedirs(col_dir, exist_ok=True)
    os.makedirs(ren_dir, exist_ok=True)

    links, joints = parse_links()

    # ---- pass 1: read every mesh once, note its size ----
    meshes = {}
    for l in links.values():
        for k in ("collision", "visual"):
            if l[k]:
                mn = l[k]["mesh"]
                if mn not in meshes:
                    p = os.path.join(MESH_SRC, mn)
                    if not os.path.exists(p):
                        continue
                    v, f = read_stl(p)
                    meshes[mn] = weld(v, f)
    src_tris = sum(len(f) for _, f in meshes.values())
    print(f"[[ source: {len(meshes)} meshes, {src_tris} triangles")

    # ---- collision: convex hulls ----
    import fast_simplification
    col_tris = 0
    col_report = []
    for mn, (v, f) in meshes.items():
        out = os.path.join(col_dir, mn)
        # the paw (knee) links are the contact surface: give them their own cap
        cap = a.paw_max_faces if mn.lower().startswith(
            ("fl_knee", "fr_knee", "bl_knee", "br_knee")) else a.collision_max_faces
        try:
            h = ConvexHull(v)
            hv, hf = h.points, h.simplices
            if len(hf) > cap:
                red = 1.0 - cap / len(hf)
                sv, sf = fast_simplification.simplify(
                    hv.astype(np.float32), hf.astype(np.uint32), target_reduction=red)
                # re-hull so the result is guaranteed convex after decimation
                h2 = ConvexHull(np.asarray(sv, dtype=np.float64))
                hv, hf = h2.points, h2.simplices
            n = write_stl(out, hv, hf)
        except Exception:
            # The four degenerate meshes (12 triangles, zero volume) have no hull.
            # They are the near-massless intermediate links; pass them through as-is
            # so the body still exists rather than silently dropping geometry.
            n = write_stl(out, v, f)
        col_tris += n
        col_report.append((mn, len(f), n))
    print(f"[[ collision: convex hulls, {col_tris} triangles "
          f"({100*col_tris/max(1,src_tris):.1f}% of source)")

    # ---- render: decimate proportionally to the budget ----
    ratio = min(1.0, a.render_budget / max(1, src_tris))
    ren_tris = 0
    for mn, (v, f) in meshes.items():
        target = max(64, int(len(f) * ratio))
        if target < len(f):
            vv, ff = fast_simplification.simplify(
                v.astype(np.float32), f.astype(np.uint32),
                target_reduction=1.0 - target / len(f))
        else:
            vv, ff = v, f
        ren_tris += write_stl(os.path.join(ren_dir, mn), np.asarray(vv), np.asarray(ff))
    print(f"[[ render:    decimated to {ren_tris} triangles "
          f"({100*ren_tris/max(1,src_tris):.1f}% of source)")

    # ---- kinematics.json: the render rig description ----
    children = {}
    for j in joints.values():
        children.setdefault(j["parent"], []).append(j["name"])
    kin = {
        "robot": "bingo_v4",
        "base_link": "origin",
        "dof_order": DOF_ORDER,
        "mesh_dir": "render",
        "collision_dir": "collision",
        "bodies": [],
        "note": ("Driven ENTIRELY from MuJoCo state. The render rig sets each body's "
                 "local rotation from the corresponding joint angle and the root from "
                 "the free joint; it must never be animated independently."),
    }
    for name, l in links.items():
        parent_joint = next((j for j in joints.values() if j["child"] == name), None)
        kin["bodies"].append({
            "name": name,
            "parent": parent_joint["parent"] if parent_joint else None,
            "joint": parent_joint["name"] if parent_joint else None,
            "joint_type": parent_joint["type"] if parent_joint else "free",
            "joint_axis": parent_joint["axis"] if parent_joint else None,
            "joint_range": ([parent_joint["lower"], parent_joint["upper"]]
                            if parent_joint and parent_joint["lower"] is not None else None),
            "pos": parent_joint["xyz"] if parent_joint else [0, 0, 0],
            "rpy": parent_joint["rpy"] if parent_joint else [0, 0, 0],
            "mesh": (l["visual"] or l["collision"] or {}).get("mesh"),
            "mesh_pos": (l["visual"] or l["collision"] or {}).get("xyz", [0, 0, 0]),
            "mesh_rpy": (l["visual"] or l["collision"] or {}).get("rpy", [0, 0, 0]),
        })
    kp = os.path.join(a.out_dir, "kinematics.json")
    with open(kp, "w") as f:
        json.dump(kin, f, indent=1)
    print(f"[[ wrote {kp}  ({len(kin['bodies'])} bodies)")

    def mb(d):
        return sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)) / 1e6
    print(f"[[ payload: collision {mb(col_dir):.2f} MB, render {mb(ren_dir):.2f} MB")


if __name__ == "__main__":
    main()
