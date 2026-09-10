"""Generate the Bingo v4 MJCF physics twin from the validated v4 URDF.

The browser cannot run Isaac Sim, so it needs a MuJoCo model of the SAME robot. This
generates it deterministically from the URDF rather than rebuilding Bingo by hand:
every mass, inertia tensor, joint origin, joint axis, joint limit and collision mesh
is copied from `bingo_urdf_w_ear_joints_physics.urdf`, which is the artefact Isaac
itself consumes (bingo_v4.py spawns a USD converted from it).

Why not `mujoco.MjModel.from_xml_path()` on the URDF directly
-------------------------------------------------------------
It refuses the model:

    Error: mesh volume is too small: head_yaw . Try setting inertia to shell

That is not an importer quirk. FOUR collision meshes in the v4 URDF are degenerate -
exactly 12 triangles and 0.0000 mm^3 of volume:

    tail_pitch  0.00300 kg      head_yaw     0.00500 kg
    l_ear_pitch 0.00200 kg      r_ear_pitch  0.00200 kg

They are the same four near-massless intermediate links that bingo_v4.py documents
as the articulation-conditioning problem ("head_yaw is 5 g driving the 0.708 kg head
- a 140:1 mass ratio the articulation solver cannot condition, so the joint jams
instead of moving"), which is why that file carries hand-tuned armature values. The
degenerate mesh is the same defect seen from the geometry side, and `discardvisual`
does not dodge it because these are collision meshes.

The fix here is not a workaround: the URDF supplies an explicit <inertial> for every
link, so this generator uses those verbatim and MuJoCo never needs a mesh volume at
all. Meshes are additionally declared inertia="shell" as a second guard.

Nothing about Bingo's morphology is altered. Run:

    python3 bingo-simulator/tools/export_bingo_mujoco.py
    python3 bingo-simulator/tools/export_bingo_mujoco.py --check   # compile it too
"""
from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(SIM_ROOT, ".."))

URDF = os.path.join(REPO, "URDF/bingo_urdf v4_w_ear_joints/urdf/"
                          "bingo_urdf_w_ear_joints_physics.urdf")
MESH_SRC = os.path.join(REPO, "URDF/bingo_urdf v4_w_ear_joints/meshes")

BASE_LINK = "origin"

# Canonical order, identical to v4_kinematics.DOF_ORDER and bake_conform.DOF_ORDER_21.
# Joints are DECLARED in this order so qpos/ctrl indices are stable and predictable.
# Runtime index lookup is still done by NAME, never positionally.
DOF_ORDER = [
    "fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
    "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
    "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
    "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
]

# The validated standing pose (stage4/stand_test.py STAND_SOLVED). Verified against
# the real collision hulls: all four paws at exactly -0.180 m, spread 0.00 mm.
# NOT the pose in BINGO_V4_CFG.init_state, which is the stale rev_3 pose and is not
# a valid v4 stance (see docs/locomotion/TASK1_DESIGN.md).
STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}
STAND_BASE_HEIGHT = 0.182          # lowest paw hull 2 mm clear of the floor

# Actuator parameters, copied from the Stage-4 validated bingo_v4.py. Those values
# are derived (stage4/actuator_analysis.py) and gain-swept; they are not re-tuned
# here. kv is the IdealPD damping, forcerange the effort ceiling that Isaac enforces
# explicitly via IdealPDActuator.
GAINS = {
    "SY":   dict(kp=40.0,  kv=0.90, force=3.0, armature=0.01),
    "SP":   dict(kp=120.0, kv=1.60, force=3.0, armature=0.01),
    "knee": dict(kp=120.0, kv=1.60, force=3.0, armature=0.01),
    "head": dict(kp=60.0,  kv=1.80, force=6.0, armature=0.06),
    "tail": dict(kp=8.0,   kv=0.20, force=6.0, armature=0.02),
    "ear":  dict(kp=0.6,   kv=0.05, force=1.0, armature=0.0005),
}


def gain_group(joint: str) -> str:
    if joint.endswith("_SY_J"):
        return "SY"
    if joint.endswith("_SP_J"):
        return "SP"
    if joint.endswith("_knee"):
        return "knee"
    if joint.startswith("head_"):
        return "head"
    if joint.startswith("tail_"):
        return "tail"
    if "_ear_" in joint:
        return "ear"
    raise KeyError(f"no gain group for joint {joint!r}")


# --------------------------------------------------------------------------- URDF
def parse_urdf(path):
    root = ET.parse(path).getroot()

    links = {}
    for l in root.findall("link"):
        name = l.get("name")
        inertial = l.find("inertial")
        entry = {"name": name, "mass": 0.0, "com": (0, 0, 0), "rpy": (0, 0, 0),
                 "inertia": None, "collision": None, "visual": None}
        if inertial is not None:
            m = inertial.find("mass")
            if m is not None:
                entry["mass"] = float(m.get("value"))
            o = inertial.find("origin")
            if o is not None:
                entry["com"] = tuple(float(v) for v in (o.get("xyz") or "0 0 0").split())
                entry["rpy"] = tuple(float(v) for v in (o.get("rpy") or "0 0 0").split())
            i = inertial.find("inertia")
            if i is not None:
                entry["inertia"] = {k: float(i.get(k, 0.0))
                                    for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        for tag in ("collision", "visual"):
            el = l.find(tag)
            if el is not None:
                mesh = el.find(".//mesh")
                if mesh is not None:
                    o = el.find("origin")
                    entry[tag] = {
                        "mesh": os.path.basename(mesh.get("filename")),
                        "xyz": tuple(float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()),
                        "rpy": tuple(float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()),
                    }
        links[name] = entry

    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        joints[name] = {
            "name": name,
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": tuple(float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()),
            "rpy": tuple(float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()),
            "axis": tuple(float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()),
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
        }
    return links, joints


def fmt(vals):
    return " ".join(f"{v:.9g}" for v in vals)


# --------------------------------------------------------------------------- MJCF
def build_mjcf(links, joints, meshdir, with_scene):
    """Emit the MJCF tree. Body hierarchy follows the URDF parent/child graph."""
    children = {}
    for j in joints.values():
        children.setdefault(j["parent"], []).append(j)
    # declare each body's child joints in canonical order where possible
    order = {n: i for i, n in enumerate(DOF_ORDER)}
    for lst in children.values():
        lst.sort(key=lambda j: order.get(j["name"], 999))

    mujoco = ET.Element("mujoco", model="bingo_v4")

    # inertiafromgeom="false": use the URDF's <inertial> verbatim. This is what makes
    # the four degenerate collision meshes harmless - MuJoCo never computes inertia
    # from mesh volume, so it never needs one.
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir=meshdir,
                  inertiafromgeom="false", balanceinertia="true",
                  autolimits="true")
    ET.SubElement(mujoco, "option", timestep="0.008333333", gravity="0 0 -9.81",
                  integrator="implicitfast", cone="elliptic")
    ET.SubElement(mujoco, "size", njmax="600", nconmax="200")

    default = ET.SubElement(mujoco, "default")
    # SELF-COLLISION IS OFF, to match Isaac. bingo.py spawns the robot with
    # ArticulationRootPropertiesCfg(enabled_self_collisions=False), so the validated
    # Stage-4 physics has never had robot-vs-robot contacts. Leaving MuJoCo's default
    # on is not a harmless difference: measured, the head chain then rests against the
    # torso and head_yaw sits with its actuator pinned at the 6 N m ceiling against a
    # -8.0 N m contact constraint, moving 0.014 rad instead of the commanded 0.200
    # (with gravity off, the same command reaches 0.165). Standing showed 30 contact
    # points for 4 paws.
    #
    # contype/conaffinity encode it: two geoms collide iff
    # (contype1 & conaffinity2) || (contype2 & conaffinity1).
    #   robot geoms  contype 1, conaffinity 0  -> robot vs robot: 0, never
    #   floor        contype 0, conaffinity 1  -> robot vs floor: 1, always
    #   props        contype 1, conaffinity 1  -> collide with robot and floor
    ET.SubElement(default, "geom", condim="4", friction="0.8 0.02 0.001",
                  margin="0.001", rgba="0.7 0.72 0.75 1",
                  contype="1", conaffinity="0")
    ET.SubElement(default, "joint", damping="0.0", frictionloss="0.0")

    asset = ET.SubElement(mujoco, "asset")
    used = []
    for l in links.values():
        src = l["collision"] or l["visual"]
        if src:
            mname = src["mesh"]
            if mname not in used:
                used.append(mname)
                # inertia="shell" is a second guard: with inertiafromgeom="false"
                # MuJoCo should not need mesh inertia at all, but four of these
                # meshes have zero volume and this makes that explicit rather than
                # dependent on compiler internals.
                ET.SubElement(asset, "mesh", name=mname, file=mname, inertia="shell")

    worldbody = ET.SubElement(mujoco, "worldbody")
    if with_scene:
        ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                      width="512", height="512", rgb1="0.16 0.17 0.19",
                      rgb2="0.20 0.21 0.24")
        ET.SubElement(asset, "material", name="grid", texture="grid",
                      texrepeat="12 12", reflectance="0.05")
        ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1",
                      directional="true", diffuse="0.7 0.7 0.7")
        ET.SubElement(worldbody, "geom", name="floor", type="plane",
                      size="8 8 0.05", material="grid",
                      condim="4", friction="0.8 0.02 0.001",
                      contype="0", conaffinity="1")

    def emit_body(parent_el, link_name, joint=None):
        l = links[link_name]
        attrs = {"name": link_name}
        if joint is not None:
            attrs["pos"] = fmt(joint["xyz"])
            if any(abs(v) > 1e-12 for v in joint["rpy"]):
                attrs["euler"] = fmt(joint["rpy"])
        elif with_scene or True:
            attrs["pos"] = f"0 0 {STAND_BASE_HEIGHT:.6f}"
        body = ET.SubElement(parent_el, "body", **attrs)

        if joint is None:
            ET.SubElement(body, "freejoint", name="root")

        # inertial straight from the URDF
        if l["inertia"] is not None and l["mass"] > 0:
            i = l["inertia"]
            ia = {"pos": fmt(l["com"]), "mass": f"{l['mass']:.9g}",
                  "fullinertia": fmt((i["ixx"], i["iyy"], i["izz"],
                                      i["ixy"], i["ixz"], i["iyz"]))}
            if any(abs(v) > 1e-12 for v in l["rpy"]):
                ia["euler"] = fmt(l["rpy"])
            ET.SubElement(body, "inertial", **ia)

        if joint is not None and joint["type"] in ("revolute", "continuous"):
            ja = {"name": joint["name"], "type": "hinge",
                  "axis": fmt(joint["axis"]), "pos": "0 0 0"}
            if joint["lower"] is not None and joint["upper"] is not None:
                ja["range"] = f"{joint['lower']:.9g} {joint['upper']:.9g}"
                ja["limited"] = "true"
            g = GAINS[gain_group(joint["name"])]
            ja["armature"] = f"{g['armature']:.9g}"
            ET.SubElement(body, "joint", **ja)

        col = l["collision"]
        if col:
            ga = {"name": f"{link_name}_col", "type": "mesh", "mesh": col["mesh"],
                  "pos": fmt(col["xyz"]), "group": "3"}
            if any(abs(v) > 1e-12 for v in col["rpy"]):
                ga["euler"] = fmt(col["rpy"])
            ET.SubElement(body, "geom", **ga)

        for cj in children.get(link_name, []):
            emit_body(body, cj["child"], cj)

    emit_body(worldbody, BASE_LINK)

    # ---- actuators, in canonical order ----
    act = ET.SubElement(mujoco, "actuator")
    for jn in DOF_ORDER:
        g = GAINS[gain_group(jn)]
        j = joints[jn]
        a = {"name": jn, "joint": jn, "kp": f"{g['kp']:.9g}", "kv": f"{g['kv']:.9g}",
             "forcerange": f"-{g['force']:.9g} {g['force']:.9g}"}
        if j["lower"] is not None and j["upper"] is not None:
            a["ctrlrange"] = f"{j['lower']:.9g} {j['upper']:.9g}"
        ET.SubElement(act, "position", **a)

    # ---- sensors the observation needs ----
    sen = ET.SubElement(mujoco, "sensor")
    ET.SubElement(sen, "gyro", name="base_gyro", site="imu")
    ET.SubElement(sen, "accelerometer", name="base_acc", site="imu")
    ET.SubElement(sen, "framequat", name="base_quat", objtype="site", objname="imu")
    ET.SubElement(sen, "velocimeter", name="base_vel", site="imu")

    # the IMU site on the base
    base_el = worldbody.find(f".//body[@name='{BASE_LINK}']")
    ET.SubElement(base_el, "site", name="imu", pos="0 0 0", size="0.005")

    # ---- keyframe: the validated stance ----
    key = ET.SubElement(mujoco, "keyframe")
    qpos = [0.0, 0.0, STAND_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0]
    qpos += [STAND_SOLVED.get(n, 0.0) for n in DOF_ORDER]
    ET.SubElement(key, "key", name="stand", qpos=fmt(qpos),
                  ctrl=fmt([STAND_SOLVED.get(n, 0.0) for n in DOF_ORDER]))
    return mujoco


def pretty(el):
    raw = ET.tostring(el, encoding="unicode")
    out = minidom.parseString(raw).toprettyxml(indent="  ")
    return "\n".join(line for line in out.split("\n") if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(SIM_ROOT, "app/public/robot"))
    ap.add_argument("--check", action="store_true", help="compile the result in MuJoCo")
    ap.add_argument("--copy-meshes", action="store_true", default=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    mesh_out = os.path.join(a.out_dir, "meshes")
    os.makedirs(mesh_out, exist_ok=True)

    links, joints = parse_urdf(URDF)
    print(f"[[ URDF: {len(links)} links, {len(joints)} joints, "
          f"{sum(l['mass'] for l in links.values()):.5f} kg")

    missing = [n for n in DOF_ORDER if n not in joints]
    if missing:
        raise SystemExit(f"URDF is missing canonical joints: {missing}")
    extra = [n for n in joints if n not in DOF_ORDER and joints[n]["type"] != "fixed"]
    if extra:
        raise SystemExit(f"URDF has non-fixed joints outside DOF_ORDER: {extra}")

    if a.copy_meshes:
        n = 0
        for l in links.values():
            for k in ("collision", "visual"):
                if l[k]:
                    src = os.path.join(MESH_SRC, l[k]["mesh"])
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(mesh_out, l[k]["mesh"]))
                        n += 1
        print(f"[[ copied {n} mesh reference(s) into {mesh_out}")

    # Collide against the convex hulls in collision/, not the raw CAD meshes. Both
    # MuJoCo and PhysX convexify collision geometry anyway (contact_model.py: "the
    # URDF importer uses convex_decomp=False, so each collision shape is the convex
    # hull of its STL"), so this is the same physics at 1/100th the download.
    # Generated by tools/export_render_model.py.
    for name, scene in (("bingo_v4.xml", False), ("bingo_scene.xml", True)):
        el = build_mjcf(links, joints, "collision", scene)
        path = os.path.join(a.out_dir, name)
        with open(path, "w") as f:
            f.write(pretty(el) + "\n")
        print(f"[[ wrote {path}")

    if a.check:
        import mujoco
        for name in ("bingo_v4.xml", "bingo_scene.xml"):
            p = os.path.join(a.out_dir, name)
            m = mujoco.MjModel.from_xml_path(p)
            print(f"[[ {name}: COMPILES  nbody={m.nbody} njnt={m.njnt} nq={m.nq} "
                  f"nv={m.nv} nu={m.nu} ngeom={m.ngeom} mass={m.body_mass.sum():.5f} kg")


if __name__ == "__main__":
    main()
