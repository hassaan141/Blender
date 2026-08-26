"""Stage 4.1 - give the four zero-mass intermediate links a credible inertial.

Problem: the CAD export writes mass = 1e-12 kg and inertia = 1.667e-23 for the
intermediate links of the compound (2-axis) joints:

    tail_pitch   head_yaw   l_ear_pitch   r_ear_pitch

Each is a ZERO-LENGTH virtual link - its child joint origin is exactly "0 0 0" -
and its visual mesh is a 684-byte stub. Physically it is the rotating part between
the two orthogonal axes: a motor rotor plus its bracket. PhysX cannot condition an
articulation around a body with 1e-12 kg / 1e-23 inertia, which is why the ears ran
at 8-10 rad/s with ~3.5 rad tracking error in the Stage 4 baseline.

Fix: replace ONLY the <inertial> mass and inertia of those four links with a small
physically plausible rotor mass and a sphere-equivalent inertia (I = 2/5 m r^2).
Everything else - joint origins, axes, kinematics, visual/collision geometry,
joint limits, every other link - is copied through byte-for-byte.

    python3 stage4/patch_zero_mass.py --urdf <in.urdf> --out <out.urdf>
"""
import argparse
import xml.etree.ElementTree as ET

# link -> (mass kg, characteristic radius m, rationale)
PATCH = {
    "head_yaw":    (0.005, 0.012, "head yaw rotor + bracket; carries the 0.708 kg head"),
    "tail_pitch":  (0.003, 0.010, "tail pitch rotor + bracket"),
    "l_ear_pitch": (0.002, 0.008, "small ear servo rotor"),
    "r_ear_pitch": (0.002, 0.008, "small ear servo rotor"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tree = ET.parse(a.urdf)
    root = tree.getroot()
    done = []
    for ln in root.findall("link"):
        name = ln.get("name")
        if name not in PATCH:
            continue
        m, r, why = PATCH[name]
        I = 0.4 * m * r * r                     # solid sphere, 2/5 m r^2
        inert = ln.find("inertial")
        old_m = float(inert.find("mass").get("value"))
        inert.find("mass").set("value", repr(m))
        it = inert.find("inertia")
        for k, v in (("ixx", I), ("iyy", I), ("izz", I),
                     ("ixy", 0.0), ("ixz", 0.0), ("iyz", 0.0)):
            it.set(k, repr(v))
        # COM origin is left exactly as exported (all four are <5e-6 m from the
        # joint, i.e. already at the virtual link's own frame).
        done.append((name, old_m, m, I, why))

    tree.write(a.out, xml_declaration=True, encoding="utf-8")
    print(f"[[ patched {len(done)} links -> {a.out}")
    for name, old_m, m, I, why in done:
        print(f"     {name:14s} mass {old_m:.1e} -> {m:.4f} kg | "
              f"I_xx=I_yy=I_zz {I:.3e} kg m^2 | {why}")
    tot = sum(x[2] for x in done)
    print(f"[[ total added mass {tot*1000:.1f} g")


main()
