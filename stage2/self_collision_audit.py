"""How far the leg links penetrate the TORSO hull, per frame.

The pipeline constrains the floor and nothing else, so a clip that folds the legs up
under a sitting body can drive a thigh or a shank straight through the chest. Nothing
downstream notices: PhysX is given self-collision off, the balance audits only look
at what touches z=0, and the joint limits are all satisfied - the pose is legal and
visibly wrong.

Penetration is measured against the real collision hull, as a convex polytope: a
vertex is inside when every face inequality A x + b <= 0 holds, and its depth is the
distance to the nearest face.

  python3 stage2/self_collision_audit.py motions/deadpan_v4.npz [--frames 0-90]
"""
import argparse, itertools, os, sys
import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "stage4"))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")


def fk(kin, ch, q, nm, rp, rq, i):
    d = {n: q[i, nm.index(n)] for n in nm}
    fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
    while st:
        par = st.pop(); Rp, pp = fr[par]
        for jn in ch.get(par, []):
            J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
            fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
            st.append(J["child"])
    return fr


def penetration(hull_eq, pts):
    """Max depth (m) of any point inside the polytope; 0 if all are outside."""
    v = pts @ hull_eq[:, :3].T + hull_eq[:, 3]        # (N, nfaces), <0 = inside
    inside = v.max(1) < 0
    return float(-v[inside].max(1).min()) if inside.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--body", default="origin", help="link whose hull is the obstacle")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--report", type=float, default=0.002,
                    help="metres of penetration worth listing")
    a = ap.parse_args()
    kin = V4Kin(URDF); cm = ContactModel()
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)
    m = np.load(a.motion, allow_pickle=True)
    q = m["dof_positions"].astype(float); nm = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q)
    parts = [f"{l}_{s}" for l in LEGS for s in ("shoulder_pitch", "knee")
             if f"{l}_{s}" in cm.hull]
    body_local = cm.hull[a.body]
    eq = ConvexHull(body_local).equations                # in the body link's frame
    # Some links are nested inside the torso BY DESIGN - each *_shoulder_pitch hip
    # housing sits 37.2 mm inside it at the URDF zero pose. Subtract that baseline so
    # the report shows only the penetration the ANIMATION introduced.
    base = {}
    fr0 = fk(kin, ch, np.zeros((1, 21)), nm, np.zeros((1, 3)),
             np.array([[1.0, 0.0, 0.0, 0.0]]), 0)
    Rb0, pb0 = fr0[a.body]
    for p in parts:
        Rl, pl = fr0[p]
        base[p] = penetration(eq, ((cm.hull[p] @ Rl.T + pl) - pb0) @ Rb0)
    print("[[ zero-pose baseline (nested by design), mm: "
          + "  ".join(f"{p.split('_',1)[1][:8]} {base[p]*1000:.1f}"
                      for p in parts[:2]) + " ...")
    dep = np.zeros((T, len(parts)))
    for i in range(T):
        fr = fk(kin, ch, q, nm, rp, rq, i)
        Rb, pb = fr[a.body]
        for c, p in enumerate(parts):
            Rl, pl = fr[p]
            w = cm.hull[p] @ Rl.T + pl                   # world
            loc = (w - pb) @ Rb                          # into the body link's frame
            dep[i, c] = max(0.0, penetration(eq, loc) - base[p])
    print(f"{os.path.basename(a.motion)}  T={T}  obstacle = {a.body} hull")
    worst = dep.max(1)
    print(f"[[ frames with any leg inside the {a.body}: "
          f"{int((worst > a.report).sum())}/{T} = {100*(worst > a.report).mean():.0f}% "
          f"| worst depth {worst.max()*1000:.1f} mm at frame {int(np.argmax(worst))}")
    for c, p in enumerate(parts):
        n = int((dep[:, c] > a.report).sum())
        if n:
            print(f"     {p:20s} {n:4d} frames, max {dep[:, c].max()*1000:5.1f} mm")
    bad = np.where(worst > a.report)[0]
    if len(bad):
        runs = []
        for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
            gg = [x[1] for x in g]; runs.append((gg[0], gg[-1]))
        print("[[ windows: " + ", ".join(f"{x}-{y}" for x, y in runs[:16])
              + (" ..." if len(runs) > 16 else ""))
    if a.frames:
        f0, f1 = (int(x) for x in a.frames.split("-", 1))
        print("  frame " + " ".join(f"{p[:9]:>10s}" for p in parts))
        for i in range(max(0, f0), min(T - 1, f1) + 1):
            print(f"  {i:5d} " + " ".join(f"{dep[i, c]*1000:10.1f}" for c in range(len(parts))))


if __name__ == "__main__":
    main()
