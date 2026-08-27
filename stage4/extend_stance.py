"""Turn every two-paw LINE support into a polygon by the smallest possible change to
the contact schedule: land one foot a few frames early, or keep one down a few
frames late.

Forcing all four paws down through such a window was measured and rejected - the
swing foot is then dragged along the floor and the robot cannot step (DeadPan: root
position error 112 -> 332 mm, Laidback 355 -> 546 mm). This asks for much less. For
each window it picks the ONE lifted paw that is closest to the floor - the one about
to land or that has just left - and extends only that paw's stance across the window
plus a short ramp. Three contacts is all roll support needs.

It only prints the --force-planted specification; run stage4/ground_fix.py with it.

  python3 stage4/extend_stance.py motions/deadpan_v4.npz --pad 2
"""
import argparse, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2")); sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")

ap = argparse.ArgumentParser()
ap.add_argument("motion")
ap.add_argument("--pad", type=int, default=2, help="frames of ramp either side")
ap.add_argument("--tol", type=float, default=0.005)
ap.add_argument("--max-lift", type=float, default=0.040,
                help="metres. Do not draft a paw that is further off the floor than "
                     "this - pulling it down would be re-authoring the swing, not "
                     "adjusting a touchdown.")
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
pz = np.zeros((T, 4)); torso = np.zeros(T, bool)
for i in range(T):
    d = {n: q[i, nm.index(n)] for n in nm}
    fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
    while st:
        par = st.pop(); Rp, pp = fr[par]
        for jn in ch.get(par, []):
            J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
            fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
            st.append(J["child"])
    for k, l in enumerate(LEGS):
        R, p = fr[f"{l}_knee"]
        pz[i, k] = float((cm.hull[f"{l}_knee"] @ R.T + p)[:, 2].min())
    torso[i] = any(float((cm.hull[ln] @ R_.T + p_)[:, 2].min()) < a.tol
                   for ln, (R_, p_) in fr.items()
                   if ln in cm.hull and not any(ln.startswith(l + "_") for l in LEGS))

down = pz < a.tol
n = down.sum(1)
bad = np.where((n <= 2) & ~torso)[0]
runs = []
for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
    gg = [x[1] for x in g]
    runs.append((gg[0], gg[-1]))
spec = []; skipped = []
for s0, e0 in runs:
    lift = np.where(~down[s0:e0 + 1].all(0))[0]
    cand = [k for k in range(4) if not down[s0:e0 + 1, k].all()]
    if not cand:
        continue
    # the lifted paw that stays closest to the floor across the window
    hgt = {k: float(pz[s0:e0 + 1, k].min()) for k in cand}
    k = min(hgt, key=hgt.get)
    if hgt[k] > a.max_lift:
        skipped.append((s0, e0, LEGS[k], hgt[k]))
        continue
    spec.append((max(0, s0 - a.pad), min(T - 1, e0 + a.pad), LEGS[k], hgt[k]))
print(f"{os.path.basename(a.motion)}  T={T}  |  {len(runs)} window(s) with <=2 paws")
for s0, e0, l, h in spec:
    print(f"  {s0:4d}-{e0:<4d}  draft {l}  (its lowest point in the window is "
          f"{h*1000:+.1f} mm)")
for s0, e0, l, h in skipped:
    print(f"  {s0:4d}-{e0:<4d}  SKIPPED - nearest lifted paw ({l}) is {h*1000:.0f} mm "
          f"up, past --max-lift")
if spec:
    print("\n--force-planted \"" +
          ";".join(f"{s0},{e0}:{l}" for s0, e0, l, _ in spec) + "\"")
