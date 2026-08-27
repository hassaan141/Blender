"""List the frame windows where a reference stands on two paws or fewer, and say
whether that pair is DIAGONAL.

This is the single mechanism behind every clip in this set that does not complete:
a diagonal pair is a line contact, a line contact has no roll support, and an
open-loop joint-target player has no roll feedback. Reported as ready-to-use
--force-planted ranges.

  python3 stage4/support_windows.py motions/deadpan_v4.npz [--max-paws 2] [--min-len 2]
"""
import argparse, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2")); sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")
DIAG = ({"fl", "br"}, {"fr", "bl"})

ap = argparse.ArgumentParser()
ap.add_argument("motion")
ap.add_argument("--max-paws", type=int, default=2)
ap.add_argument("--min-len", type=int, default=2)
ap.add_argument("--tol", type=float, default=0.005)
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
down = np.zeros((T, 4), bool); torso = np.zeros(T, bool)
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
        down[i, k] = float((cm.hull[f"{l}_knee"] @ R.T + p)[:, 2].min()) < a.tol
    torso[i] = any(float((cm.hull[ln] @ R_.T + p_)[:, 2].min()) < a.tol
                   for ln, (R_, p_) in fr.items()
                   if ln in cm.hull and not any(ln.startswith(l + "_") for l in LEGS))

n = down.sum(1)
print(f"{os.path.basename(a.motion)}  T={T}")
print("  paws on the floor: " + "  ".join(f"{k}:{int((n==k).sum())} ({100*(n==k).mean():.0f}%)"
                                          for k in range(5))
      + f" | torso/head/tail also touching on {100*torso.mean():.0f}% of frames")
diag = np.array([n[i] == 2 and {LEGS[k] for k in range(4) if down[i, k]} in DIAG
                 for i in range(T)])
lat = np.array([n[i] == 2 and not diag[i] for i in range(T)])
print(f"  two-paw frames: {int((n==2).sum())} of which DIAGONAL {int(diag.sum())} "
      f"({100*diag.mean():.0f}% of the clip) and same-side/same-end {int(lat.sum())}")
bad = np.where((n <= a.max_paws) & ~torso)[0]
runs = []
for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
    gg = [x[1] for x in g]
    if len(gg) >= a.min_len:
        runs.append((gg[0], gg[-1]))
tot = sum(e - s + 1 for s, e in runs)
print(f"  windows with <= {a.max_paws} paws and no torso contact "
      f"({len(runs)} windows, {tot} frames = {100*tot/T:.0f}% of the clip):")
for s, e in runs:
    kind = "DIAGONAL" if diag[s:e+1].any() else "lateral "
    print(f"    {s:4d}-{e:<4d} ({e-s+1:3d} fr) {kind}  --force-planted {s},{e}")
