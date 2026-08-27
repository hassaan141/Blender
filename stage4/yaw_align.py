"""Rotate a finished clip about the world Z axis so its net travel runs along +X.

Purely a choice of world frame. Gravity is along Z, so a yaw about Z changes nothing
physical, and because every joint angle in this pipeline is body-relative, NOT ONE
DOF IS TOUCHED - the head, ears, tail and legs come out bit-identical. Only the root
pose and the world-space diagnostic arrays are re-expressed.

Why it is needed: the exported world frame is built from the character's frame-0
heading, and a clip whose character CRABS then runs diagonally across the Blender
grid. Eccentric is the case - Ashley's body faces 27.0 deg off her own direction of
travel for the whole clip and the retarget reproduces that to 0.2 deg, but her travel
happens to lie along her -Y axis while ours ended up 27.9 deg off +X.

This is done as a post-pass on purpose. Folding the same rotation into the solver's
axis alignment `A` does NOT work: `A` also appears in the expressive-chain target
construction (`C = A @ Rbody[0]`), which expresses the head/tail/ear delta in the
world frame at frame 0 rather than in the body frame, so rotating `A` silently
rewrites the head and ear motion (measured: legs moved <0.001 rad, but ear pitch
moved up to 2.42 rad). That is a real latent bug in the solver worth fixing on its
own; this pass sidesteps it entirely.

  python3 stage4/yaw_align.py --motion in.npz --out out.npz [--axis x|y]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--axis", choices=("x", "y"), default="x",
                help="world axis the net travel should run along")
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True)
d = {k: m[k] for k in m.files}
rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
trav = rp[-1, :2] - rp[0, :2]
n = np.linalg.norm(trav)
if n < 1e-6:
    print("[[ clip does not travel - nothing to align")
    np.savez(a.out, **d); raise SystemExit
trav /= n
want = np.array([1.0, 0.0]) if a.axis == "x" else np.array([0.0, 1.0])
th = np.arctan2(want[1], want[0]) - np.arctan2(trav[1], trav[0])
c, s = np.cos(th), np.sin(th)
R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
qz = np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])          # wxyz


def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


POS3 = ("root_pos", "tips_world", "ankles_world", "contacts_world",
        "planted_points_world", "support_patch_world", "sp_world", "knees_world",
        "contact_anchor_world", "root_contact_offset", "root_contact_offset_raw",
        "head_tail_positions", "ear_positions", "body_positions", "wrench_offset",
        "stance_polish_offset")
QUAT = ("root_quat", "body_rotations")
for k in POS3:
    if k in d:
        v = np.asarray(d[k], float)
        d[k] = (v.reshape(-1, 3) @ R.T).reshape(v.shape).astype(np.float32)
for k in QUAT:
    if k in d:
        v = np.asarray(d[k], float).reshape(-1, 4)
        d[k] = np.array([qmul(qz, q) for q in v]).reshape(
            np.asarray(m[k]).shape).astype(np.float32)
d["yaw_align_deg"] = np.array(np.degrees(th))
np.savez(a.out, **d)
assert np.array_equal(np.asarray(d["dof_positions"]), np.asarray(m["dof_positions"]))
rp2 = np.asarray(d["root_pos"] if "root_pos" in d else d["body_positions"][:, 0], float)
t2 = rp2[-1, :2] - rp2[0, :2]; t2 /= np.linalg.norm(t2)
print(f"[[ yaw-aligned by {np.degrees(th):+.1f} deg: net travel now "
      f"({t2[0]:+.3f},{t2[1]:+.3f}), i.e. {np.degrees(np.arctan2(t2[1], t2[0])):+.1f} deg "
      f"from +X | joint angles unchanged (verified)")
print(f"[[ wrote {a.out}")
