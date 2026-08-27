"""Freeze one reference frame into a constant-pose clip, to ask whether that POSE
is a static equilibrium under full physics - separating "the pose is impossible"
from "the transition into it is".

Used on Timid to prove every reference pose held for 5 s (so only the transitions
failed); use it the same way on any clip whose tracker diverges.

  python3 stage4/hold_frame.py --motion motions/eccentric_v4.npz --frame 10 \
      --seconds 5 --out /tmp/hold10.npz
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--frame", type=int, required=True)
ap.add_argument("--seconds", type=float, default=5.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True)
fps = float(m["fps"]); N = int(round(a.seconds * fps))
d = {}
for k in m.files:
    v = m[k]
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == m["dof_positions"].shape[0]:
        d[k] = np.repeat(v[a.frame:a.frame + 1], N, axis=0)
    else:
        d[k] = v
d["dof_velocities"] = np.zeros_like(np.asarray(d["dof_positions"], float), dtype=np.float32)
np.savez(a.out, **d)
print(f"[[ held frame {a.frame} of {a.motion} for {N} frames ({a.seconds:g} s) -> {a.out}")
