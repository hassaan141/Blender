"""Cut frames [a,b] out of a motion, keeping every per-frame field in step.

Used to ask whether a hard window fails ON ITS OWN or only because the open-loop
tracker arrived at it already displaced: the Stage-4 tracker initialises from
reference frame 0, so a sub-clip starting at the window begins there exactly.

  python3 stage4/subclip.py --motion motions/laidback_v4.npz --range 280-400 --out /tmp/x.npz
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--range", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = np.load(a.motion, allow_pickle=True)
f0, f1 = (int(x) for x in a.range.split("-", 1))
T = m["dof_positions"].shape[0]
f0, f1 = max(0, f0), min(T - 1, f1)
d = {}
for k in m.files:
    v = m[k]
    d[k] = v[f0:f1 + 1] if (isinstance(v, np.ndarray) and v.ndim >= 1
                            and v.shape[0] == T) else v
np.savez(a.out, **d)
print(f"[[ frames {f0}-{f1} ({f1-f0+1}) of {a.motion} -> {a.out}")
