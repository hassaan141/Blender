"""Stage 4 fix #2 - retime ONE segment, leaving the rest of the clip at speed.

Uniform retiming slows the whole performance; STMR-style local retiming stretches
only the infeasible window. Every pose is preserved exactly - just revisited on a
slower schedule - so velocities/accelerations in that window scale by 1/f, 1/f^2
while the rest of the clip is untouched.
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--factor", type=float, default=2.0, help=">1 = slower in the window")
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
T = m["dof_positions"].shape[0]
s, e = max(0, a.start), min(T - 1, a.end)
# new source-time samples: unchanged outside [s,e], stretched inside
n_in = int(round((e - s) * a.factor))
src = np.concatenate([np.arange(0, s), np.linspace(s, e, n_in, endpoint=False),
                      np.arange(e, T)])
T2 = len(src)

def resamp(x):
    x = np.asarray(x, float); flat = x.reshape(T, -1)
    o = np.stack([np.interp(src, np.arange(T), flat[:, c]) for c in range(flat.shape[1])], 1)
    return o.reshape((T2,) + x.shape[1:])

for k in ("dof_positions", "body_positions", "body_rotations", "root_pos", "root_quat",
          "head_tail_positions", "ear_positions"):
    if k in d: d[k] = resamp(d[k]).astype(np.float32)
for k in ("body_rotations", "root_quat"):
    if k in d:
        q = d[k].astype(float); q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        d[k] = q.astype(np.float32)
# Resample EVERY per-frame array, not just "contacts": the boolean schedules and
# the world-point tracks are consumed downstream and must stay in step with the
# motion (same fix as stage4/retime.py).
idx = np.clip(np.round(src).astype(int), 0, T - 1)
for k in list(d):
    v = np.asarray(d[k])
    if v.ndim >= 1 and v.shape[0] == T and k not in (
            "dof_positions", "body_positions", "body_rotations", "root_pos",
            "root_quat", "head_tail_positions", "ear_positions", "dof_velocities"):
        d[k] = v[idx] if v.dtype == bool or v.dtype.kind in "iub" else resamp(v).astype(v.dtype)
dt = 1.0 / float(m["fps"])
d["dof_velocities"] = np.gradient(d["dof_positions"].astype(float), dt, axis=0).astype(np.float32)
d["stage4_retime"] = np.array([s, e, a.factor], float)
np.savez(a.out, **d)
v = np.abs(d["dof_velocities"])
print(f"[[ retimed frames {s}-{e} x{a.factor} : {T} -> {T2} frames "
      f"({T/float(m['fps']):.2f}s -> {T2/float(m['fps']):.2f}s)")
print(f"[[ max |joint vel| legs {v[:,:12].max():.2f} (lim 10) | expr {v[:,12:].max():.2f} (lim 8)")
print(f"[[ wrote {a.out}")
