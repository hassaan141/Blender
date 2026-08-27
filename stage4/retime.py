"""Stage 4 fix #2 - uniform temporal retiming (STMR-style) of a reference.

Resamples the motion onto a longer time base at the SAME fps, so every pose is
preserved exactly and only the speed changes. Joint velocities and accelerations
scale by 1/f and 1/f^2, which is the standard remedy for velocity/torque
infeasibility. It cannot change any pose, so it cannot create ground support.
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--factor", type=float, default=2.0, help=">1 = slower")
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True)
d = {k: m[k] for k in m.files}
T = m["dof_positions"].shape[0]
T2 = int(round(T * a.factor))
src = np.linspace(0, T - 1, T2)

def resamp(x):
    x = np.asarray(x, float)
    out = np.empty((T2,) + x.shape[1:], float)
    flat = x.reshape(T, -1)
    o = np.stack([np.interp(src, np.arange(T), flat[:, c]) for c in range(flat.shape[1])], 1)
    return o.reshape((T2,) + x.shape[1:])

for k in ("dof_positions", "body_positions", "body_rotations", "dof_velocities",
          "root_pos", "root_quat", "head_tail_positions", "ear_positions"):
    if k in d:
        d[k] = resamp(d[k]).astype(np.float32)
if "body_rotations" in d:                       # renormalise quaternions
    q = d["body_rotations"].astype(float)
    q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    d["body_rotations"] = q.astype(np.float32)
if "root_quat" in d:
    q = d["root_quat"].astype(float); q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    d["root_quat"] = q.astype(np.float32)
# Every per-frame array must be resampled, not just "contacts": the boolean
# schedules (source_contacts, physical_height_contacts) and the world-space point
# tracks are consumed downstream, and leaving them at the OLD length silently
# desynchronises them from the motion (finalize_motion_contacts then fails on a
# shape mismatch, which is the good case - a same-length array would just be wrong).
idx = np.clip(np.round(src).astype(int), 0, T - 1)
for k in list(d):
    v = np.asarray(d[k])
    if v.ndim >= 1 and v.shape[0] == T and k not in (
            "dof_positions", "body_positions", "body_rotations", "dof_velocities",
            "root_pos", "root_quat", "head_tail_positions", "ear_positions"):
        d[k] = v[idx] if v.dtype == bool or v.dtype.kind in "iub" else \
            resamp(v).astype(v.dtype)
dt = 1.0 / float(m["fps"])
d["dof_velocities"] = np.gradient(d["dof_positions"].astype(float), dt, axis=0).astype(np.float32)
d["stage4_retime_factor"] = np.array(a.factor)
np.savez(a.out, **d)
v = np.abs(d["dof_velocities"])
print(f"[[ retimed x{a.factor}: {T} -> {T2} frames @ {float(m['fps']):g} fps "
      f"({T/float(m['fps']):.2f}s -> {T2/float(m['fps']):.2f}s)")
print(f"[[ max |joint vel| legs {v[:,:12].max():.2f} (lim 10) | expr {v[:,12:].max():.2f} (lim 8) rad/s")
print(f"[[ wrote {a.out}")
