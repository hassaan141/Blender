"""Rate-limit the expressive chains (head, tail, ears) without touching the legs,
the root or the timing.

Why. DeadPan's physics run leaves the reference at frame 48 with all four paws
50 mm off the floor, and the reason is above the shoulders: over frames 39-47 the
head joints are pinned at their 6.0 N m limit doing 7.4-7.8 rad/s. head_roll
carries 0.708 kg - 29% of the robot's 2.46 kg - so a whip that fast throws the
whole body. The legs are innocent; retiming the window slows them too and costs
the momentum that was holding the marginal poses together.

This caps |dq/dt| on DOF 12-20 only, by the same forward-backward projection the
Stage-2 solver uses for its own rate limits, so no pose is re-authored - the
gesture simply cannot be performed faster than the cap. The cap is the physical
knob: the reaction wrench on the body scales with the limb's angular acceleration.

  python3 stage4/limit_expression.py --motion motions/deadpan_v4.npz \
      --out /tmp/dp_soft.npz --max-vel 4.0
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-vel", type=float, default=4.0, help="rad/s cap on DOF 12-20")
ap.add_argument("--start", type=int, default=None)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--iterations", type=int, default=12)
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True); d = {k: m[k] for k in m.files}
q = m["dof_positions"].astype(float).copy()
T = len(q); fps = float(m["fps"]); dt = 1.0 / fps
s = 0 if a.start is None else max(0, a.start)
e = T if a.end is None else min(T, a.end + 1)
step = a.max_vel * dt
before = np.abs(np.gradient(q, dt, axis=0)[:, 12:]).max()
seg = q[s:e, 12:]
for _ in range(a.iterations):
    for i in range(1, len(seg)):
        seg[i] = np.clip(seg[i], seg[i - 1] - step, seg[i - 1] + step)
    for i in range(len(seg) - 2, -1, -1):
        seg[i] = np.clip(seg[i], seg[i + 1] - step, seg[i + 1] + step)
q[s:e, 12:] = seg
after = np.abs(np.gradient(q, dt, axis=0)[:, 12:]).max()
dq = np.abs(q[:, 12:] - m["dof_positions"][:, 12:])
print(f"[[ expression rate limit {a.max_vel:g} rad/s over frames {s}-{e-1}: "
      f"max |dq/dt| {before:.2f} -> {after:.2f} rad/s")
print(f"[[ expressive joint change: mean {np.degrees(dq.mean()):.2f} deg "
      f"max {np.degrees(dq.max()):.2f} deg")
d["dof_positions"] = q.astype(np.float32)
d["dof_velocities"] = np.gradient(q, dt, axis=0).astype(np.float32)
d["expression_rate_cap"] = np.array(a.max_vel)
np.savez(a.out, **d)
print(f"[[ wrote {a.out}")
