"""Smooth the loop-seam discontinuity in a cyclically-replayed (non-canonical)
locomotion reference, for the RL residual env only. Does NOT touch the canonical
Stage 4 file -- writes a derived copy.

The walk_ref shaking diagnostic (rl/tools/diagnose_walk_ref_shaking.py) measured
frame[-1] -> frame[0] wrapping produces the single largest per-step dof_positions
jump in the whole clip (~0.33-0.41 rad on one joint) because the source clip is a
single authored pass, not an actual treadmill loop -- there's no reason
frame[-1] and frame[0] should already match.

Standard motion-loop technique: distribute HALF the frame[-1]-vs-frame[0] gap,
per joint, as a smoothstep-weighted correction added over the last W frames and
subtracted over the first W frames. At the seam itself both corrections converge
to the same midpoint value, so frame[-1]_new == frame[0]_new exactly; W frames
away from the seam on either side the correction has decayed to ~0, so the rest
of the authored motion is untouched. Only dof_positions is corrected (that's the
only field `bingo_walk_ref_env.py`'s PD target reads); dof_velocities is
recomputed by finite difference afterward so it stays consistent.

    python3 stage4/smooth_loop_seam.py --motion motions/bingo_walk_v4_upright_grounded.npz \
        --out motions/bingo_walk_v4_upright_grounded_loopsmooth.npz --window 12
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=12, help="frames of correction ramp on each side")
a = ap.parse_args()

m = np.load(a.motion, allow_pickle=True)
d = {k: m[k] for k in m.files}
q = d["dof_positions"].astype(np.float64).copy()
T = q.shape[0]
W = min(a.window, T // 2)

delta = q[0] - q[T - 1]                      # the gap that must close to zero
s = np.linspace(1.0 / W, 1.0, W)             # smoothstep ramp, 0 (far) -> 1 (at seam)
smooth = 3 * s ** 2 - 2 * s ** 3

# last W frames: correction ramps 0 -> +delta/2, closing half the gap by frame T-1
q[T - W:T] += 0.5 * delta[None, :] * smooth[:, None]
# first W frames: correction ramps (at frame 0, s=1 -> -delta/2) -> 0 (far from seam)
q[0:W] -= 0.5 * delta[None, :] * smooth[::-1][:, None]

max_seam_jump_before = float(np.abs(delta).max())
max_seam_jump_after = float(np.abs(q[0] - q[T - 1]).max())

d["dof_positions"] = q.astype(np.float32)
dt = 1.0 / float(m["fps"])
d["dof_velocities"] = np.gradient(q, dt, axis=0).astype(np.float32)
d["loop_seam_smoothed_window"] = np.array(W)

np.savez(a.out, **d)
print(f"[[ loop-seam max joint jump: {max_seam_jump_before:.4f} rad -> {max_seam_jump_after:.6f} rad "
      f"(window {W} frames each side)")
print(f"[[ wrote {a.out}")
