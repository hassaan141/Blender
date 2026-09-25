# loop_0008

REJECT: Dense comparison montage shows no convincing overall visual improvement over loop7. Survival 100%, mean vx unchanged -.183; acceleration 7.73 vs7.98 and action-rate .01239 vs.01359 improve, but RR knee saturation rises 5.0 to7.29%, mean saturation1.89 to2.02%, vx std .0651 to.0661 and roll5.38 to5.46. Yaw remains .269rad/s and below-target clearance persists at rear. Retain loop7. Eight total loops reached; no loop9 authorized.

Parent: loop_0007/A

Defect: Residual rapid corrections contribute to remaining yaw oscillation and velocity pulsing.

Evidence: Loop7 vx std .0651, yaw RMS .273. Trace spectrum: 35.6% raw-action power, 34.6% yaw-rate power and 21.8% vx power above 3Hz; roll is dominated by gait cadence. Geometry/contact edits have already improved planting and slip.

Hypothesis: Lower residual EMA alpha from .30 to .20, preserving reference geometry, contact timing, cadence, residual scale and rewards, so PPO adapts balance corrections to a smoother action stream.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | -0.1828 | 0.0661 | 7.735 | vx std, yaw drift/oscillation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
