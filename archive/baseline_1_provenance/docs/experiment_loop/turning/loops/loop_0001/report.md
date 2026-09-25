# loop_0001

REJECT: Full-duration overview and left-fall montage inspected: left tips over at18.5s; right survives but barely turns. Left/right measured yaw +.063/-.009 for +/-.4 commands; maximum-left also falls. Straight vx .0987 matches original policy .0997 at same .175 command, so low speed is inherited calibration rather than a training regression. Mandatory survival and yaw tracking fail; next isolate heading-local observations.

Parent: locomotion1_quality_run05

Defect: The inherited controller has no learned yaw-command response; task-space steering alone is insufficient.

Evidence: Forward baseline has no yaw channel. Smoke test confirms migrated checkpoint loads and the yaw-conditioned reference executes without physics modifications.

Hypothesis: Warm-start the native forward policy with one zero-initialized yaw observation, conservative body-twist reference geometry and one yaw-tracking reward, learning both signs together at vx .15–.20.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.0872 | 0.0779 | 8.851 | left survival, left joint limits, left forward smoothness, left yaw tracking, right forward smoothness, right yaw tracking, right yaw oscillation, left_max survival, left_max forward smoothness, left_max yaw tracking, right_max saturation, right_max forward smoothness, right_max yaw tracking, right_max yaw oscillation, straight forward smoothness, survival, mean vx, vx std, joint limit violation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
