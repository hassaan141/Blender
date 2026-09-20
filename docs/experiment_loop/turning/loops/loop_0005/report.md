# loop_0005

KEEP: Viewed stride andfull overview: more discernible BLrecovery, no new hop/fall/high step. Both main andlimit turns survive and track yaw. BLclearance6.50/8.76mm versus4.27/6.66; right maxsat9.51 versus12.29%. Left rollRMS rises7.74 from6.69 largely sustainedlean: rollstd4.66 versus4.30; rightstd3.83 unchanged. Accept overall incremental foot-quality improvement, not success. Leftduty.858 andmaximum-yawdrag/saturation still fail; continue contact-consistency experiment.

Parent: loop_0004/A

Defect: Rear-left inside-turn swing remains too low, especially at positive maximum yaw.

Evidence: Loop4 BL clearance/contact: left4.27mm/.896,right6.66mm/.839,left_max2.94mm/.950. Stable yaw and survival in all cases; phase/timing correction helped but left still skims.

Hypothesis: Add only3mm smooth rear-left task-space lift within the existing corrected swing windows, with matching IK/FK channels; retain cadence, contact timing and all reward weights.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1580 | 0.0562 | 10.821 | left bl dragging, left saturation, left_max bl dragging, left_max saturation, right_max saturation, straight saturation, max vx | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
