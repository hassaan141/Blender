# loop_0006

REJECT: Mandatory joint-limit failure: BL_SY_J .420205rad exceeds.42 at maximumleft yaw, violation.000205>1e-5. Video shows stable coordinated mainturns and improved footclearance, but strongerleftlean; cannot retain a limit-violating candidate despite slip/clearance improvements. All5survive,mainyaw+.438/-.413. Keep loop5; next target lateral reference demand before more reward changes.

Parent: loop_0005/A

Defect: Uneven support transfer leaves left-turn rear-foot skimming and front-left knee saturation.

Evidence: Loop5 improves BL clearance6.50/8.76mm but left duty.858,max-left.903 and saturation12.85/14.86%. Contact audit: legacy tip test disagrees with collision geometry25–33% on front feet; authored front contact windows disagree with recovery geometry26–41%. Most FLknee saturation occurs during intended BLswing.

Hypothesis: Make contact supervision consistent for all feet: derive remaining feet stance/swing labels from existing recovery strokes and replace their tip-height contact tests with collision-hull clearance. Keep geometry, cadence, contact reward total weight.10 and physics unchanged.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1537 | 0.0521 | 10.571 | left saturation, right saturation, left_max bl dragging, left_max joint limits, left_max saturation, right_max saturation, straight saturation, joint limit violation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
