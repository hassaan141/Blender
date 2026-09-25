# loop_0008

REJECT: Eight-loop budget reached. Mandatory joint-limit failure at maximumleftyaw: BL_SY_J .420264rad versus.42, violation.000264>1e-5. Bothvideo montages inspected: stable coordinatedmainturns but persistentleftlean andincompleteBLunloading. FLpenaltyreducesleftsaturation10.14 versus12.85%, but shifts load into rearabductionlimit. Cannot retain. Finalbest remainsloop5; natural-gait goal NOTfullymet. No loop9.

Parent: loop_0005/A

Defect: Front-left support loading remains concentrated despite stable turns and improved rear-left recovery.

Evidence: Retainedloop5 FLknee saturation12.85%left,13.33%maximumright and14.86%maximumleft; most saturation overlaps intendedBLswing. Loop6 contact correction crossedBLabductionlimit; loop7geometry traded lowerleftloading forrightmax17.22%, so both rejected.

Hypothesis: From retainedloop5 increase only existing fl_torque_weight .03→.06. With current geometry and cadence fixed, modest additional cost on FLSP/knee effort should favor less concentrated support loading without sacrificing the learned turn or adding more lift.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1621 | 0.0509 | 10.578 | left bl dragging, left saturation, right saturation, left_max bl dragging, left_max joint limits, left_max saturation, right_max saturation, straight saturation, joint limit violation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
