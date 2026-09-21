# loop_0004

KEEP: Viewed dense stride and60s overview: stable continuous turning, no hop/fall; rear-left swing improves visibly modestly and numerically on both sides. Main BL duty .896/.839 versus .983/.955, clearance4.27/6.66 versus2.23/2.91mm. Yaw+.425/-.422 preserved, all5survive, accelerationwithin fixedcap. Still fails left clearance and saturation/slip gates; provisional improvement only, continue targeted BL lift.

Parent: loop_0003/A

Defect: Rear-left dragging during otherwise stable left/right turns

Evidence: Loop3 BL duty .983/.955 and p95 clearance2.23/2.91mm; original forward baseline has same inherited defect. Prior natural_walk attempt03 showed corrected collision contact definition and6mm recovery improved BL clearance.

Hypothesis: Enable only the known BL recovery channels from reference02b and their matching collision-hull contact definition; preserve all other reference channels, yaw geometry, cadence, rewards weights and physics.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1546 | 0.0565 | 11.092 | left bl dragging, left saturation, right saturation, left_max bl dragging, left_max slip, left_max saturation, right_max saturation, straight bl dragging, straight saturation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
