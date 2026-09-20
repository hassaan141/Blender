# loop_0003

KEEP: Provisional progress only: all five cases survive; main yaw +.428/-.410 tracks commands and actual vx .157/.136 improves requested travel, acceleration 9.92/9.61 below fixed cap. Inherited BL dragging remains unacceptable (duty98/95%, clearance2.23/2.91mm); saturation gates also fail. Not a finished natural walk. Next single defect is BL recovery.

Parent: loop_0002/A

Defect: Actual forward travel remains .10–.11m/s during otherwise stable coordinated turns, below requested .15–.20m/s.

Evidence: Loop2 survives all five cases and tracks yaw within .15rad/s. Matched original forward policy also travels .0997m/s at .175 command; canonical .25 command gives .1929. Native stand-blend threshold .198 attenuates stride below that command.

Hypothesis: Calibrate the existing speed input to .215m/s (training .190–.240) while retaining reference geometry, yaw reward and heading-local observations. Interpolation between measured forward baselines predicts actual travel near .15m/s.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1573 | 0.0538 | 10.856 | right saturation, left_max saturation, right_max saturation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
