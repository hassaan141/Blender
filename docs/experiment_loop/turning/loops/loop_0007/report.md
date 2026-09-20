# loop_0007

REJECT: Both video montages inspected. Stable coordinatedturns and improvedBLclearance, but leftlean grows8.80degRMS versus7.74; rightmain saturation13.89 versus9.51%, maximumright17.22 versus13.33% (+29%). Reducedlateralreference shifts loading between sides rather than improving overallgait. Allsurvive,no limits,accelwithincap, but rejectphysicaltradeoff. Retainloop5; finalrun targetsFLload only.

Parent: loop_0005/A

Defect: Left-turn side loading and rear-left abduction approach the joint limit; foot clearance and front-knee loading remain asymmetric.

Evidence: Loop6 rejected: BL_SY reaches.420205 against.42 limit at+.6yaw; qref.2292 plus residual.1883 at event. Left rollmean-7.05deg, remainingBLduty.886,maxsat11–12%. Retainedloop5 has same yaw geometry with less severe abduction but leftlean/drag/saturation.

Hypothesis: From retainedloop5 reduce only task-space lateral turning gain .7→.45, retaining inside/outside fore-aft stride asymmetry and all learned control settings. This reduces reference-imposed lateral leg travel and lateral support demand.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1601 | 0.0560 | 10.939 | right saturation, left_max bl dragging, right_max saturation, straight saturation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
