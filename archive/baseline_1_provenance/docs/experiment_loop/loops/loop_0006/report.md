# loop_0006

REJECT: Dense comparison montage shows no convincing overall gain; rear lift is reduced. Survival 100%, vx unchanged -.175, std .07195 and roll 5.10 improve modestly, but RR knee saturation rises 4.51 to 6.74%, mean saturation 1.65 to 1.86%, rear clearance falls from 12–13mm to 9.6–10.6mm and yaw RMS .283 to .298. Retain loop5 for better balanced support and torque margin.

Parent: loop_0005/A

Defect: Remaining lateral rocking and cyclic yaw during otherwise improved backward stepping.

Evidence: Loop5 roll 5.47 deg, yaw-rate RMS .283 rad/s and vx std .0754 remain too large. Lateral scaling in loop3 previously reduced rocking; corrected contacts in loop5 reduced knee saturation/slip without geometry changes.

Hypothesis: Reduce local lateral foot excursions from 70% to 45% of original, about the unchanged mean stance width. Keep fore-aft trajectory, lift, contacts and cadence fixed to reduce lateral impulses and body oscillation.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | -0.1752 | 0.0719 | 7.979 | vx std, yaw drift/oscillation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
