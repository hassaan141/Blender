# loop_0004

REJECT: 60 s survival 100%, vx -.1701/std .0691 improve, but roll rises 6.58 to 7.14 deg, acceleration 8.70 to 9.38, mean saturation 2.45 to 3.76%, RR knee 8.26% fails regression gate. Dense comparison montage retains uneven rocking and no convincing planting improvement. Faster is not enough. Retain loop3.

Parent: loop_0003/A

Defect: Backward velocity remains pulsed; native fore-aft path reverses direction during nominal planted stance.

Evidence: Loop3 survived and reduced rocking, but vx=-.159/std=.0759. stance_diagnosis.json: FL/FR/BR have wrong-direction paw velocity during 33–38% of labelled stance. Reference foot x ranges already approximately match the required stride.

Hypothesis: Blend 35% of a monotone planted-stance progression and smooth swing return into the existing fore-aft paths, keeping range, contacts, lateral geometry and cadence unchanged, to reduce braking/slip and speed pulses.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | -0.1701 | 0.0691 | 9.376 | vx std, br_knee major saturation regression, yaw drift/oscillation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
