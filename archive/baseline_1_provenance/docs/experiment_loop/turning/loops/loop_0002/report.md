# loop_0002

KEEP: Reviewed dense stride and full-duration left/right montages. Stable continuous turning through multiple headings with visible alternating four-leg lift, no new crouch/hop or collapse. All five cases survive100%, zero limits. Main yaw +.327/-.350 vs+/-.4; max yaw+.476/-.534 vs+/-.6. Acceleration6.9–7.6, mean slip .172–.184, max joint saturation9.03%. Clear overall improvement from failed loop1. Retain provisionally: actual vx .10–.112 misses speed gate.

Parent: locomotion1_quality_run05

Defect: Heading changes move policy observations outside the straight-walking coordinate distribution, causing asymmetric steering and eventual falls.

Evidence: Loop1 left falls at18.5s, right yaw=-.009 for-.4 command. Native compute_obs exposes world-axis orientation/velocities/foot offsets. Full evaluation and fall montage reviewed.

Hypothesis: Express orientation, linear/angular velocity and foot offsets relative to current heading, retaining roll/pitch and native gait phase; leave geometry, rewards and training budget unchanged.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | 0.1044 | 0.0515 | 7.576 | left forward smoothness, right forward smoothness, left_max forward smoothness, right_max forward smoothness, straight forward smoothness, mean vx | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
