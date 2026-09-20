# loop_0001

REJECT: All candidates failed metrics gates

Parent: locomotion1_quality_run05

Defect: Forward-trained residual policy and positive velocity metadata conflict with reversed playback required for backward travel.

Evidence: Controller _pre_physics_step already uses signed 1.4*cmd_vx/.66 and indexes contacts at that same phase. _sample_ref and reset consume unscaled, positive-direction stored joint/root velocities. Attempt03 forward-only vx_range=(.2,.3).

Hypothesis: Traversing the existing reference and contacts backward, correcting stored velocity channels to the signed playback derivative, and fine-tuning the existing residual policy at -.20 m/s will produce backward stepping without redesigning poses or physics.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | -0.1366 | 0.0805 | 9.163 | mean vx, vx std, br_knee major saturation regression, yaw drift/oscillation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
