# loop_0003

KEEP: Provisional overall backward improvement: 100% survival, FL clearance 9.4→35.3mm and front support duty balanced .617/.650, roll 7.73→6.58deg, vx -.137→-.159, std .0805→.0759, acceleration9.16→8.70, saturation2.81→2.45%. Montage shows clearer FL stepping and less lateral weight shift. Yaw-rate RMS slightly higher .286 vs .265; not goal-qualified.

Parent: loop_0001/A

Defect: Lateral rocking and uneven front support dominate the visible gait.

Evidence: loop_0001 comparison_stride.jpg; roll RMS 7.73 degrees, FL contact duty .831 vs FR .626; native lateral paw excursions 47–58 mm. Loop 2 ended before training because copy2 retained parent read-only permissions; corrected using a fresh writable copy, same physical hypothesis.

Hypothesis: Reduce lateral task-space paw excursions 30% about each foot mean, retaining stance width and x/z path, to reduce lateral rocking and improve support consistency.

| Arm | vx mean | vx std | acceleration | Gate failures | Video |
|---|---:|---:|---:|---|---|
| A | -0.1590 | 0.0759 | 8.696 | mean vx, vx std, yaw drift/oscillation | [comparison](A/comparison.mp4) |

Full metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.
