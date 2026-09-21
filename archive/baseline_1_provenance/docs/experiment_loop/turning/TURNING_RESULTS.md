# Turning campaign results

Eight loops completed. Best: **loop_0005/A**, best experimental turning candidate; full natural-gait gates NOT met. Training stopped at the requested limit.

[Left/right video](loops/loop_0005/A/left_right.mp4) · [Forward/left/right comparison](final_comparison.mp4) · [Checkpoint](loops/loop_0005/A/policy.pt) · [Exact runtime bundle](BEST.json) · [Full metrics](TURNING_RESULTS.json)

| Case | Survival | vx m/s | vx std | Yaw command / actual rad/s | Accel rad/s² | Slip m/s | Max joint saturation | BL duty / clearance mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| left | 100% | 0.158 | 0.056 | +0.40 / +0.431 | 9.70 | 0.213 | 12.85% | 85.8% / 6.50 |
| right | 100% | 0.134 | 0.049 | -0.40 / -0.423 | 9.75 | 0.176 | 9.51% | 82.2% / 8.76 |
| left_max | 100% | 0.155 | 0.062 | +0.60 / +0.625 | 10.82 | 0.231 | 14.86% | 90.3% / 4.50 |
| right_max | 100% | 0.128 | 0.045 | -0.60 / -0.624 | 10.44 | 0.178 | 13.33% | 81.4% / 8.59 |
| straight | 100% | 0.144 | 0.052 | +0.00 / +0.002 | 9.50 | 0.181 | 10.14% | 82.5% / 8.56 |

Remaining gate failures: left bl dragging, left saturation, left_max bl dragging, left_max saturation, right_max saturation, straight saturation.

| Loop | Change | Decision |
|---|---|---|
| loop_0001 | {'turn_geometry_gain': 0.7} | REJECT |
| loop_0002 | {'turn_heading_local': 1} | KEEP |
| loop_0003 | {'turn_command_vx': 0.215} | KEEP |
| loop_0004 | {'turn_bl_recovery': 1} | KEEP |
| loop_0005 | {'bl_lift_mm': 3} | KEEP |
| loop_0006 | {'contact_timing_mode': 1, 'all_contacts_corrected': 1} | REJECT |
| loop_0007 | {'turn_lateral_gain': 0.45} | REJECT |
| loop_0008 | {'fl_torque_weight': 0.06} | REJECT |

KEEP before full qualification means provisional progress, not a claim that all natural-gait gates passed. Loop 6 was rejected for a BL abduction limit crossing; loop 7 for worse right-side saturation and left lean; loop 8 for another rear-left abduction limit crossing. Each detailed report records the visual review and tradeoffs.

The decisive stability change was expressing observations relative to heading, so a turned robot no longer presents unfamiliar world-oriented observations. Rear-left recovery geometry and collision-based contact feedback improved an inherited near-continuous drag. More lift helped clearance but increased left lean; consistent all-foot contact supervision improved support metrics but crossed a joint limit. These are diagnostic findings, not proof of a fully natural gait.

Forward and backward baselines remain unchanged and separate. This is a turning extension with 71 observations and an adjacent yaw-reference bank; do not load it with the original 70-observation evaluator. The comparison uses the canonical forward speed and the candidate turning speed, so it is a visual benchmark rather than a matched-command test.

Reproduce one 60-second turn into a new directory:

```bash
python3 rl/bingo_rl/experiment_loop/turning/reproduce.py \
  --candidate docs/experiment_loop/turning/loops/loop_0005/A \
  --out docs/experiment_loop/turning/replay_left_new --yaw 0.4 --seconds 60
```

Use `--yaw -0.4` and a different fresh output directory for the right turn. Full native argv, environment settings and training logs are in the sealed candidate folder. No loop 9 was started.

Final validation: all eight loop seals and 2,385 protected entries verified; 18 runner tests passed. Matching video-mode replay reproduced the first 10 seconds exactly. No-video mode has a different reset/render initialization path and should not be used for bitwise trajectory comparisons. Intermediate ±0.2 rad/s probes survived 30 seconds with measured yaw+.216/−.213 and zero joint-limit violations. No training or loop process remains active.
