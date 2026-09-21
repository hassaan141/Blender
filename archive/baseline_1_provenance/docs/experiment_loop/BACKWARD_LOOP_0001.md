# Backward loop 0001 — REJECT

Exactly one training attempt completed: 300 iterations, 512 environments, seed 42. No loop 2. Locomotion 1 and attempt 03 remain untouched.

## Hypothesis

Traversing the existing reference and contacts backward, correcting stored velocity channels to the signed playback derivative, and fine-tuning the existing residual policy at -.20 m/s will produce backward stepping without redesigning poses or physics.

The original controller already reverses phase with negative velocity; this trial also corrects stored velocity metadata used by the reward and resets. All pose/contact samples, physics, gains, limits and reward weights remain unchanged. Signed phase rate is −0.424242; full reference period is 2.651786 seconds and contains multiple steps.

## Result

The learned policy completed 60 seconds without falling, unlike the seed’s backward preflight (fall at approximately 1.29 seconds including settling). It travelled backward but missed −0.20 m/s, with excessive speed variation and yaw oscillation. Lower acceleration and mean saturation do not override these failures.

| Metric | Locomotion 1 forward benchmark | Attempt 03 forward seed | Backward candidate |
|---|---:|---:|---:|
| Survival | 1.00000 | 1.00000 | 1.00000 |
| Mean vx (m/s) | 0.19294 | 0.18631 | -0.13658 |
| vx std (m/s) | 0.05269 | 0.04801 | 0.08047 |
| Joint acceleration RMS (rad/s²) | 11.00019 | 11.53409 | 9.16329 |
| Action rate RMS (rad/step) | 0.01813 | 0.01838 | 0.01204 |
| Mean torque saturation (%) | 4.67593 | 4.97685 | 2.80671 |
| Body roll RMS (deg) | 5.48211 | 4.92662 | 7.72790 |
| Body pitch RMS (deg) | 2.01058 | 2.02623 | 1.23834 |
| Body bob RMS (mm) | 1.50977 | 1.48979 | 1.98449 |

Maximum absolute vx: 0.32479 m/s. Mean contact-slip RMS across feet: 0.20737 m/s (seed 0.21814). Net heading change 0.611°, but yaw-rate RMS 0.265 rad/s. Joint-limit violation: 0.000000 rad. World displacement [-9.958309173583984, 1.1291030645370483, 9.725987911224365e-05] m; this differs from body-frame velocity.

| Foot | Contact duty | Clearance p95 (mm) | Contact-slip RMS (m/s) | Median touchdown interval (s) |
|---|---:|---:|---:|---:|
| fl | 0.8313 | 9.443 | 0.2390 | 0.9167 |
| fr | 0.6264 | 44.896 | 0.2434 | 0.9583 |
| bl | 0.7035 | 11.333 | 0.1657 | 0.8333 |
| br | 0.6271 | 18.600 | 0.1813 | 0.8333 |

Contacts/slip are collision-geometry proxies, not force measurements. Forward runs are morphology/quality benchmarks, not matched backward controls; the preflight supplies the untrained backward comparison.

## Decision and visible defect

Backward stepping is sustained, but lateral rocking and uneven foreleg support remain. FL spends 83.1% in geometric contact and achieves only 9.4mm p95 clearance versus 44.9mm on FR. Front-foot slip remains substantial. Not a natural, relaxed backward walk.

Failed gates: mean vx, vx std, br_knee major saturation regression, yaw drift/oscillation.

Visual review used sampled frames across the clip and a dense stride sequence, not continuous human playback. The no-drag/no-shuffle requirement is not established. The largest remaining defect is uneven front-foot support with body rocking. No new adaptation was attempted.

## Artifacts

- [Candidate video](loops/loop_0001/A/evaluation/eval.mp4)
- [Locomotion 1 versus backward comparison](loops/loop_0001/A/comparison.mp4)
- [Checkpoint](loops/loop_0001/A/policy.pt)
- [Backward reference](loops/loop_0001/A/reference.npz)
- [Metric comparison](loops/loop_0001/A/metric_comparison.json)
- [Loop report](loops/loop_0001/report.md)
- [Visual/integrity audit](reviews/loop_0001_visual.json)

## Changed files and execution

New backward adapter, worker, reference/diagnostics analyzer, and kinematic-preview script under `rl/bingo_rl/experiment_loop/`; `core.py` gained direction-aware gates and two bounded backward settings. New config/proposal/report/artifacts under `docs/experiment_loop/`, plus README usage notes. No historical controller, URDF, actuator, physics, Stage asset, checkpoint or reference was modified.

The existing loop runner executed planning input → reference preparation → kinematic review → backward seed preflight → one PPO training → 60s evaluation → diagnostics → comparison rendering → automatic REJECT → immutable report. Agent visual review is attached separately after sealing. All three execution receipts succeeded, the checkpoint tensors are finite and nine policy tensors changed from the seed. All 940 historical protection hashes and the complete before/after protection inventory match. Champion unchanged; loop 2 not started.

Post-evaluation clarification: yaw-rate RMS decreased from 0.41066 rad/s in attempt 03 to 0.26490 rad/s in this trial. It failed the predeclared 0.15 rad/s limit but is **not a regression versus the seed**. Right-rear knee saturation increased from 2.43056% to 7.91667%. The missed mean-speed and speed-variation gates independently require rejection.

A final infrastructure guard rejects backward-only settings in the generic forward worker/config. This guard was added after the sealed run and was not part of training; the run’s exact source snapshot remains untouched.
