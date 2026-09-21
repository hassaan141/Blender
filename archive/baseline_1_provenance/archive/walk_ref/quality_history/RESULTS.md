# Softer reference-guided gait: six-experiment result

Retained **quality Run 05**, warm-started through accepted Runs 02 and 04 from the user's refinement Run 05 baseline. Six experiments were completed; none changed actuator physics, gains, effort limits, reference motion, global phase rate, cadence, training commands, or the speed reward. Forward speed was a constraint, never a ranked objective.

## Diagnosis before reward changes

See [DIAGNOSIS.md](DIAGNOSIS.md), [reference_asymmetry.json](reference_asymmetry.json), and baseline/diagnostic_trace.npz. The baseline rollout exactly reproduced the retained scalar evaluation before training began.

The existing reference asks for about 40% larger FL SP excursion and 82% larger FL knee excursion than FR. FL SP saturation is more frequent during swing, and about 90% of its saturated samples oppose joint velocity, consistent with high braking/trajectory demand. FL knee saturation occurs in low-foot support phases, with almost twice the FR knee's residual-adjusted target error. Its residual RMS is actually smaller than FR's, so an oversized FL residual alone does not explain the problem. All legs retain identical SP/knee gains and limits. Contact timing uses the environment's height proxy; no contact forces were measured, so exact load sharing is not established.

## Result

| Measure | Retained baseline | Quality Run 05 |
|---|---:|---:|
| Survival after 1 s settling | 100%, 60 s | 100%, 60 s |
| Warmup falls | 0 | 0 |
| Mean vx | 0.198253 | 0.192935 m/s |
| vx std | 0.051268 | 0.052690 m/s |
| Max vx | 0.385477 | 0.365836 m/s |
| FL SP saturation | 17.014% | 10.417% (−38.8%) |
| FL knee saturation | 21.181% | 15.972% (−24.6%) |
| Joint acceleration RMS | 12.5365 | 11.0002 rad/s² (−12.3%) |
| Detrended body-height RMS | 2.3599 | 1.5098 mm (−36.0%) |
| Body-height p95–p05 | 8.2759 | 4.9490 mm |
| Vertical velocity RMS | 0.06720 | 0.06233 m/s (−7.2%) |
| Reference phase-cycle period | 2.121914 | 2.121914 s |

Both FL joints and mean overall saturation improve; no net saturation increase was transferred to other joints. The FL/FR knee saturation gap drops from 15.07 to 9.79 percentage points. The gait is not artificially mirrored: its existing reference and authored sequence remain intact.

These improvements are specific: roll RMS rises slightly from 5.138° to 5.482°, and residual action-rate RMS rises from 0.017374 to 0.018130 rad/step. Neither is concealed or substituted for the requested physical acceleration and vertical-bobbing measures. Reference tracking RMS changes from 0.028671 to 0.030015 rad, within the predeclared guard. Speed variance remains below 0.06.

## Retained changes

Three cumulative reward weights: FL SP/knee normalized torque cost 0.03; body-height cost 0.05 around the measured baseline mean height 0.1716569811 m; vertical COM velocity cost 0.01. EMA remains at the baseline 0.3. The original acceleration penalty 0.2, action-rate penalty 0.06, residual scale 0.3, phase multiplier 1.4 and reference are retained. Height/velocity scales are 0.01 m and 0.1 m/s respectively; no new actuator command or physics setting was introduced.

Each experiment changed at most two related parameters, ran 100 PPO iterations/2400 steps with 512 environments and seed 42, and evaluated the predetermined final checkpoint with the same 60-second evaluator plus read-only sidecar diagnostics. See [leaderboard.csv](leaderboard.csv) and [PROTOCOL.md](PROTOCOL.md). Each run preserves its hypothesis, parent, source snapshots, exact commands, logs, checkpoint, scalar metrics and diagnostic trace.

## Artifacts and reproduction

- Checkpoint: [run_05/policy.pt](run_05/policy.pt).
- Complete metrics: [best_metrics.json](best_metrics.json).
- Video: [best_render/eval.mp4](best_render/eval.mp4).
- From the repository root: `bash docs/walk_ref/quality_runs/best_render/render_command.sh`.
- Runtime sources match the selected run's snapshots. The unchanged reference is docs/walk_ref/refinement_runs/run_05/reference.npz.
- The original evaluator's `_summarize` and `_format` function ASTs are unchanged. Added optional diagnostics record extra data without changing standard metric definitions; baseline equivalence and final replay are checked.

Final verification: the 60.959-second, 1280×720, 24 fps video reproduces both scalar and diagnostic metrics exactly. Sampled visual review preserves the current gait. All 217 protected fingerprints match, source snapshots match the selected run, and all six decisions were independently recomputed. See completion_audit.json and best_render/visual_review.md.
