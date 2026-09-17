# Refinement results

The eight-experiment search is complete. The retained checkpoint is refinement Run 05. Its mean speed is 0.198253 m/s, so the full target is **not** met: it remains just below 0.20 m/s. All other speed gates and the fixed acceleration ceiling pass.

## Selection correction and starting checkpoint

The original recorder compared acceleration to the changing parent. It now compares against the immutable original baseline: 12.236963272094727 rad/s², with a fixed ceiling of 14.684355926513671 rad/s². The correction has boundary, historical-regression, survival, speed-gate and ranking tests. Existing leaderboard updates are idempotent.

All eight old checkpoints were audited. Only original Runs 01 and 06 pass the fixed-baseline guards. Run 06 improves on the last valid retained checkpoint, Run 01, and is the audited starting artifact. Its historical training parent was invalid Run 05; that provenance is explicitly preserved, not rewritten. No old valid checkpoint meets all speed targets. See audit_existing.json and audit_existing.csv. The old ledger and best manifest were corrected/invalidated, with the historical ledger preserved.

## New bounded loop

Each new candidate was warm-started from the retained checkpoint, with 100 PPO iterations, 512 environments and seed 42. The final checkpoint was selected before the existing 60-second fixed-command evaluation. Only one or two allowed parameters changed per candidate. No speed reward, physics, gains, effort limits, canonical assets or environment structure changed.

Accepted refinements adjust EMA responsiveness, the acceleration penalty, and the timing of the existing reference. Timing redistribution slows high front-left joint-demand phases while preserving total duration and ordered poses. The selected derived reference is run_05/reference.npz; the original motion asset is unchanged.

See leaderboard.csv for every decision, best_metrics.json for complete metrics and target checks, and experiment_audit.json for independent decision/provenance checks. Run directories contain source snapshots, parameter changes, hypotheses, train/eval commands, logs and checkpoints. The general training curriculum is unchanged from the audited starting checkpoint.

## Retained configuration

Checkpoint: run_05/policy.pt. EMA alpha 0.3; residual scale 0.3; action-rate penalty 0.06; joint-acceleration penalty 0.2; phase multiplier 1.4; reference timing redistribution strength 0.5. The phase multiplier is inherited unchanged from the audited start. Current runtime sources match this run's saved snapshots.

Relative to the audited start, acceleration drops about 6.1% and both front-left saturation percentages improve, but action-rate RMS rises about 18.6%. Relative to the invalid old Run 05, acceleration drops about 25.6%. These are separate comparisons; neither is used to relax the immutable original-baseline cap.

Final replay command (from repository root): `bash docs/walk_ref/refinement_runs/best_render/render_command.sh`. The best video is best_render/eval.mp4. No additional training experiments were launched after run_08.

Final video verification: 60.959 seconds at 1280×720, 24 fps; rendered metrics match the selected evaluation exactly. Sampled visual review preserves the reference-guided gait. All 217 protected fingerprints match. See completion_audit.json and best_render/visual_review.md.
