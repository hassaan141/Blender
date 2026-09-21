# Gait refinement protocol

The starting artifact is original Run 06, selected by auditing all eight original checkpoints against the corrected fixed-baseline protocol. Its historical parent was invalid Run 05; that provenance is preserved. Run 06 itself passes the fixed guards and improves on the last valid incumbent, Run 01. No existing valid checkpoint meets every speed target.

The immutable acceleration baseline remains ORIGINAL baseline/metrics.json: 12.236963272094727 rad/s². Reject any candidate above 14.684355926513671 rad/s², irrespective of its parent. Never reset this baseline to Run 06 or a new best.

Mandatory gates: 100% survival, max vx <0.40 m/s, original-baseline acceleration ceiling, reference-error RMS <=125% of original baseline. Reject a higher-mean candidate if either velocity std or acceleration increases: faster but shakier is worse.

Selection: first minimize distance to mean speed band 0.20–0.30 m/s, without rewarding higher speed inside that band. Until vx std is below 0.06, prefer lower std. Once speed gates pass, prefer lower acceleration, then lower action-rate RMS, then lower front-left SP+knee saturation and lower mean saturation. Mean/std gate failures are recorded as incomplete targets, not called full success. Quantitative reference guard applies throughout; inspect final rendered gait against the existing reference-guided video.

At most eight new experiments, named run_01 through run_08 here, always warm-starting from the retained checkpoint and configuration. Same 100 PPO iterations/2400 steps, 512 environments, seed 42, fixed final checkpoint, and existing 60-second evaluator at 0.25 m/s after one second settling. The training curriculum is inherited unchanged from Run 06.

Only EMA alpha, action-rate penalty, joint-acceleration penalty, residual scale, phase-rate mapping, and reference timing may change. At most two related parameters per candidate. No environment redesign, speed reward, reference tracking reward weight, physics gains/limits, or canonical asset changes. Revert rejected changes. Save hypotheses, deltas, commands, logs, sources, checkpoint, metrics and decisions. Update leaderboard.csv after each evaluation. At the limit, restore the best, render, verify metrics and protected files, and stop even if targets remain unmet.
