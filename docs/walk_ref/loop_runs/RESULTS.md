# Walking optimization: eight-experiment result

Stopped at the requested limit of eight experiments. **Run 05 is retained. The full metric target was not reached.**

The original baseline was re-evaluated with the existing `eval_walk_loop.py` and reproduced exactly. All eight candidates were warm-started from the retained best, trained for 100 PPO iterations (2400 steps), with 512 environments and seed 42. Each changed one or two related parameters. Every candidate was evaluated for 60 seconds after one second settling, using a single robot and fixed 0.25 m/s command. The predeclared selection criteria are in [PROTOCOL.md](PROTOCOL.md); all decisions are in [leaderboard.csv](leaderboard.csv).

| Metric | Original baseline | Retained Run 05 | Target / outcome |
|---|---:|---:|---|
| Survival | 60 s | 60 s | Pass |
| Mean forward velocity | 0.108842 | 0.201675 m/s | Pass: 0.20–0.30 |
| Velocity standard deviation | 0.110747 | 0.069559 m/s | **Unmet:** <0.06 |
| Maximum forward velocity | 0.504771 | 0.392333 m/s | Pass: <0.40 |
| Joint acceleration RMS | 12.236963 | 16.859755 rad/s² | **Unmet:** increased 37.8% |
| Residual action-rate RMS | 0.020062 | 0.016498 rad/step | Improved 17.8% |
| Mean joint torque saturation | 3.946759% | 8.923611% | **Worse:** higher-priority speed gains have a torque cost |
| Reference tracking error RMS | 0.077160 | 0.028893 rad | Improved; final visual review recorded separately |

Velocity RMSE about 0.25 m/s dropped from 0.179417 to 0.084698 m/s. Selection prioritizes survival and speed; it is not a claim that every shaking metric improved. The smoother alternatives (Runs 06 and 08) lost too much forward speed under the recorded selection rule. Runs 04 and 07 improved speed RMSE but exceeded the peak-speed guard.

## Retained configuration and reproduction

- Checkpoint: [run_05/policy.pt](run_05/policy.pt).
- Runtime: phase rate = `1.4 * commanded_vx / 0.660`; residual EMA alpha = `0.2`; residual scale remains `0.3`.
- Training curriculum: `env.vx_range=[0.2,0.3] env.stand_prob=0.0`. These are training-command overrides, not changes to the general Stage C defaults.
- All other reward weights, PD gains, effort limits, and reference assets remain at baseline settings. Rejected changes have been reverted.
- The two current walk-ref source files exactly match the saved Run 05 snapshots. Each run includes its source snapshots, hypothesis, parent checkpoint, parameter changes, train/eval commands, logs, checkpoint, metrics, report, and decision.
- From the repository root, replay the final rendered evaluation with `bash docs/walk_ref/loop_runs/best_render/render_command.sh`. To replay a different run, first restore its two saved source snapshots, then use its saved evaluation command.
- The existing baseline evaluator and canonical assets were not modified. Existing ONNX exports were not regenerated; this deliverable is the retained training checkpoint and its matching runtime configuration.

## Final verification

See [experiment_audit.json](experiment_audit.json) for all eight experiment checks and [completion_audit.json](completion_audit.json) for final render, source, and protected-file verification. The final video is [best_render/eval.mp4](best_render/eval.mp4), with metrics and report alongside it. These results describe this fixed-command evaluation; no additional commands or seeds were evaluated.

The final rendered metrics exactly equal the selected Run 05 evaluation. The video is 1280×720 at 24 fps, covering more than 60 seconds. All 217 fingerprinted protected files are unchanged, and the baseline evaluator is byte-identical to its committed version. Sampled visual comparisons retain the reference gait; see [visual_review.md](best_render/visual_review.md). The bounded loop is complete, while the unmet numeric targets remain explicitly unmet.
