# MuJoCo sim-to-sim adaptation

Three bounded PPO adaptations were trained from the unchanged `BASELINE_1/policy.pt` and evaluated for nine seconds on each of the nine keyboard command cases in Python MuJoCo. Attempt 03 survived all nine in that evaluator, versus 8/9 for the baseline, and reduced aggregate normalized command-tracking error from 0.3472 to 0.3272. A subsequent probe in the actual browser MuJoCo/WASM runtime **rejected attempt 03 for deployment**: only stand survived with the normal browser expressive-joint animation and stand-to-command transition. See the [browser probe](browser_probe.md).

Across the eight moving commands, attempt 03 averages 11.32 rad/s² joint-acceleration RMS, 1.449 action-rate RMS, 5.52% torque saturation, 0.157 m/s slip, and 0.286 yaw-rate error. Versus baseline, acceleration, slip, and yaw error improve slightly; action-rate RMS increases from 1.369 to 1.449 and torque saturation is essentially unchanged. The videos show the same broad stepping pattern, but use the MuJoCo collision-mesh renderer; they are useful for gait comparison, not final appearance approval.

This is a **research checkpoint, not an accepted browser policy**. An ONNX was exported only to scratch for the browser probe and served through a temporary Playwright route; the installed browser policy was not changed. The training/evaluation harness used the existing 95-observation, 12-residual interface; baseline hash after the run is `35955f919f41e128fe40a1148c03b5a0d3e64a7dca4e99132feda544c0859cb9`.

| Attempt | Survived | Tracking error | Decision | Targeted change |
|---|---:|---:|---|---|
| Baseline | 8/9 | 0.3472 | Reference | Original BASELINE_1 policy |
| 01 | 6/9 | 0.4282 | Reject | Initial bounded residual PPO |
| 02 | 8/9 | 0.3602 | Reject | Stronger baseline-action regularization |
| 03 | 9/9 Python; 1/9 browser | 0.3272 | Reject deployment | 8 s command rotation and 3× backward-left sampling |

Per-command metrics are in [attempt 03 comparison](attempt_03/comparison.csv). The all-command video and baseline/candidate montage are [here](attempt_03/all_nine_commands.mp4) and [here](attempt_03/baseline_vs_candidate_montage.mp4).

## Attempt 03 artifacts

Checkpoint: `tools/experiment_loop/results/sim2sim_attempt_03_train_20260923/candidate/policy.pt`

Evaluation directory: `tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/`

| Command | Video |
|---|---|
| Stand | [stand.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/stand.mp4) |
| Forward | [forward.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/forward.mp4) |
| Backward | [backward.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/backward.mp4) |
| Left | [left.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/left.mp4) |
| Right | [right.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/right.mp4) |
| Forward-left | [forward_left.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/forward_left.mp4) |
| Forward-right | [forward_right.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/forward_right.mp4) |
| Backward-left | [backward_left.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/backward_left.mp4) |
| Backward-right | [backward_right.mp4](../../../tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/backward_right.mp4) |

All three attempts started from the baseline, were bounded, and completed via the one-shot run-and-wake workflow. No training loop remains running.
