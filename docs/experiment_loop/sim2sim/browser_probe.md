# Browser MuJoCo survival probe

The [probe script](../../../bingo-simulator/app/tools/probe_sim2sim_browser.mjs)
loads the real browser MuJoCo/WASM runtime in headless Chromium. It serves the
attempt 03 ONNX from scratch through an in-memory route, so neither the installed
browser policy nor the MJCF is changed. Each case runs 24 control steps of settle
and 192 measured control steps (8 seconds), with five fixed MuJoCo physics steps
per control step.

The browser fetched MJCF SHA-256 is
`85665ddd371d59c543eaa6dd97aeb2f1e5da5b96c4c78f3a5f9b29f6abbebf3f`,
identical to the file loaded by Python at
`bingo-simulator/app/public/robot/bingo_scene.xml`. Both use a 1/120 s physics
step and five steps per policy action. The probe drives `controlStep()` directly,
without wall-clock scheduling; falls still occur, so scheduling jitter is not the
cause of this result.

| Policy and conditions | Survived / 9 | Failed commands |
|---|---:|---|
| Installed baseline, normal browser | 2/9 | Backward, left, right, both forward turns, both backward turns |
| Attempt 03, normal browser | 1/9 | All moving commands |
| Installed baseline, training-reference expressive joints | 8/9 | Forward-right |
| Attempt 03, training-reference expressive joints | 6/9 | Backward, forward-left, backward-right |
| Attempt 03, training-reference expressive joints and command before settle | 8/9 | Backward-right |

The training adapter replays expressive-joint targets from the forward reference.
The browser normally drives those nine joints with its independent Neutral
animation. This controlled switch explains much of the survival difference. The
Python evaluator also starts each command immediately and counts its first second
as "settle"; the browser test normally stands for one second and then starts the
command. The remaining backward-right failure under Python-matched conditions
shows that these two differences do not explain everything. Python uses MuJoCo
3.14.0; the browser package specifies MuJoCo 3.11.0, which remains a possible
source of the last difference and was not isolated here.

Raw results: [baseline browser](../../../scratch/sim2sim_browser_probe/baseline_browser_results.json),
[attempt 03 browser](../../../scratch/sim2sim_browser_probe/browser_results.json),
[baseline with reference expression](../../../scratch/sim2sim_browser_probe/baseline_expr_results.json),
[attempt 03 with reference expression](../../../scratch/sim2sim_browser_probe/candidate_expr_results.json),
[attempt 03 with Python-matched start](../../../scratch/sim2sim_browser_probe/candidate_pythonmatched_results.json).

Deployment decision: **REJECT attempt 03**. The next training adaptation should
use the browser's expressive-joint targets and evaluate stand-to-command
transitions. Add timing randomization only if a measured timing effect appears.
