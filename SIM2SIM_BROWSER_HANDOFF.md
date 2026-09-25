# Bingo sim-to-sim handoff — 2026-09-24

For low-usage job execution in either agent, see [LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md](LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md).

## Update 2026-09-24 (Claude): handoff steps 1–2 done, no training run

Python harness now reproduces the browser gate (43/45 case outcomes). Four fixes: Neutral expression port (bit-exact), stand→command eval start, MuJoCo 3.11.0 via `PYTHONPATH=$PWD/scratch/mujoco_3_11_0`, and root position from `xpos` like the browser. Details and next-step options: [python_browser_match.md](docs/experiment_loop/sim2sim/python_browser_match.md). Attempt 04 (matched PPO) REJECTED. Root cause found: frozen normalizer saturates on the 9 expressive-joint inputs; holding them at training mean gives the unmodified baseline 8/9 in the real browser. Applied in the runtime (user-approved): installed baseline now 8/9 in the normal browser gate; backward-left still falls.

## Read this first

The MuJoCo-trained attempt 03 **passed 9/9 commands in the Python evaluator but failed the real browser MuJoCo/WASM survival gate**. In the normal browser runtime it survived only stand (1/9). **Do not deploy it or replace the installed policy.** The [browser probe report](docs/experiment_loop/sim2sim/browser_probe.md) and [decision](docs/experiment_loop/sim2sim/attempt_03/decision.md) now say REJECT for browser deployment. The previous claim that attempt 03 "worked in MuJoCo" applied only to the Python evaluation harness.

The user wants one command-conditioned policy for stand, forward, backward, left, right, and the four diagonal turns. The target contract is raw 95D observation → 12 leg residual actions, with `[vx,yaw]` commands, 0.3 rad residual scale, 0.3 residual EMA, 0.1 command smoothing, frozen reference banks, and a 120 Hz MuJoCo physics / 24 Hz policy clock. Preserve the expressive Stage 1–5 work, existing reference banks, morphology, physics assets, actuator gains and limits, and all historical results. Avoid AMP and broad searches. The user has limited model usage: use targeted reads and bounded jobs, and do not poll long-running logs.

## Exact assets and code

- Frozen original policy and runtime: `BASELINE_1/policy.pt`, `BASELINE_1/runtime.py`, `BASELINE_1/references/`. Original checkpoint SHA-256: `35955f919f41e128fe40a1148c03b5a0d3e64a7dca4e99132feda544c0859cb9` (unchanged after these experiments).
- Browser MJCF: `bingo-simulator/app/public/robot/bingo_scene.xml`. SHA-256 `85665ddd371d59c543eaa6dd97aeb2f1e5da5b96c4c78f3a5f9b29f6abbebf3f`.
- Browser runtime: `bingo-simulator/app/src/game/runtime/sim.js`, `command_runtime.js`; loader `src/game/policies/loader.js`; physics `src/game/physics/mujoco.js`; expressive targets `src/game/controllers/expression.js`.
- Python adaptation harness: `tools/experiment_loop/sim2sim/mujoco_env.py`, `train_ppo.py`, `evaluate.py`, `compare_results.py`. `mujoco_env.py` loads the **same browser MJCF path**, by name; it does not define a second MJCF.
- Attempt 03 checkpoint: `tools/experiment_loop/results/sim2sim_attempt_03_train_20260923/candidate/policy.pt`. Nine Python metrics/videos: `tools/experiment_loop/results/sim2sim_attempt_03_eval_20260923/evaluation/`. Python comparison montage: `docs/experiment_loop/sim2sim/attempt_03/baseline_vs_candidate_montage.mp4`.
- Installed browser ONNX remains `bingo-simulator/app/public/policies/policy.onnx` and is the original baseline. The attempt 03 ONNX exists **only for probing** at `scratch/sim2sim_browser_probe/candidate.onnx` (SHA-256 `65465c6a3cc2e71ab64f56ccd996653565e0cd6a722c5b1ba5506080db08b2de`). The probe serves it through an in-memory Playwright route; the policy name in probe JSON still reads `bingo_baseline_1` because the manifest is unchanged. Distinguish runs by the ONNX file passed to the probe, not that name.
- The current working tree has substantial pre-existing staged/untracked archive and simulator changes. Do **not** reset or clean the tree globally. `rl/bingo_rl/bingo_rl/natural_walk/` and `walk_ref/` are real source directories even if untracked. For Isaac imports, start `AppLauncher(headless=True)` before importing `isaaclab`/`pxr`; no Isaac run is needed for this browser issue.

## Experiments completed

All three bounded PPO attempts started from the unchanged `BASELINE_1/policy.pt` and used the browser MJCF in Python MuJoCo 3.14.0. [Leaderboard](docs/experiment_loop/sim2sim/leaderboard.csv), [summary](docs/experiment_loop/sim2sim/summary.md):

| Policy | Python MuJoCo 9 s survival | Python normalized tracking error | Decision |
|---|---:|---:|---|
| BASELINE_1 | 8/9 | 0.3472 | Original baseline |
| Attempt 01 | 6/9 | 0.4282 | Reject |
| Attempt 02 | 8/9 | 0.3602 | Reject |
| Attempt 03 | 9/9 | 0.3272 | Retain for research; reject browser deployment |

Attempt 03 changed command coverage during training: healthy episodes end after 8 s, and backward-left is sampled 3×. Training used 60 PPO iterations, 12 environments, horizon 128, learning rate 3e-5 and baseline-action penalty 0.5. Its Python moving-command averages: joint acceleration RMS 11.32 rad/s², action-rate RMS 1.449, torque saturation 5.52%, slip 0.157 m/s. Action-rate RMS is *worse* than the original baseline's 1.369. Do not treat the Python aggregate score as a visual or browser pass.

Exact attempt 03 training/eval commands are saved in the two `command.txt` files under the attempt 03 train/eval result dirs. The run used `tools/experiment_loop/run_and_wake.sh` with a finite timeout and one `codex queue` completion notice per job. There is no running job now.

## Browser probe: decisive evidence

Script: `bingo-simulator/app/tools/probe_sim2sim_browser.mjs`. It imports the **actual JS `BingoRuntime`** in headless Chromium, uses browser MuJoCo WASM and onnxruntime-web, calls `controlStep()` directly, and runs nine cases: stand, W, S, A, D, W+A, W+D, S+A, S+D. It allows 24 settle steps and then measures 192 control steps (8 s). Each control step advances exactly five MuJoCo physics steps. It does not use a rendering frame clock. The probe fetches the MJCF served to the browser and hashes it; the result matched the local Python MJCF byte-for-byte (`85665…f3f`). The installed browser MuJoCo package is 3.11.0; Python uses 3.14.0.

| Probe conditions | Baseline survival | Attempt 03 survival |
|---|---:|---:|
| Normal browser Neutral expression, stand 1 s then command | 2/9 | **1/9** |
| Replace expressive targets with training reference; stand then command | 8/9 | 6/9 |
| Training-reference expressive targets; command before 1 s settle | Not run | 8/9 (backward-right fails at 6.375 s) |

These controlled runs identify **two concrete gaps**. The Python training adapter commands the nine expressive joints from the forward reference at the next phase, while the real browser drives them with the independent Neutral expression controller. The Python evaluator also sets the command from the first step and calls the first second "settle"; the realistic browser test stands first and then transitions into the command. Both change the state seen by the leg policy. Matching them in the probe improves survival greatly but does not close the last failure. MuJoCo version differences are a possible remaining cause, not proven. Wall-clock jitter is **not** the cause of these observed falls: direct fixed-step calls still fall. Do not add jitter training without a measured timing effect.

Raw JSON: `scratch/sim2sim_browser_probe/{baseline_browser,browser,baseline_expr,candidate_expr,candidate_pythonmatched}_results.json`. The Playwright browser binaries are in `scratch/sim2sim_browser_probe/browsers/`; the snap Chromium was unusable because of home-directory and glibc errors. The script routes `public/vendor` requests from `node_modules` in memory because those generated vendor files are absent in this checkout. It does not modify browser source, MJCF, installed policy, or manifest.

To reproduce the **normal attempt 03 browser gate**, from repo root:

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/scratch/sim2sim_browser_probe/browsers" \
  timeout 240 /pub0/muhammadf/.nvm/versions/node/v22.23.1/bin/node \
  bingo-simulator/app/tools/probe_sim2sim_browser.mjs \
  "$PWD/scratch/sim2sim_browser_probe/candidate.onnx" \
  "$PWD/scratch/sim2sim_browser_probe/browser_results.json"
```

For diagnostic-only conditions, prefix `PROBE_EXPRESSION=python`; prefix both `PROBE_EXPRESSION=python PROBE_START=python` to match the Python evaluator's expression and command start. These flags change only the probe, not the application.

## Recommended next work

1. Correct the Python training/evaluation adapter to use the **browser's actual Neutral expressive-joint target generator** and to evaluate a true 1 s stand → command transition. Keep an option for reference-expression experiments, but make browser conditions the acceptance gate. Verify the JS/Python target vectors at a few timestamps before another PPO run. This is a targeted environment-match fix; do not change morphology, MJCF physics, Kp/Kd, effort limits, observation/action order, or references.
2. Re-evaluate both the frozen baseline and existing attempt 03 under these matched Python conditions before training. If they still disagree with the browser, isolate the remaining gap, including the 3.14.0 vs 3.11.0 engine versions, using a few deterministic joint-target traces. Do not infer from the matching MJCF bytes that solver implementations are identical.
3. If a further training attempt is warranted, use one bounded adaptation from the original baseline or the best valid checkpoint, then run the **normal browser gate on all nine commands**. Keep only if stand and all eight movements survive and the video looks acceptable. Never promote based solely on the Python metrics. An ONNX can be exported to scratch for probing without overwriting the installed browser file.

The user wants Claude to continue from this point, without repeating the earlier Isaac/ONNX/95-observation audits. Those had already found the 95D observation and ONNX inference contract correct to roughly 1e-6. The present issue is a mismatch in training/evaluation conditions and possibly engine version, not a discovered observation-order bug.
