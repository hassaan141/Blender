# Python harness matched to the browser gate — 2026-09-24

Handoff steps 1–2 (env match + re-evaluation, **no training**). Python now reproduces the
normal browser probe: 43/45 per-case survival outcomes agree, fall times mostly within 0.1–0.4 s.

## Three gaps found and closed

| Gap | Evidence | Fix |
|---|---|---|
| Expressive joints: Python replayed the forward reference; browser runs `ExpressionController` (Neutral) | Dominant: both policies drop to 1–2/9 under Neutral | `mujoco_env.neutral_expression()` ports `expression.js`; bit-exact vs JS (0.0 max diff, 1000 steps). Default `expression='neutral'`; `'reference'` keeps the old adapter |
| Command start: Python commanded from step 0 | Probe comparison in handoff | `evaluate.py --start stand_then_command` (default): 24 steps at `[0,0]`, then command, 192 measured steps |
| MuJoCo engine 3.14.0 vs browser 3.11.0 | Open-loop replay of browser ctrl trace: 3.14 differs 3.5e-4 after **one** control step and falls where browser stands; 3.11 matches to 1e-16, 8.8e-5 after 9 s | Run Python with isolated MuJoCo 3.11.0: `PYTHONPATH=$PWD/scratch/mujoco_3_11_0` (shared isaaclab env untouched) |
| Observation root position: browser uses `xpos('origin')` (one substep stale after `mj_step`), Python used `qpos[:3]` | Obs terms 42 and feet (55–66) differed 2e-4 / 1e-3 as soon as walking began, while qpos matched 1e-7 | `_proprio` reads `xpos['origin']`. After fix every obs segment matches ≤3e-6, torch vs ONNX action ≤6e-7 |

Feedforward (`feedforwardAtTime` vs `_ff`) was already equal to float32 precision (≤6e-7); not a gap.

## Re-evaluation (MuJoCo 3.11 + all fixes vs browser probe JSON)

Case order: stand, fwd, back, left, right, fwd-left, fwd-right, back-left, back-right.

| Policy / condition | Python | Browser |
|---|---|---|
| BASELINE_1, Neutral, stand→cmd (**real gate**) | `1........` | `11.......` |
| Attempt 03, Neutral, stand→cmd (**real gate**) | `1........` | `1........` |
| BASELINE_1, reference expr, stand→cmd | `111111111` | `111111.11` |
| Attempt 03, reference expr, stand→cmd | `11.11.11.` | `11.11.11.` |
| Attempt 03, reference expr, cmd first | `11111111.` | `11111111.` |

The two disagreements are knife-edge rollouts: in closed loop, 5e-7 float32/ONNX noise grows
to 2e-2 by step 96 and flips the outcome. **A single deterministic rollout per command is a
noisy gate**; any new candidate should also survive small perturbations (several seeds/pushes).

Results: `tools/experiment_loop/results/sim2sim_matched_mj311_rootfix_20260924/` (includes the
diagnostic scripts). Superseded intermediate runs: `sim2sim_matched_20260924/` (3.14, old root),
`sim2sim_matched_mj311_20260924/` (3.11, old root).

## What this means

Under true browser conditions both policies fail 8/9 walking commands; the leg policy is not
robust to the Neutral expressive-joint motion. Python can now train against that faithfully.
`train_ppo.py` defaults to `--expression neutral` and must be run with the 3.11 `PYTHONPATH`;
it still starts episodes with the command already set (no stand→command transition in training).

## Tooling changes

- `bingo-simulator/app/tools/probe_sim2sim_browser.mjs`: `PROBE_TRACE=<case>` records per-step
  ctrl/qpos/qvel/obs/action into the result JSON (stdout omits it). App source untouched.
- Browser traces: `scratch/sim2sim_browser_probe/baseline_browser_trace_forward.json`.

## Attempt 04 (browser-matched PPO, 60 it) — REJECT, and the real root cause

Attempt 04 = attempt 03 hyperparameters in the matched env (+ stand U(0.5,1.5) s → command).
Job exit 1 was the video renderer: `evaluate.py` re-created an EGL context per case (fails on
this machine) and an indentation bug from the `--init-noise` edit. Both fixed (one renderer,
`MUJOCO_GL=egl`). Python gate `1...1....` = browser probe `1...1....`, fall times within ~0.2 s
except fwd-right (3.62 vs 4.50 s). Training `mean_vx ≈ -0.01`: it never learned to walk.
(A separate manual browser check reported 1/9 with different fall times — different protocol.)

**Root cause of the browser failure: observation normalization, not dynamics.** BASELINE_1 was
trained with the 9 expressive joints static. Its frozen normalizer std for them is tiny (ear pos
0.0003–0.001, tail 0.0015, head pitch 0.005). The browser's Neutral controller holds ears at
±0.2 rad and oscillates the tail, so those inputs sit at 40–600σ and clip at ±5 — the leg policy
sees saturated garbage. Ears are 2–18 g with no contacts; physically they barely matter
(baseline 7/9 with static reference expression). Fine-tuning with the normalizer frozen could
not fix this.

Hold the 9 expressive pos (obs 12–20) + vel (33–41) inputs at the normalizer mean, physics
unchanged (real Neutral expression):

| BASELINE_1 (unmodified weights) | Result |
|---|---|
| Real browser probe (`PROBE_EXPR_OBS`) | `11111111.` **8/9** (back-left falls 7.21 s) vs `11.......` 2/9 unmasked |
| Python matched, 5 perturbed seeds | 43/45 |

Results: `tools/experiment_loop/results/sim2sim_expr_obs_mask_20260924/`. Probe option
`PROBE_EXPR_OBS='[18 values]'` patches `rt.buildObs` in the page only.

## Applied in the browser runtime (user-approved, 2026-09-24)

`constants.js` `EXPR_OBS_TRAINING_MEAN` + `sim.js buildObs()` now feed those 18 inputs at
training values; physics and visible expression unchanged; policy/ONNX/MJCF unchanged
(ONNX `7a747b…`, MJCF `85665…`). `mujoco_env.py` reads the same constant (`expr_obs='training_mean'`
default, `'live'` for the old behaviour).

| Normal gate, installed BASELINE_1 | Result |
|---|---|
| Real browser probe, unpatched app | `11111111.` 8/9 (back-left falls 7.21 s) |
| Python matched gate | `11111111.` identical |

Videos + 3×3 montage: `tools/experiment_loop/results/sim2sim_runtime_exprobs_20260924/`.
Open: backward-left (a pre-existing baseline weakness) and live-app check with the frame clock.
