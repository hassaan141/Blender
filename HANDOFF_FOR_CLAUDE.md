# Bingo handoff for Claude

Prepared 2026-09-16 (America/Toronto), after the locomotion refinement, gait-quality campaign, and repository cleanup. This file summarizes the completed work and points to authoritative records; it does not authorize further training or changes.

## Start here and preserve these rules

Repository: `/pub0/muhammadf/Blender` on the existing machine. Current work is a slow reference-guided walk, alongside the authoritative Stage 1–5 animation pipeline. The user values a softer, more symmetric, elegant gait with the same cadence. **Faster is not better.** Do not redesign the environment or optimize forward speed.

The canonical locomotion baseline is **quality campaign Run 05**, at:

```text
/pub0/muhammadf/Blender/docs/walk_ref/quality_runs/run_05/
```

Treat that entire directory as read-only. Do not modify, retrain in place, overwrite, rename internally, or change any files inside it. Also preserve the final video directory `docs/walk_ref/quality_runs/best_render/`, the winning training logs, and runtime reference. Protection is an instruction, not an OS permission lock. New evaluations must write to a fresh directory. Never invoke historical render/train commands verbatim without checking their output destinations.

Do not change actuator physics, `rl/bingo_rl/bingo_rl/bingo_v4.py`, Kp/Kd, effort limits, or canonical Stage 1–5 assets. Cleanup was documentation/artifact organization only: source and protected assets were unchanged. No additional experiments are currently requested; wait for the user's next objective.

Read [CURRENT_BEST](docs/walk_ref/CURRENT_BEST.md), [walk-ref README](docs/walk_ref/README.md), and the [current quality protocol](docs/walk_ref/quality_runs/PROTOCOL.md). Older status notes, particularly the command-conditioned locomotion branch, are historical and do not supersede these.

## What happened: three campaigns, 22 experiments

1. **Original loop: 8 experiments.** A selection bug retained original Run 05 despite joint acceleration rising approximately 37.8% (12.24 → 16.86 rad/s²). The protocol required rejection above a 20% regression. Selection was corrected to compare against the fixed original baseline, 12.236963272094727 rad/s², not a moving parent. The cap was 14.684355926513672 rad/s². All eight runs were audited; original Run 06 was the best protocol-valid starting point for refinement. See [audit](docs/walk_ref/refinement_runs/audit_existing.json) and [fixed selection rules](docs/walk_ref/loop_runs/selection_rules.py).
2. **Refinement: 8 experiments.** Focused smoothing, action-rate/acceleration penalties, residual authority, phase mapping and reference timing established refinement Run 05 as the retained slow gait. This is the approximately 12.54 rad/s² baseline mentioned in earlier user messages. Its derived `reference.npz` remains a live dependency.
3. **Gait quality: 6 experiments.** The user accepted the speed behavior and asked for less FL SP/knee saturation, lower acceleration and bobbing, preserving cadence. Diagnosis preceded reward changes. Accepted ancestry was refinement Run 05 → quality Run 02 → quality Run 04 → **quality Run 05**. Quality Run 06 was rejected; the winner was restored and rendered.

**Run names are campaign-scoped.** Original Run 05 is the invalid historical selection; refinement Run 05 is the earlier baseline/reference dependency; quality Run 05 is the latest canonical winner. Never choose a checkpoint merely because it is called Run 05 or has the newest modification time.

The [single active leaderboard](docs/walk_ref/experiments/leaderboard.csv) covers all campaigns. Its KEEP/REJECT decisions use each campaign's own protocol, not one retrospective ranking. The [concise history](docs/walk_ref/experiments/summary.md) and archived original ledgers retain the decision trail.

## Diagnosis before the quality experiments

The [full diagnosis](docs/walk_ref/quality_runs/DIAGNOSIS.md) and raw baseline trace document these findings:

- The reference is asymmetric: FL/FR SP excursion 1.081/0.773 rad; knee excursion 0.799/0.438 rad. Phase/sign alignment does not eliminate the difference. Wholesale mirroring would change the desired gait.
- FL knee saturation occurred in low-foot support phases, while FL SP saturation was also prominent during swing. Contact was inferred from a shank-tip height threshold, not a force sensor. Threshold-sensitive contact timing cannot establish exact load sharing or touchdown impacts.
- FL knee residual RMS was smaller than FR, but tracking of the residual-adjusted target was worse. Shrinking residuals indiscriminately could remove useful compensation.
- Corresponding actuator settings were identical: SP/knee Kp 120, Kd 1.60, effort ceiling 3 Nm. No left/right gains mismatch explained the difference.
- SP torque often opposed joint motion, consistent with braking/trajectory demand. Knee evidence was more consistent with support-phase tracking/load mismatch. These are supported hypotheses, not a uniquely established causal explanation.
- Body height uses link-origin position, while vertical velocity uses COM velocity; these are complementary measures, not identical-point derivatives.

## Quality campaign decisions

| Run | Decision | Mean vx | Acceleration rad/s² | Bobbing mm | Main outcome |
|---|---|---:|---:|---:|---|
| Baseline | Retained input | 0.198253 | 12.536512 | 2.359893 | Refinement Run 05 remeasured |
| 01 | REJECT | 0.189823 | 11.847548 | 2.607886 | Mean speed below gate; bobbing worse |
| 02 | KEEP | 0.195186 | 12.177318 | 1.812811 | FL torque and height costs improved quality |
| 03 | REJECT | 0.186553 | 11.652144 | 1.599430 | Mean speed below gate |
| 04 | KEEP | 0.190945 | 11.607508 | 1.560142 | Stronger FL torque cost improved quality |
| 05 | **CURRENT BEST** | 0.192935 | 11.000194 | 1.509768 | Small vertical-velocity cost improved quality |
| 06 | REJECT | 0.191554 | 11.078765 | 1.626604 | Quality score did not improve |

All six candidates survived the measured evaluation. Exact parameter changes and rejection reasons are in archived manifests and the leaderboard.

The quality protocol required 100% survival, no warmup falls, mean vx 0.19–0.23 m/s, vx std <0.06, max vx <0.40, unchanged cadence within one 24 Hz sample, and bounded reference tracking error. Acceleration had to be below the fixed refinement baseline; saturation and bobbing measures could not exceed that baseline. Among passing candidates the score weighted normalized mean FL SP/knee saturation 50%, acceleration 25%, bobbing 25%. Action-rate RMS was reported separately. This superseded the earlier speed-refinement ranking; do not silently substitute a different protocol.

## Latest baseline artifacts and metrics

- [Checkpoint](docs/walk_ref/quality_runs/run_05/policy.pt)
- [Authoritative metrics](docs/walk_ref/quality_runs/run_05/metrics.json)
- [Physical-quality metrics](docs/walk_ref/quality_runs/run_05/quality_metrics.json)
- [Final evaluation video](docs/walk_ref/quality_runs/best_render/eval.mp4)
- [Visual review](docs/walk_ref/quality_runs/best_render/visual_review.md)
- [Environment snapshot](docs/walk_ref/quality_runs/run_05/bingo_walk_ref_env.py) and [config snapshot](docs/walk_ref/quality_runs/run_05/bingo_walk_ref_env_cfg.py)
- [Training provenance](docs/walk_ref/quality_runs/run_05/experiment.json)

| Metric | Latest baseline |
|---|---:|
| Survival | 100% |
| Mean vx | 0.1929352582 m/s |
| vx std | 0.0526900031 m/s |
| Max vx | 0.3658355176 m/s |
| Joint acceleration RMS | 11.0001935959 rad/s² |
| Filtered residual action-rate RMS | 0.0181303322 |
| FL SP saturation | 10.4166667% |
| FL knee saturation | 15.9722222% |
| Detrended body-height RMS | 1.5097682 mm |
| Vertical COM velocity RMS | 0.06233259 m/s |
| Phase-cycle period | 2.12191358 s |
| Reference tracking RMS | 0.0300148726 rad |

Relative to the refinement baseline, acceleration decreased approximately 12.25%, bobbing 36%, FL SP saturation 38.8%, and FL knee saturation 24.6%. **Action-rate RMS increased** from 0.01737397 to 0.01813033; do not claim every metric improved. Survival refers to the recorded deterministic evaluation, not a guarantee across arbitrary conditions.

## Runtime, architecture, and reproducibility

The policy supplies 12 leg residuals on top of a cyclic motion reference. An EMA filters policy actions before residual scaling; expressive channels follow the reference. The policy/reference timing must be preserved together.

Live source:

```text
rl/bingo_rl/bingo_rl/walk_ref/bingo_walk_ref_env.py
rl/bingo_rl/bingo_rl/walk_ref/bingo_walk_ref_env_cfg.py
rl/tools/eval_walk_loop.py
rl/tools/walk_quality_diagnostics.py
```

Runtime settings: 120 Hz physics, decimation 5, 24 Hz control; EMA alpha 0.3; residual scale 0.3 rad; phase multiplier 1.4. Reward weights are action rate 0.06, joint acceleration 0.2, FL torque 0.03, body height 0.05, vertical velocity 0.01; body-height target 0.17165698111057281 m. Saved configuration and source are authoritative for the full definitions and inherited settings.

**Required historical dependency:** `docs/walk_ref/refinement_runs/run_05/reference.npz`. That path resolves through an archive compatibility link and is hardcoded in the unchanged live config. Do not remove it. Quality Run 04's checkpoint is also retained for training ancestry.

Winning training used seed 42, 512 environments, 100 PPO iterations / 2400 steps, task `Bingo-WalkRef-v4-C-v0`, `env.vx_range=[0.2,0.3]`, and `env.stand_prob=0.0`. The exact command is retained inside the canonical run as provenance; this handoff does not request rerunning it.

Winning training records remain unchanged at:

```text
logs/skrl/walk_quality/run_05/2026-09-16_20-21-46_ppo_torch/
  params/env.yaml
  params/agent.yaml
  checkpoints/agent_2400.pt
outputs/2026-09-16/20-21-46/
```

Safe evaluation, from any shell on this machine:

```sh
cd /pub0/muhammadf/Blender
bash docs/walk_ref/best/reproduce.sh
```

The wrapper renders video and diagnostics into a fresh `outputs/walk_ref_reproduction/` directory. An optional single argument selects a new output directory; existing destinations and protected locations are rejected. It uses GPU 1 and `/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python`, with the original Kit flags `--/rtx/verifyDriverVersion/enabled=false --no-window`.

Evaluation is `Bingo-WalkRef-v4-Play-v0`, one environment, held vx command 0.25 m/s, vy/yaw 0, one second settling plus 60 seconds measurement (1440 control samples). Always pass the latest checkpoint explicitly if bypassing the wrapper: the evaluator's unchanged default checkpoint/output still refer to historical work. **Do not run `best_render/render_command.sh` verbatim:** it targets the canonical video directory.

Recorded package versions:

```json
{
  "python": "3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0]",
  "executable": "/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python",
  "packages": {
    "torch": "2.7.0+cu128",
    "numpy": "1.26.0",
    "gymnasium": "1.2.1",
    "skrl": "2.1.0"
  },
  "CUDA_VISIBLE_DEVICES": "1",
  "note": "Recorded during cleanup replay; exact config snapshots are in canonical run_05."
}
```

## Cleanup completed

The user requested direct cleanup without changing source or the latest baseline. We audited `docs/`, `logs/`, `exports/`, and experiment outputs, then:

- Created current documentation entrypoints, a canonical-baseline page, one combined leaderboard, a concise history, and the guarded reproduction wrapper.
- Kept all authoritative Stage 1–5 documents/assets, source, canonical run/video, winning training logs and launch settings unchanged.
- Moved older campaign records under `archive/walk_ref/` and older training/Hydra records under `archive/locomotion_training/`. `docs/walk_ref/archive` links to the archive. Original paths needed by source, provenance and historical tools resolve through compatibility links.
- Preserved original script directory depth in the archive because historical scripts derive repository paths from `__file__`. Do not casually move archives or replace these links.
- Retained historical rejected-run configurations/diagnostics as selection-audit and reproducibility evidence, along with final/best historical checkpoints. Removed **189 obsolete intermediate checkpoints** and generated caches, freeing approximately **876 MiB**. Saved references were checked before removing intermediates.
- Found no byte-identical historical videos. Unique comparison videos were archived rather than discarded. `best/metrics.json`, `best/quality_metrics.json`, and `best/eval.mp4` link to originals; they are not duplicate copies.
- Kept unique reaction/expression comparison and room/asset preview outputs as pipeline handoff material.
- Kept both historical ONNX exports and manifests, clearly labelled in [exports/README.md](exports/README.md). **Neither is an export of the current baseline.** No new export was produced.

The exact archive/deletion inventory is [cleanup_manifest.json](docs/walk_ref/archive/cleanup_manifest.json). This was a workspace cleanup, not a commit or push; review `git status` before making further changes. Many apparent deletions are moves plus new archive files and compatibility links. Preserve that existing work.

Current organization:

```text
docs/
  README.md
  BingoMocapPipelineSpec.md
  Bingo_Robot_Spec_for_Animation.md
  Blender_Animation_to_Robot_Motion_Guide.pdf
  locomotion -> archived command-conditioned documentation
  walk_ref/
    README.md
    CURRENT_BEST.md
    best/
      metrics.json -> canonical metrics
      quality_metrics.json -> canonical diagnostics
      eval.mp4 -> canonical video
      checkpoint_reference.txt
      reproduce.sh
      runtime_versions.json
    experiments/
      leaderboard.csv
      summary.md
    quality_runs/
      run_05/       # original, unchanged
      best_render/  # original, unchanged
      ...historical compatibility links
    loop_runs -> archive/loop_runs
    refinement_runs -> archive/refinement_runs
    archive -> ../../archive/walk_ref
archive/
  walk_ref/               # campaign evidence and cleanup audits
  locomotion_training/    # older training and launch records
```

## Verification performed after cleanup

The [sanity report](docs/walk_ref/archive/sanity_check.json) records:

- Full 60-second baseline evaluation with video rendering completed successfully, exit code 0, in `/tmp/bingo-cleanup-baseline-eval`.
- Both `metrics.json` and `quality_metrics.json` matched the canonical baseline exactly as parsed JSON.
- IsaacLab, bingo_rl, Torch, Gymnasium, skrl and diagnostic imports succeeded through the real evaluation.
- All **363 protected file SHA-256 hashes** were unchanged, including canonical checkpoint/metrics/video, runtime reference, source and protected assets. Live environment/config matched the canonical snapshots.
- **16 selection tests passed** (8 quality, 8 refinement).
- Active documentation links and archive compatibility links resolved; canonical video passed ffprobe.
- The reproduction wrapper refused the canonical run as an output destination before launching evaluation.

Temporary replay output under `/tmp` is disposable and may disappear. It is not the authoritative baseline. The durable report and original artifacts are in the repository. These checks describe the cleanup-time state; recheck if subsequent changes occur.

Key artifact SHA-256 hashes:

```json
{
  "docs/walk_ref/quality_runs/run_05/policy.pt": "48b6f2ca979883b7bb1650359e8fa1ae1f1cc7fc02dd5329c5b6eee46b937fbd",
  "docs/walk_ref/quality_runs/run_05/metrics.json": "2733bd1fe9406b5f654702b5f767fd263607e86cb7d21b19fa7e4dd53d4482c9",
  "docs/walk_ref/quality_runs/run_05/quality_metrics.json": "f5ba7112ed21155376d148d8811ee1bf972fb59c6e765c89d4a7cb66a05bb6af",
  "docs/walk_ref/quality_runs/best_render/eval.mp4": "bb208100ba7ac8b951973fb200a76860462c61bdf56c4a4a6efc1810269776df",
  "docs/walk_ref/refinement_runs/run_05/reference.npz": "9bbdf82a6cbf4495871f4a06453985a86271b68d57d8f9ddeac7aede074b3d89"
}
```

The complete fingerprint manifest is [protected_before.json](docs/walk_ref/archive/protected_before.json).

## How to continue safely

1. Read this handoff, CURRENT_BEST, the quality protocol, diagnosis and active leaderboard. Inspect the canonical video to understand the gait that must be preserved.
2. Check `git status`; do not discard the cleanup or other existing user changes. Resolve symlinks when inspecting dependencies.
3. Wait for the next user instruction. Cleanup and the bounded campaigns are complete; do not automatically restart an optimization loop, regenerate exports, or retrain.
4. If future experimentation is requested, start from **quality Run 05**, create separate outputs and explicit fixed-baseline gates, preserve the original artifacts, and use the user's newly authorized experiment budget. Speed remains a constraint, not a reason to prefer a shakier walk.
5. Report physical smoothness, saturation, bobbing, action rate, survival and visual cadence honestly, including tradeoffs and measurement limitations. Never relax a hard gate silently or compare acceleration only to a moving parent.
