# Canonical locomotion baseline

**Retained run:** `/pub0/muhammadf/Blender/docs/walk_ref/quality_runs/run_05/`.
Do not modify, retrain in place, rename, or overwrite anything in this directory or its final video directory.

| Artifact | Authoritative location |
|---|---|
| Checkpoint | [quality_runs/run_05/policy.pt](quality_runs/run_05/policy.pt) |
| Evaluation metrics | [quality_runs/run_05/metrics.json](quality_runs/run_05/metrics.json) |
| Physical-quality metrics | [quality_runs/run_05/quality_metrics.json](quality_runs/run_05/quality_metrics.json) |
| Final video | [quality_runs/best_render/eval.mp4](quality_runs/best_render/eval.mp4) |
| Exact environment snapshot | [bingo_walk_ref_env.py](quality_runs/run_05/bingo_walk_ref_env.py) |
| Exact configuration snapshot | [bingo_walk_ref_env_cfg.py](quality_runs/run_05/bingo_walk_ref_env_cfg.py) |
| Training provenance | [experiment.json](quality_runs/run_05/experiment.json) and [train command](quality_runs/run_05/train_command.sh) |
| Evaluation implementation | [eval_walk_loop.py](../../rl/tools/eval_walk_loop.py) and [diagnostics](../../rl/tools/walk_quality_diagnostics.py) |
| Runtime reference | [refinement_runs/run_05/reference.npz](refinement_runs/run_05/reference.npz) |

Measured survival 100%; vx mean/std/max **0.192935 / 0.052690 / 0.365836 m/s**; joint acceleration **11.000194 rad/s²**; action-rate RMS **0.0181303**; front-left SP/knee saturation **10.4167% / 15.9722%**; detrended body bobbing **1.50977 mm**; cadence **2.121914 s**.

## Exact runtime and reproduction

Run from the repository root:

```sh
bash docs/walk_ref/best/reproduce.sh
```

This runs the original 60-second evaluation and generates a new video under a fresh `outputs/walk_ref_reproduction/` directory. It uses GPU 1, `/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python`, task `Bingo-WalkRef-v4-Play-v0`, one environment, held vx command 0.25 m/s, zero lateral/yaw command, one-second settling, diagnostics, headless mode and the original Kit flags. The original [render command](quality_runs/best_render/render_command.sh) is retained only as provenance: running it verbatim would overwrite the canonical video.

The [recorded runtime versions](best/runtime_versions.json) identify the Python, Torch, NumPy, Gymnasium and skrl environment used for the cleanup replay. The live environment/config match the saved snapshots. Control runs at 24 Hz (120 Hz physics, decimation 5), EMA alpha 0.3, residual scale 0.3 rad, phase multiplier 1.4. Reward weights: action rate 0.06, joint acceleration 0.2, FL torque 0.03, body height 0.05 (target 0.17165698111057281 m), vertical velocity 0.01. Full settings, including inherited physics, remain in the unchanged source and training parameters.

The older refinement Run 05 supplies the required runtime reference through a compatibility link; do not remove that path. The preceding quality Run 04 checkpoint is retained for training provenance. Winning [training logs and parameter YAMLs](../../logs/skrl/walk_quality/run_05/) and the winning Hydra launch directory are unchanged. [Pre-cleanup SHA-256 fingerprints](archive/protected_before.json) identify the exact protected files; [sanity results](archive/sanity_check.json) record the replay and integrity checks.

Future work starts from this checkpoint. Preserve survival, slow cadence, and gait quality; speed is a constraint, not an optimization target.
