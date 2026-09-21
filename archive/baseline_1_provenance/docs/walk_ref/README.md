# Current reference-guided walk

Start with [CURRENT_BEST.md](CURRENT_BEST.md). The canonical baseline is **quality campaign Run 05**, preserved in its original directory. Earlier campaigns also have a Run 05; those are different checkpoints.

- [Latest measured metrics](best/metrics.json), [physical-quality diagnostics](best/quality_metrics.json), and [evaluation video](best/eval.mp4).
- [Safe reproduction command](best/reproduce.sh), using the original evaluator and runtime.
- [Single experiment leaderboard](experiments/leaderboard.csv) and [handoff summary](experiments/summary.md).
- [Current quality protocol](quality_runs/PROTOCOL.md) and [front-left saturation diagnosis](quality_runs/DIAGNOSIS.md).
- [Archive index](archive/README.md): older experiments, selection-bug audit, and legacy architecture.

Architecture: a cyclic motion reference supplies joint targets and expressive channels. The policy supplies 12 leg residuals, filtered with EMA and scaled before addition to the reference. Phase follows the held forward command; the retained reference timing and phase mapping preserve the slow cadence. The exact implementations are [environment](../../rl/bingo_rl/bingo_rl/walk_ref/bingo_walk_ref_env.py) and [configuration](../../rl/bingo_rl/bingo_rl/walk_ref/bingo_walk_ref_env_cfg.py).

The `best/` metrics/video are symbolic links, not new copies. `quality_runs/run_05/` and `quality_runs/best_render/` are immutable. Legacy campaign names are compatibility links into `archive/`, needed by unchanged source and recorded commands. Do not launch old optimization scripts as a current workflow.

The two [existing ONNX exports](../../exports/README.md) are historical exports, not this baseline. The current baseline is the PyTorch checkpoint above. The Stage 1–5 animation pipeline remains authoritative and unchanged; see the [pipeline documentation](../README.md).
