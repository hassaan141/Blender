# Existing ONNX exports

`bingo_walk_reference/` and `bingo_walk_reference_smooth/` contain historical ONNX policies and their original runtime manifests. They are retained as export-interface history and are **not exports of the current locomotion baseline**. Their manifest provenance, reference timing and rollout verification must not be substituted for current-baseline settings.

The [current baseline](../docs/walk_ref/CURRENT_BEST.md) remains a PyTorch checkpoint. The unchanged [walk-ref exporter](../rl/tools/export_walk_ref_onnx.py) documents the exporter interface; any future export needs a new destination and verification against the exact current reference, EMA and phase mapping. Cleanup did not generate or overwrite exports.
