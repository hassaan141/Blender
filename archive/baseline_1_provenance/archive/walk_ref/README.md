# Locomotion archive

This archive holds historical evidence, not active baseline candidates.

- `loop_runs/`: original eight-run campaign, fixed selection logic, and the pre-audit record.
- `refinement_runs/`: eight-run refinement, including the runtime reference still required by the current baseline.
- `quality_history/`: diagnosis, protocol, accepted ancestry, rejected experiments, helper scripts and original ledgers from the six-run quality campaign.
- `early_walk_ref/`: initial C/D/smoothing/PD-only evaluations and videos.
- `command_locomotion/`: earlier command-conditioned architecture and evaluations; distinct from the current reference-guided walk.
- [Archived training logs](/pub0/muhammadf/Blender/archive/locomotion_training/): historical final/best checkpoints, config YAMLs, events and Hydra launch records. Winning quality Run 05 training logs were not moved.

Original paths are compatibility symlinks so unchanged source, historical commands and document links continue to resolve. Relative paths in historical documents are interpreted through those original locations (for example [legacy locomotion docs](/pub0/muhammadf/Blender/docs/locomotion/README.md)). Historical result decisions belong to their own protocol; the current entrypoint is [CURRENT_BEST](/pub0/muhammadf/Blender/docs/walk_ref/CURRENT_BEST.md).

[cleanup_manifest.json](cleanup_manifest.json) lists moves and deletions with hashes and reasons. [protected_before.json](protected_before.json) fingerprints source, canonical assets, baseline artifacts and winning training records before cleanup. [sanity_check.json](sanity_check.json) records the final integrity/replay check. Unique old videos and exports were retained; no byte-identical historical videos were found.
