# Locomotion history: what to watch and why

Start with the current baseline, then follow the progression below. This is a
navigation guide, not another runtime specification or leaderboard. The frozen
BASELINE_1 README remains authoritative for using the robot.

[Current baseline and limitations](../../BASELINE_1/README.md) · [All 69 recorded run rows](../../BASELINE_1/history/RUN_INDEX.csv) · [Complete video inventory](videos.csv)

## Main progression

These clips were selected from the recorded outcomes and reports. They are a
short handoff viewing list, not a new visual re-evaluation or a claim that every
stage improved the gait. Metrics across different campaigns are not automatically
comparable: reference, commanded speed, physics version, camera and measurement
definitions can differ. Compare paired videos and each campaign’s own protocol.

| Stage | Watch | Read | What carried forward / lesson |
|---|---|---|---|
| Early command curriculum | [Video](../../archive/baseline_1_provenance/archive/walk_ref/command_locomotion/eval_video/closeup_forward_S6.mp4) | [Report](../../archive/baseline_1_provenance/archive/walk_ref/command_locomotion/README.md) | Earlier velocity curriculum reached 6/7 gates; torque saturation remained. Historical controller, not the current checkpoint. |
| Slow reference walk | [Video](../../archive/baseline_1_provenance/docs/walk_ref/best/eval.mp4) | [Report](../../archive/baseline_1_provenance/docs/walk_ref/experiments/summary.md) | Quality Run 05 reduced acceleration 12.54→11.00 rad/s² and bobbing 2.36→1.51 mm while keeping cadence. The earlier >20% acceleration selection bug is documented here too. |
| Natural-walk reference edits | [Video](../../archive/baseline_1_provenance/docs/natural_walk/attempt_03/comparison.mp4) | [Report](../../archive/baseline_1_provenance/docs/natural_walk/RESULTS.md) | Task-space paw edits and contact-definition correction helped locally, but all three attempts were rejected as a visual replacement. Attempt 03 later became a teacher; that does not retroactively make it a visual success. |
| Backward walking | [Video](../../archive/baseline_1_provenance/docs/experiment_loop/backward_start_vs_best.mp4) | [Report](../../archive/baseline_1_provenance/docs/experiment_loop/BACKWARD_RESULTS.md) | Signed phase and backward-specific geometry led to retained loop 7. Preserve the start-versus-best comparison and the remaining tracking limitations. |
| Left/right turning | [Video](../../archive/baseline_1_provenance/docs/experiment_loop/turning/loops/loop_0005/A/left_right.mp4) | [Report](../../archive/baseline_1_provenance/docs/experiment_loop/turning/TURNING_RESULTS.md) | Heading-relative observations and yaw-reference banks produced retained loop 5; contact/slip and turning-quality limits remained. |
| Unified policy | [Video](../../archive/baseline_1_provenance/docs/experiment_loop/command/loops/loop_0006/teacher_comparison.mp4) | [Report](../../archive/baseline_1_provenance/docs/experiment_loop/command/RESULTS.md) | Actual rollouts from incompatible teachers were distilled into a shared 95-value interface, then PPO fine-tuned. Loop 6 supplied the final neural checkpoint, but was not a fully solved command gait. |
| Why stronger turns were rejected | [Video](../../archive/baseline_1_provenance/docs/experiment_loop/command/loops/loop_0008/evaluation/stand.mp4) | [Report](../../archive/baseline_1_provenance/docs/experiment_loop/command/loops/loop_0008/report.md) | Loop 8 improved yaw but retained an unacceptable leaning stand. This failure is worth keeping: stronger command tracking alone was not sufficient. |
| Final stand repair | [Video](../../BASELINE_1/videos/locomotion_to_stand.mp4) | [Report](../../archive/baseline_1_provenance/docs/experiment_loop/command/stand_fix/RESULTS.md) | The checkpoint stayed unchanged. Runtime uses STAND_SOLVED and fades conflicting residual authority only near zero command; held moving-command metrics were preserved. |
| Current integrated baseline | [Video](../../BASELINE_1/videos/all_commands.mp4) | [Report](../../BASELINE_1/README.md) | Use this matched policy/runtime/reference bundle for reproduction. It has the corrected upright stand, but pivot/backward yaw and extreme-command limitations remain. |

## Useful side branches

These preserve the process without presenting abandoned approaches as current baselines.

| Branch | Watch | Read | Why retain it |
|---|---|---|---|
| July AMP v9 benchmark | [Video](../../archive/baseline_1_provenance/docs/JulyAMPProgress/bingo_amp_FINAL_v9.mp4) | [Report](../../archive/baseline_1_provenance/docs/JulyAMPProgress/July9AMP.md) | A separate historical visual benchmark, not part of the current policy lineage or directly comparable physics. |
| Retargeting: kinematics versus physics | [Video](../../archive/baseline_1_provenance/docs/locomotion2/motions/stage4/stage4_physics.mp4) | [Report](../../archive/baseline_1_provenance/docs/locomotion2/LOCOMOTION2_PIPELINE_REPORT.md) | Keep this alongside the kinematic replay: valid IK did not imply a viable physical gait. |
| Separate AMP exploration | [Video](../../archive/baseline_1_provenance/docs/locomotion2/amp/loop_runs/run_06/eval_video/rl-video-step-0.mp4) | [Report](../../archive/baseline_1_provenance/docs/locomotion2/amp/AMP_PIPELINE_REPORT.md) | A separate experiment branch; it was not merged into the final reference/residual controller. |

The retargeting [kinematic replay](../../archive/baseline_1_provenance/docs/locomotion2/motions/retarget/kinematic_replay.mp4) is the counterpart to its physics failure above.

## Finding the complete process

- `videos.csv` indexes all 331 existing MP4s: 12 in BASELINE_1 and 319 in the
  provenance archive (about 851 MiB of archived video). Paths are relative to the
  repository root. Search by campaign, run/loop name, command or `comparison`.
- The CSV records file size and SHA-256 so exact duplicate content can be
  recognized without deleting it or confusing it with another experiment.
- The 69-row run index and its adjacent original ledgers preserve decisions and
  reasons. Reports, hypotheses, configuration/source snapshots, metrics, logs,
  checkpoints and original videos remain in the archive beside their runs.
- Historical reports may contain original absolute paths or links to superseded
  bundles. Use this guide’s current links, the video inventory, and the baseline
  provenance map to locate moved artifacts. Historical reports are evidence,
  not instructions to overwrite or promote an old checkpoint.
- Render probes and diagnostic captures are searchable but are not on the main
  viewing list. In particular, the unified campaign recorded black-camera probes;
  inspect the associated report before treating a capture as gait evidence.

## Keep history useful without multiplying files

Keep the canonical bundle, milestone comparisons, instructive failures, original
run ledgers, and the evidence needed to reproduce decisions. Leave detailed old
runs archived. Link to existing videos rather than copying or re-encoding them.
A rejected run that explains a selection bug, physical failure, or regression is
valuable; unsuccessful work should not be erased merely because it failed.

For future experiments, add one concise record of hypothesis, targeted change,
measured result, visual decision, and links to its representative video/report.
Keep extra camera/debug captures out of the main viewing list. Consider removing
only verified duplicates, caches or disposable intermediates after confirming
that no unique result, reproduction dependency or manifest depends on them.

This documentation pass deletes no files and changes no baseline, historical
artifact, checkpoint, video, metric, source code or physics setting. It adds only
this guide and a CSV index, with entry-point links. No commit or push is made.
