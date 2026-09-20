# Autonomous backward walking — final result

Stopped at eight total loop directories. Best retained candidate: **loop_0007/A**. **The full natural-walk goal was not met.** It is a provisional backward best, not a replacement for Locomotion 1. No loop 9 was started.

- [Best checkpoint](loops/loop_0007/A/policy.pt)
- [Required paired reference](loops/loop_0007/A/reference.npz)
- [Best 60-second video](loops/loop_0007/A/evaluation/eval.mp4)
- [Starting loop 1 versus best loop 7](backward_start_vs_best.mp4)
- [Loop 5 versus loop 7 comparison](loops/loop_0007/A/comparison.mp4)
- [Exact evaluation command](loops/loop_0007/A/evaluate.command.json), [runtime overlay](loops/loop_0007/A/evaluate.effective_settings.json), [complete config](loops/loop_0007/A/config.json)

## Metrics

One deterministic 60-second rollout per trained candidate, after one second of settling. Survival is the observed rollout result, not a multi-seed reliability estimate.

| Metric | Starting loop 1 | Best loop 7 |
|---|---:|---:|
| Survival | 1.00000 | 1.00000 |
| Mean vx (m/s) | -0.13658 | -0.18300 |
| vx std (m/s) | 0.08047 | 0.06506 |
| Max absolute vx (m/s) | 0.32479 | 0.34294 |
| Joint acceleration RMS (rad/s²) | 9.16329 | 7.98256 |
| Action-rate RMS (rad/control step) | 0.01204 | 0.01359 |
| Roll RMS (deg) | 7.72790 | 5.37893 |
| Pitch RMS (deg) | 1.23834 | 1.41653 |
| Body bob RMS (mm) | 1.98449 | 1.45726 |
| Mean foot slip RMS (m/s) | 0.20737 | 0.14705 |
| Mean joint saturation (%) | 2.80671 | 1.89236 |
| RR knee saturation (%) | 7.91667 | 5.00000 |
| Yaw-rate RMS (rad/s) | 0.26490 | 0.27272 |
| Heading change (deg) | 0.61103 | 3.40865 |

Best mean body height: 0.17066 m. Joint-limit violation: 0.0 rad. All 1,440 evaluated frames survived; no warmup fall. Signed vx range: -0.3429 to 0.0331 m/s. Fixed acceleration ceiling is 1.2 × 11.00019 = 13.20023 rad/s², not a rolling baseline.

Remaining failures: vx std **0.06506 > 0.06 m/s** and yaw-rate RMS **0.27272 > 0.15 rad/s**. Lateral rocking and finite slip remain. Action-rate RMS and yaw RMS are slightly worse than loop 1 despite the overall improvement. Do not describe this as a fully solved natural backward walk.

## Gait measurements

Reference duration remains 2.65179 seconds for three native strokes: approximately 1.1313 strokes/s. The commanded speed remains −0.20 m/s. No speed/cadence reward increase was used.

| Foot | Contact duty | Clearance p95 (mm) | Median touchdown interval (s) | Slip RMS (m/s) |
|---|---:|---:|---:|---:|
| FL | 0.5361 | 26.87 | 0.8333 | 0.1444 |
| FR | 0.6229 | 30.38 | 0.8750 | 0.2011 |
| BL | 0.6715 | 11.52 | 0.8333 | 0.1126 |
| BR | 0.5979 | 10.26 | 0.8333 | 0.1301 |

Full touchdown phases and paw ranges are in [natural_metrics.json](loops/loop_0007/A/evaluation/natural_metrics.json); frame-level paw trajectories and contacts are in [paw_trajectories.npz](loops/loop_0007/A/evaluation/paw_trajectories.npz). Contacts use lowest collision-hull clearance <3 mm, not a force sensor. Slip is horizontal velocity of the lowest hull material point during geometric contact. These measurements have proxy limitations.

## Loop history

| Loop | Targeted change | Decision | Mean vx | vx std | Accel RMS | RR knee saturation |
|---|---|---|---:|---:|---:|---:|
| 01 | Signed backward reference; attempt03 warm start | REJECT | -0.137 | 0.080 | 9.163 | 7.917 |
| 02 | Lateral geometry trial; file-copy failure before training | FAILED | — | — | — | — |
| 03 | Lateral excursion 1.00 → 0.70 | KEEP | -0.159 | 0.076 | 8.696 | 7.153 |
| 04 | 35% stance-shape blend (old contacts) | REJECT | -0.170 | 0.069 | 9.376 | 8.264 |
| 05 | Correct contact timing + collision-hull contact measurement | KEEP | -0.175 | 0.075 | 7.898 | 4.514 |
| 06 | Lateral excursion 0.70 → 0.45 | REJECT | -0.175 | 0.072 | 7.979 | 6.736 |
| 07 | 20% stance-shape blend (corrected contacts) | KEEP | -0.183 | 0.065 | 7.983 | 5.000 |
| 08 | Residual EMA alpha 0.30 → 0.20 | REJECT | -0.183 | 0.066 | 7.735 | 7.292 |

Loop 1 was the existing, user-authorized backward starting point; its original goal REJECT remains intact. Loops 3, 5 and 7 were provisional KEEP decisions for overall improvement, not full success. Loop 2 failed before training because copying a sealed reference preserved read-only permissions. The copy operation was fixed; loop 2 remains immutable. Six new candidates completed training and 60-second evaluation (3–8). Each used 200 PPO iterations, 512 environments, seed 42, and the current retained backward checkpoint. No A/B sweep was run.

## Why the retained changes helped

1. Reducing lateral excursion to 70% about each foot’s unchanged mean position reduced excessive side-to-side reach while preserving stance width, clearance and cadence. Loop 3 improved support balance and rocking.
2. Reference contact labels for FL/FR/RR were aligned with their three actual recovery strokes. Contact matching then used the same collision-hull measurement for all feet, extending the prior BL correction. Loop 5 reduced conflicting planting guidance, slip and knee loading.
3. With consistent contact timing, a conservative 20% monotone stance/smooth-swing blend improved fore-aft planting and speed regularity in loop 7. The 35% blend under old contact labels was rejected.

These mechanisms are supported by the reference audits and rollout changes, but not isolated causal proof: each trial also continued PPO training, and only one seed was evaluated. The visual reviews used dense 8-fps stride montages and reference replays, with full-duration overview checks where noted; the complete videos are retained for inspection.

The 45% lateral path and stronger EMA were rejected because the modest smoothing gains came with substantially worse RR knee loading. Reference joint angles were never replaced with dog angles. No AMP, URDF, physics, Kp/Kd, effort limits or protected source changes were used.

## Reproduction and integrity

Use the checkpoint, reference and runtime overlay together. The original native Hydra YAML is written before the runtime override, so it is not sufficient by itself. The saved train/evaluate commands invoke `backward_worker.py`; `BINGO_NATURAL_REFERENCE` selects the paired reference and `BINGO_NATURAL_CONTACT_FIX=1` enables the inherited BL correction. The config additionally enables `all_contacts_corrected=1` for this best candidate. Output from any reproduction must go to a fresh directory; sealed loop directories are read-only.

Historical `changed_files.json` text says “no reward/actuator changes”; the precise meaning is no reward-weight or actuator changes. The contact reward measurement was deliberately corrected in loop 5. The proposal, runtime overlay and source snapshots are authoritative.

[Final validation](FINAL_VALIDATION.json): 1,577 protected inventory entries unchanged; 940 historical hashes unchanged; all eight loop manifests valid; best checkpoint/reference/metrics/video hashes valid; forward champion pointer unchanged; all 18 runner regression tests passed. The continuous runner exited after loop 8.

The autonomous agent supplied planning, kinematic review and visual decisions through `agent_bridge.py`; no human approval was requested between loops. The bridge requires an active agent, not an unattended random search. Internal boundary reloads registered targeted geometry options; no historical loop was overwritten.
