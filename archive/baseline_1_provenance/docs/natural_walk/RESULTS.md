# Natural walk loop: three attempts, no visual pass

**The requested natural-walk milestone was not achieved.** Locomotion 1 remains the retained best. Three short, focused PPO attempts completed; none was promoted. All survived the recorded 60-second evaluation, but the videos did not establish a convincingly more natural four-leg walk.

## Deliverables

mimi

- Strongest experimental reference: [reference_02b/reference.npz](reference_02b/reference.npz), with [construction record](reference_02b/reference.json) and [kinematic preview](reference_02b/kinematic.mp4).
- Strongest experimental checkpoint: [attempt_03/policy.pt](attempt_03/policy.pt). **Rejected as a replacement baseline**; retained for inspection and reproducibility.
- [Matched close-up comparison video](attempt_03/comparison.mp4), Locomotion 1 on the left, attempt 03 on the right; original playback rate and camera, identical crop. [Candidate video](attempt_03/evaluation/eval.mp4).
- [Full leaderboard](leaderboard.csv), [all comparison metrics](metrics_comparison.json), [paw/contact comparison](attempt_03/paw_comparison.png), and [raw paw trajectories](attempt_03/evaluation/paw_trajectories.npz).
- [Standard metrics](attempt_03/evaluation/metrics.json), [body/joint diagnostics](attempt_03/evaluation/quality_metrics.json), [contact/slip/limit metrics](attempt_03/evaluation/natural_metrics.json).
- [Exact retained and experimental artifact paths](best.json); [safe experimental replay](reproduce_experimental.sh).

## What was tried

| Attempt | Reference/control change | Decision |
|---|---|---|
| 01 | Original Bingo walk, 4 mm dog-shaped swing-lift addition on all feet; 15% blend toward lightly smoothed task-space paths; original contact timing | REJECT: nearly unchanged visual gait; FL knee saturation increased |
| 02 | Remove additional front/right lift; 6 mm BL lift during each existing forward recovery stroke; correct missing first BL swing window | REJECT: less BL slip but persistent skimming; slower and slightly higher acceleration |
| 03 | Same reference as 02; replace only the BL quarter of the existing contact reward with collision-hull clearance rather than tip-height classification | REJECT: strongest local clearance improvement, but not a convincing natural gait |

Every attempt started from the same immutable Locomotion 1 quality Run 05 checkpoint: seed 42, 512 environments, 100 PPO iterations / 2400 steps. Exact command, training parameters, final checkpoint and isolated source snapshots are retained per attempt. The third attempt changed a contact-measurement definition, not its weight; other reward terms and the residual controller were unchanged. No AMP training, new general framework, or physics changes were made.

A preliminary `reference_02/` was not trained: inspection showed it retained the missing first BL swing window. `reference_02b/` corrected that before attempt 02. An 8 mm prototype exceeded the conservative IK-change guard and was reduced to 6 mm before replay/training. Reference 01's zero-residual PD test fell after roughly 2.3 seconds; its retained-policy preflight survived. This work relies on learned balance corrections, not a claim that direct PD replay is independently stable.

## Metrics: baseline versus strongest new candidate

One-second settling followed by 60 seconds at a held 0.25 m/s command; one robot, same source metric definitions and camera.

| Metric | Locomotion 1 | Attempt 03 |
|---|---:|---:|
| Survival | 100% | 100% |
| Mean vx, m/s | 0.19294 | 0.18631 |
| vx std, m/s | 0.05269 | 0.04801 |
| Max vx, m/s | 0.36584 | 0.36348 |
| Full reference-loop period, s | 2.12191 | 2.12191 |
| Joint acceleration RMS, rad/s² | 11.0002 | 11.5341 |
| Residual action-rate RMS | 0.018130 | 0.018380 |
| Mean joint torque saturation | 4.676% | 4.977% |
| FL SP / knee saturation | 10.417% / 15.972% | 10.694% / 15.486% |
| BL geometric contact duty | 95.49% | 85.42% |
| BL collision clearance, 95th percentile | 2.91 mm | 7.12 mm |
| BL contact slip RMS estimate | 0.373 m/s | 0.284 m/s |
| Mean per-foot slip RMS estimate | 0.247 m/s | 0.218 m/s |
| Body roll RMS | 5.482° | 4.927° |
| Body pitch RMS | 2.011° | 2.026° |
| Detrended body-height RMS | 1.510 mm | 1.490 mm |
| Mean body height | 0.171860 m | 0.171940 m |
| Recorded leg joint-limit violation | 0 rad | 0 rad |

The 2.12-second reference loop contains multiple per-leg strokes; it is not one footstep. Per-leg touchdown phase lists, median touchdown intervals, duty factors and trajectories are in `natural_metrics.json` and `paw_trajectories.npz` for every evaluated candidate. All per-joint saturation percentages are in `metrics.json`.

Contact means lowest collision-hull clearance below 3 mm; slip is sampled horizontal velocity of the current lowest hull material point during those intervals. These are geometric estimates, **not contact-force measurements**. Slower candidate motion can explain some slip reduction; do not attribute the whole reduction to better planting.

## Why this did not pass

The initial reference modification improved a marker path, but barely changed the loaded foot's recovery. A narrow clearance peak does not give a clean swing throughout the forward stroke. In the baseline, the BL foot slid forward while staying near the floor for most of the cycle. The strongest candidate reduces this but still does it too much. The same overall mechanical stepping pattern remains in the matched comparison. Acceleration and average saturation also increased slightly, and mean speed moved farther below the initial 0.20–0.30 target.

The measurements identify two concrete failures in the small-edit strategy:

1. **Reference clearance was not preserved by the controller.** On attempt 01, predicted BL collision clearance during marked swing averaged 5.25 mm at the reference pose, but only 1.54 mm at the residual-adjusted target. Actual joint tracking was close to that adjusted target. This is not explained simply by motors failing to track. See [clearance diagnosis](clearance_diagnosis.json) and [tracking diagnosis](tracking_diagnosis.json).
2. **The existing contact reward missed successful clearance.** For attempt 02 it labelled BL in contact for 100% of samples, even though the collision geometry showed 134 clear samples during intended swing. The targeted correction in attempt 03 helped, but did not produce sustained clean recovery. See [contact-label diagnosis](contact_label_diagnosis.json).

These three runs are evidence that **small paw-tip lift edits plus short adaptation under this controller/objective are insufficient for the requested result**. They do not prove that reference-guided imitation RL is fundamentally wrong. A future revision would need to solve the actual collision-support trajectory throughout recovery, along with stance planting and body support, before another training loop. No such extra loop was started here.

## Sources and visual review

The existing Blender-derived Bingo reference supplied posture, timing, phase order and body motion. The short InterPet segment supplied only normalized clearance shape from four complete swing bouts; no dog joint angles, native cadence, or large root wobble were imposed. Its inferred contacts were asymmetric and its root trace imperfect, so it was not treated as exact ground truth. Source measurements are in [source_comparison.json](source_comparison.json). July v9 remains a historical visual coordination benchmark, not a transferable current-v4 checkpoint.

Visual review used kinematic sequences before training, matched close-up stride sequences and full-run overview samples afterward. This is sampled-frame inspection, not an automated naturalness score or a substitute for the user's video judgment. The full videos are supplied. No candidate is called a visual success on the basis of its reward or metrics.

## Protection and reproduction

All **940 protected hashes remain unchanged**. The separate baseline evaluation reproduced every original standard metric/metadata field except the intentionally different isolated task name. See [completion audit](completion_audit.json), [baseline reproduction check](baseline_reproduction_check.json), and [protected fingerprints](protected_sha256.json).

New work is confined to `docs/natural_walk/` and `rl/bingo_rl/bingo_rl/natural_walk/`. Canonical Locomotion 1 files, Stage 1–5, URDF/physics, gains, effort limits, and prior results were not edited. Nothing was committed.

To inspect the strongest rejected candidate again, run from the repository root:

```sh
bash docs/natural_walk/reproduce_experimental.sh
```

This writes to a fresh directory under `docs/natural_walk/`. It does not retrain or overwrite the canonical baseline. Three training attempts have been used; `run_train.py` enforces the limit.
