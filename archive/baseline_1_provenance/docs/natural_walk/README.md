# Bingo natural walk experiment

Status: bounded loop finished after three training attempts. **No candidate passed the visual gate; Locomotion 1 remains best.** Read [RESULTS.md](RESULTS.md) for the comparison video, reference/checkpoint paths, full metrics, and evidence.

This work inherits the immutable Locomotion 1 residual PPO controller and validated v4 physics. New task registration, evaluation and training entrypoints live in `rl/bingo_rl/bingo_rl/natural_walk/`. Original source and protected assets are read-only.

- `PROTOCOL.md`: limits and acceptance criteria.
- `source_comparison.json`: measured Bingo/Blender and InterPet source properties. The 2.12 s reference loop contains multiple foot events; it is not necessarily one foot stride.
- `protected_sha256.json`: pre-work protected file fingerprints.
- `reference_01/`: first conservative task-space edit, IK provenance, kinematic video, physics checks and visual training gate.
- `attempt_*/`: at most three training attempts, command/config records, deterministic final checkpoint, 60-second video/metrics, and matched close-up comparisons.
- `leaderboard.csv`: physical metrics generated from completed evaluations; visual decisions are recorded separately and control acceptance.

The InterPet extraction is a short, imperfect recovered motion, with asymmetric inferred contacts and nonuniform timing. Only normalized clearance shape from four complete swing bouts is used initially. Native dog cadence, root wobble and joint angles are not imposed on Bingo. The existing Blender-derived Bingo reference supplies posture, cycle timing, contact order and body motion.

Evaluation extends the existing metric definitions. New contacts use the lowest collision-hull point within 3 mm of the floor; slip is sampled horizontal material-point speed in those intervals. These remain geometric proxies, not measured contact forces. Raw trajectories are retained so the labels can be audited.

Example evaluation (fresh output directory required):

```sh
python3 docs/natural_walk/run_eval.py review_new --reference docs/natural_walk/reference_01/reference.npz --checkpoint docs/natural_walk/attempt_01/policy.pt
```

No source, physics, previous experiment result, or canonical Locomotion 1 artifact is overwritten. Direct-PD failure is reported separately from the learned-controller survival result.
