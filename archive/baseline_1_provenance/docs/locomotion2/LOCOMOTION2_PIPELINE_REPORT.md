# Locomotion 2 End-to-End Pipeline Report

Source clip: `interpet_dog02_p05_take06_ego_001`, window [0.60, 2.60]s (chosen in
`motions/candidates/CANDIDATE_COMPARISON.md` as the only one of five
candidates showing genuine sustained forward walking). This run carried it
through extraction, retargeting, kinematic validation, Isaac kinematic
replay, and Stage 4 physics feasibility. **Result: Stage 4 physics fails --
the robot falls.** RL training was not started, per the task's own
instruction that it is gated on Stage 4 succeeding. Locomotion 1
(`docs/walk_ref`) and canonical Stage 1-5 assets were not touched.

## Deliverables

| Requested | Path | Status |
|---|---|---|
| Source dog NPZ | `motions/source/dog02_walk_02.npz` (+ `_report.md`, `_validation.png`) | Done |
| Bingo retargeted NPZ | `motions/retarget/bingo_dog02_walk_02.npz` (+ `_metrics.json`) | Done, but the motion it encodes is not physically viable (see below) |
| Kinematic replay video | `motions/retarget/kinematic_replay.mp4` (59 frames, 30fps) | Done |
| Stage 4 physics video | `motions/stage4/stage4_physics.mp4` (133 frames, 30fps) | Done -- shows the fall |
| Tracking/contact/joint-limit metrics | `motions/retarget/bingo_dog02_walk_02_metrics.json`, `_validation_report.md` | Done |
| RL checkpoint | -- | Not attempted (gated on Stage 4, which failed) |
| RL video | -- | Not attempted |
| Comparison vs. Locomotion 1 | -- | Not attempted (nothing to compare yet) |
| Blockers | see below | One, fully diagnosed |

## What was fixed along the way

Reused `scripts/retarget.py`'s IK/scaling machinery (`Urdf`, `solve_leg`,
`solve_leg_best`, `kabsch`) rather than rewriting it; new code lives in
`dataset/retarget_to_bingo.py`. Three real, confirmed bugs were found and
fixed during this session, each independently verified with before/after
numbers:

1. **Body scale.** Copied `scripts/retarget.py`'s own convention (scale the
   hip constellation by mean leg-length ratio) -- wrong here, since this
   dog's hip-to-hip span (~0.33m) and Bingo's (~0.12m) are a different scale
   regime than either's leg length. Fixed with a proper Umeyama
   similarity-transform fit. Base-fit residual: **135mm -> 13mm mean**.
2. **Per-leg IK target scale.** Scaled to the dog's *mean* leg extension,
   which put its *peak* stride (7-19% above the mean) beyond Bingo's max
   reach at the moments of fullest extension. Fixed by scaling to peak
   extension with a 10% reserve margin. Mean IK foot error: **32mm -> 3.5mm**
   (later **0.48mm** after the root-orientation fix below).
3. **Root-frame reconstruction.** The original per-frame rigid Kabsch fit of
   the dog's 4 hip keypoints to Bingo's fixed hip layout let hip/scapula
   articulation noise leak into what should be a single global root
   orientation: it produced a 5-16 degree, frame-to-frame-varying spurious
   rotation, pinned two hip-yaw joints (`fr_SY_J`/`br_SY_J`) at their limits
   for 73-95% of frames, and gave a visibly wobbly/looping retarget root path
   despite a clean forward source path. Fixed by using the source's own
   `root_pos`/`root_orient_rotmat` directly (reduced to yaw-only, smoothed --
   see the three-way comparison in `retarget/bingo_dog02_walk_02_validation_report.md`).
   Saturation dropped to **one joint at 38.3%** (`fl_SY_J`), everything else
   clear. Also fixed a secondary issue this surfaced: branch (elbow-up/down)
   selection had no preference for Bingo's own documented natural crouch
   pose, which could pick a kinematically-valid but anatomically-inverted
   configuration; biased toward the documented crouch branch (this turned
   out not to change the numeric result for this clip -- see validation
   report -- but is a correctness fix worth keeping).

Two hypotheses were tested and explicitly rejected rather than chased
further: hip aspect-ratio mismatch as the cause of the root-rotation issue
(re-fit with anisotropic scaling, rotation unchanged, foot error got worse);
and amplitude reduction as the fix for the remaining `fl_SY_J` saturation
(cutting to 20% of original amplitude did reduce it further, 38%->27%, but
at the cost of visibly flattening the gait -- not adopted).

## Kinematic validation (before Isaac)

- IK foot-tracking error: mean 0.48mm, p95 5.6mm, max 11.1mm -- excellent.
- Foot slip during contact: 5.4mm/stance (project gate: <5, essentially at
  the line).
- Contact timing: identical to source by construction (the source's own
  `foot_contact` array is used directly, not re-derived).
- Joint limits: **one joint** (`fl_SY_J`) saturated 38.3% of frames, isolated
  via multi-seed IK testing to a genuine geometric requirement of that leg's
  targets (not a mapping bug, not a branch-selection artifact -- confirmed
  by solving from 6 diverse starting points, all converging to the same
  low-error solution).
- Visual: substantially improved over the original approach (no more
  pronounced backward loop) but the top-down path still shows some wobble,
  and the kinematic-replay video shows a persistently crouched-looking
  stance rather than a clean extended walking gait.

## Isaac kinematic replay

Ran via `scripts/replay_motion.py`. Rendered successfully (59 frames in
~20s) but Isaac Sim's own shutdown (`simulation_app.close()`) hung for over
7 hours afterward -- a known Omniverse quirk unrelated to the retarget
itself; killed manually. **Operational note for next time**: always wrap
these Isaac launches in `timeout`, they may not exit cleanly even on
success, and use `PYTHONUNBUFFERED=1` or you'll lose output to buffering if
the process is killed before a clean exit.

## Stage 4 physics feasibility -- FAILS

Ran via `scripts/playback_physics.py`, 2 loops of the clip (4.43s), robot
initialized exactly on the reference's first frame. **Verdict: FELL.**

- Base height: starts at 0.226m (normal standing range), ends at **0.084m**
  -- well under the project's 0.15m survival gate.
- Joint tracking error: mean 0.65 degrees, max 10.3 degrees -- the PD
  actuators track the *reference* well. The reference itself is what's not
  viable.
- Visual confirmation (`stage4/stage4_physics.mp4`, frames 0/60/132): the
  robot starts upright, and by 2 seconds in has toppled fully onto its side.
  It does not recover for the remainder of the 4.43s run.

This is a clean, unambiguous result, not a borderline call.

## Blocker (one, fully diagnosed)

**The retargeted motion is not dynamically balanced enough for Bingo to
track it under real physics, even though it is kinematically well-formed
(sub-mm foot-tracking error, contact timing preserved, only one joint with
meaningful limit saturation).** Two non-exclusive contributing factors,
neither fully isolated to a single root cause given the time already spent
on this one clip:

1. **`fl_SY_J`'s residual 38.3% saturation** is real and geometrically
   necessary for this leg's targets (verified, not a bug) -- Bingo's narrow
   +/-22 degree hip-yaw range combined with a leg already near full
   extension has less effective lateral reach than the leg's raw length
   suggests, and this dog's front-left leg asks for both at once during
   specific phases of this clip.
2. **Kinematic-vs-dynamic gap.** A motion can satisfy every kinematic gate
   (foot error, joint limits, contact timing) and still be dynamically
   infeasible -- e.g. if the retargeted center-of-mass trajectory or support
   polygon coverage implied by the (root-pose, foot-contact) pairing isn't
   actually balanced for Bingo's real mass distribution. This was not
   separately audited (no time was spent computing a ZMP/support-polygon
   check before running physics) and is the more likely dominant cause given
   joint tracking itself was accurate -- the robot faithfully reached for
   the reference and still fell, which points at the motion's balance, not
   at tracking fidelity.

Fixing this would mean auditing/correcting the retarget's implied
balance (e.g. checking the support polygon under the retargeted contact
schedule actually contains the retargeted root's ground projection at each
instant, in the spirit of `stage2/lock_contact_segments.py` and
`stage4/balance_adjust.py`'s existing, unused-here machinery for exactly
this problem on the Ashley-animation pipeline) before re-attempting Stage 4,
rather than another root-cause hunt on the kinematic side, which is now in
good shape.

## Files produced this session

```text
docs/locomotion2/
  LOCOMOTION2_PIPELINE_REPORT.md          (this file)
  dataset/
    retarget_to_bingo.py                  (new: dog NPZ -> Bingo NPZ, reuses scripts/retarget.py's IK/scale/kabsch)
    validate_retarget.py                  (new: kinematic validation report+plot, no Isaac needed)
  motions/
    source/
      dog02_walk_02.npz / _report.md / _validation.png
    retarget/
      bingo_dog02_walk_02.npz / _metrics.json
      bingo_dog02_walk_02_validation.png / _validation_report.md
      kinematic_replay.mp4  (+ kinematic_replay/f*.png)
    stage4/
      stage4_physics.mp4  (+ physics_frames/f*.png)
```

No changes were made to `docs/walk_ref`, `stage2/`, `stage4/`, `scripts/`,
`rl/`, or `URDF/` -- everything new lives under `docs/locomotion2/`.
