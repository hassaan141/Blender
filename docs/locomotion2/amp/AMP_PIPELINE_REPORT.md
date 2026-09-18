# Locomotion 2 AMP Pipeline Report (strategy pivot from exact retargeting)

**Status: complete for this pass.** 6 training runs, each independently evaluated
and diagnosed; **run_06 is the current best / final deliverable** (checkpoint
`checkpoints/run_06_best_agent.pt`, ONNX export in `export/`). See "Experiments"
for the full run-by-run diagnosis trail and "Blockers" for honest follow-up items.

## Why this pivot happened

The first Locomotion 2 attempt (`docs/locomotion2/LOCOMOTION2_PIPELINE_REPORT.md`)
retargeted a single InterPet4D dog clip directly onto Bingo's joint space (solving
IK per frame so Bingo's 12 leg DOFs track the dog's pose) and carried it all the way
through kinematic validation and Isaac replay -- both passed -- only to fail Stage 4
physics feasibility outright (the robot fell, base height 0.226m -> 0.084m). The
diagnosis there: exact joint-space retargeting can produce a motion that is
kinematically well-formed (accurate foot tracking, respected joint limits, preserved
contact timing) but still dynamically infeasible for Bingo's real mass distribution,
because nothing in that pipeline ever asked whether the motion was balanced under
Bingo's own physics.

This AMP pivot sidesteps that failure mode by construction: Bingo's controller is
never asked to hit dog joint angles at all. Instead, an adversarial discriminator
compares a **morphology-tolerant style feature vector** -- root-local velocities,
gravity direction, per-leg segment directions, normalized paw kinematics, contact
state -- computed independently from the dog reference and from Bingo's own
simulated state, each normalized by its OWN body scale. A separate velocity-command
task reward (identical in spirit to the project's existing Task 1 velocity
controller) gives Bingo a reason to actually translate. Because Bingo's own PPO
policy is what generates the "Bingo-shaped" transitions the discriminator sees, the
resulting gait is automatically filtered through Bingo's own physics and morphology
-- there is no separate "is this dynamically feasible" question left over, unlike
the exact-retarget approach.

## Dataset: Lifelike Dog (BVH mocap)

Source: `dataset/lifelike_dog/raw_bvh/raw_bvh_data/` (33 clips, 120 fps, a
3ds-Max-Biped-style rig with human bone names retargeted onto a quadruped mesh).
Full inspection: `docs/locomotion2/amp/DATASET_REPORT.md`.

**Important finding, numerically verified (not trusted from filenames):** the
`dog_idle_*.bvh` clips are NOT standing still -- they show substantial locomotion
(mean speed 0.97-1.53 m/s, comparable to or faster than the `quad_walk` clips).
Genuine stationary moments exist only as short (1-2s) windows inside otherwise-moving
clips. This is the same lesson the InterPet4D work already learned the hard way
("never trust a clip's name or an automated aggregate metric without checking the
actual motion") showing up again in a completely different dataset.

**Expert set used for this first controller:**
- **idle/stand**: 2 genuine stationary windows (`dog_idle_002.bvh` [0.7,2.0]s,
  `dog_quad_walk_001.bvh` [0.1,1.3]s) -- thin coverage, so standing behavior is also
  backstopped by the env's own upright/zero-command task reward.
- **forward walk**: heading-stable (rolling circular std < 20 deg), straight
  (straightness > 0.85), >= 2s windows scanned out of the two long (110s/151s)
  `dog_quad_walk_001/002.bvh` captures -- NOT the full files, which wander all over
  the mocap volume and repeatedly turn.
- **excluded**: `*_run*`, `*_jump*`, `play_*`, `hit_*` (per brief), and
  `*_star_walk*`/`*_zig_walk*`/`*_back*` (turning/backward, out of scope for a
  straight-forward-only first controller), and `*_walkrun*` (no sustained walk-paced
  plateau long enough to use without also teaching a running gait).
- Left/right mirrored copies of every window are added (Householder reflection of
  raw joint positions/rotations about the window's own mean-heading plane, then
  swap L<->R before recomputing features -- not hand-flipped signs on an
  already-differentiated pseudovector).

## AMP feature schema (61-dim), per side, root-local, own-scale-normalized

See `docs/locomotion2/amp/extract_dog_amp_features.py`'s module docstring for the
authoritative layout. Summary:

| slice | content | normalization |
|---|---|---|
| [0:3] | root-local linear velocity | / leg_scale |
| [3:6] | root-local angular velocity | raw rad/s |
| [6:9] | gravity vector, root-local frame | unit-scale by construction |
| [9:33] | 4 legs x 2 segments (hip->mid, mid->paw), unit direction vectors | direction-only, scale-free |
| [33:45] | 4 legs x paw position rel. root, root-local frame | / leg_scale |
| [45:57] | 4 legs x paw velocity rel. root, root-local frame | / leg_scale |
| [57:61] | 4 legs x ground-contact state {0,1} | -- |

**Two bugs found and fixed during construction, each with numeric evidence:**

1. **Wrong normalization constant.** Initially normalized the DOG's positions by
   BINGO's leg scale (0.2036m) instead of the dog's own (~0.58m, p95 hip->paw
   distance) -- produced a `paw_rel` value of 3.4 (should be O(1)), caught by
   inspecting the single largest per-frame feature jump in the buffer (11.2, later
   traced to this channel) rather than trusting the aggregate NaN/jump check alone.
   Fixed: each side always normalizes by its OWN leg scale (dog: 0.58m measured;
   Bingo: 0.2036m, `Urdf.leg_reach()` on the v4 physics URDF).
2. **Timestep/speed mismatch (the more consequential one).** The dog's native walk
   speed (~1.0-1.5 m/s) is 4-6x Bingo's 0.15-0.30 m/s task band -- confirmed this
   persists even in leg-length-normalized units (dog ~5-6 leg-lengths/s vs. Bingo's
   target ~0.7-1.5 leg-lengths/s), i.e. leg-scale normalization alone does not fix
   it; the dog genuinely steps faster relative to its own leg length. Encoding raw
   velocity would force the discriminator to reward Bingo for moving at dog speed,
   fighting the task reward directly -- exactly the failure the prior (July-2026,
   rev_3-era) AMP experiment's own tuning notes flag ("v15 task 1.3 + 2:1 trot bias
   REGRESSED"). Fixed by **re-timing** each walk window (stretching its time axis by
   a per-window factor so its mean speed lands at 0.22 m/s, the middle of Bingo's
   target band) BEFORE resampling to the 30 Hz control rate -- this divides every
   velocity-like channel by the stretch factor while leaving purely geometric
   channels (segment directions, normalized paw-rel positions, contact-vs-phase
   pattern) untouched, i.e. same gait SHAPE, slower playback.

Expert transitions are (feature_t, feature_t+1) pairs resampled to exactly 30 Hz
(matching the env's 120Hz-physics/decimation-4 control rate) via linear
interpolation (positions) and quaternion slerp (rotations) -- built at the mocap's
native rate first and NOT simply subsampled, so the pairs reflect true finite
differences at the sim's own dt rather than a rate mismatch that would let the
discriminator trivially separate real/fake by transition magnitude alone.

Validated: 0 NaNs, max per-frame feature jump 1.52 (after both fixes, down from
11.2), contact duty ~0.30-0.40 per paw (a brisk trot-like duty factor -- see
"blockers/notes" below), 40 windows / 36224 frames / 36184 transition pairs total
after mirroring. Visual debug plot (`expert_buffer_debug.png`) confirms clean
periodic forward velocity, regular alternating contact states, and clear paw-height
swing/stance cycles on a representative walk window.

## Architecture

Built on `BINGO_V4_CFG` (validated Stage-4 physics, unmodified) rather than the
prior (July-2026) AMP experiment's rev_3 asset -- new package
`rl/bingo_rl/bingo_rl/locomotion2_amp/`, registered as
`Bingo-Locomotion2-AMP-Direct-v0` (+ `-Play-v0` for eval). Reused directly:
- The prior AMP experiment's skrl AMP `DirectRLEnv` + discriminator plumbing pattern
  (`rl/bingo_rl/bingo_rl/amp/bingo_amp_env.py`, itself proven across 21 training runs
  in July 2026) -- command-conditioned obs, 2-frame `amp_obs` stacking,
  `collect_reference_motions` hook, skrl `AMP` agent class and its tuned
  hyperparameters (network sizes, discriminator regularization) as a starting point.
- Task 1's validated v4 stance (`STAND_SOLVED`, solved against real collision hulls)
  and per-joint-type action scales (`ACTION_SCALE`) from
  `rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_env_cfg.py`.
- `scripts/retarget_to_bingo.py`'s `Urdf.leg_reach()` for Bingo's own leg-scale
  constant.

New: the 61-dim morphology-tolerant feature function itself (computed independently
in torch on Bingo's simulated state and in numpy on the dog reference -- no shared
joint-space mapping), the BVH parser/preprocessing pipeline, and the re-timing fix.

Policy: `[vx, vy, yaw_rate]` (3, vy/yaw pinned to 0 for this first controller) +
61 style features -> PPO policy (GaussianMixin, 1024-512 MLP) -> 12 leg joint
targets (`default_pose + scale*action`) -> v4 physics. Reward = velocity-tracking
task term (dominant) + small regularizers (upright, action-rate, joint-accel,
torque) + the AMP style reward (computed and logged separately by skrl's AMP agent,
not hand-mixed into the task reward) -- per the brief, handcrafted terms are kept
deliberately light so AMP supplies the gait shape.

Known v1 simplification: resets always use the default (STAND_SOLVED) pose with
small xy/yaw noise, not reference-state-initialization (RSI) sampled from the dog
motion -- there's no shared joint space to sample an RSI pose FROM. This may slow
exploration into varied gait phases; flagged as a candidate follow-up, not chased
in this pass.

## Experiments

Full table: `EXPERIMENTS.csv`. Each run: 2048 envs, 3000 iterations (48000 timesteps),
skrl AMP, ~47-49 min wall clock on a single GPU.

**run_01 (baseline)**: excellent velocity tracking (RMSE 0.013-0.019 m/s across
0.10-0.30 m/s, 100% survival including a 60s sustained hold) but two real defects,
both confirmed with hard evidence rather than assumed from reward curves alone:
1. **Standing falls 100% of the time.** Root cause found by inspecting the expert
   buffer composition: idle transitions were <0.5% of it (146 of 45090), so the AMP
   discriminator had essentially never seen "expert standing" and kept pulling the
   policy toward walk-like motion even at cmd_vx=0, fighting the task reward at
   exactly the operating point with the least room for error.
2. **Degenerate gait even while walking well.** `fl`/`br` contact duty was exactly
   0.0 for the whole segment (vs `fr` 0.67 / `bl` 0.33) -- confirmed with an
   instrumented joint-trace diagnostic (`rl/bingo_rl/scripts/diagnose_gait_cycle.py`,
   written specifically because rendered-video inspection was ambiguous -- see
   below), not just the aggregate contact-sensor stat: all 12 leg joints showed only
   0.01-0.13 rad (~1-8 deg) peak-to-peak swing and paw lift was only 2-6mm (the dog
   reference shows ~40-80mm paw-height p2p in the debug plot). The robot reaches its
   commanded speed mostly through high-frequency joint jitter and foot slip
   (~200-250mm/s) on a near-static crouched pose, not a real stride.

**Video-inspection pitfall worth recording**: two side-by-side video frames several
tenths of a second apart looked identical at first glance, which read as "the video
capture is broken." A pixel-diff check across a finer time series showed real,
substantial frame-to-frame change (~23% of pixels) -- but that change turned out to
be the follow-camera panning with genuine root translation, not leg articulation;
the actual limb pose barely changes. Lesson for this workstream specifically: video
alone is not a reliable falsifier OR verifier of stepping quality at this
scale/frame rate -- the instrumented joint/paw-height trace (with FFT for a
dominant-frequency sanity check) is the deciding instrument, and was written for
exactly this reason.

**run_02 (idle oversample fix, single variable vs. run_01)**: expert buffer's idle
transitions boosted from 146 to 9052 of 45090 (20%, matching the training-time
`stand_prob`). Result: **standing is fully fixed** (6/6 eval gates pass, was 4/6;
vx=0 survival 100%, roll/pitch RMS 0.4/0.1 deg). Walking tracking stays excellent
(RMSE 0.005-0.011 for vx>=0.2). Contact-duty asymmetry improved (0.4/0.4/0.2/0.2 vs.
run_01's 0.67/0.33/0/0) but the SAME joint-trace diagnostic proves the underlying
gait defect persists nearly unchanged (paw lift 4-8mm, joint swings 0.03-0.13 rad) --
the idle fix and the gait-shape fix are, as expected, two independent problems.

**run_03 (+ anti-degenerate-gait bound, on top of run_02's buffer)**: adds the
`max_air_time`/`max_contact_time` = 0.35s bounded penalty (weight -0.3) that
`rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_mdp.py` already documents as the
fix for two PREVIOUSLY MEASURED failure modes on this exact robot ("v3 held one leg
permanently in the air and dragged on three", "v6 parked the front-left foot 85% of
the time as a static prop") -- i.e. this run's defect is not a new problem, it is a
known Bingo failure class recurring under a new (AMP-based) reward, and the fix is
reused rather than invented. Result: 6/6 eval gates still pass, but the instrumented
diagnostic shows only a MARGINAL amplitude improvement (paw lift 3-9mm vs run_02's
4-8mm). Root cause of why this specific fix under-delivered: the FFT dominant
frequency of the degenerate gait is 6-12 Hz, i.e. each foot's phase only lasts
~0.04-0.08s -- nowhere near the 0.35s air/contact-time threshold, so the bound
essentially never fires. Right diagnosis (a documented Bingo failure class), wrong
lever for THIS failure's time scale.

**run_04 (style_reward_scale 1.0 -> 2.5, on top of run_03's fixes)**: hypothesis
REJECTED -- paw lift went to 2-4mm, slightly worse than run_03. But it surfaced the
real diagnosis: every run so far (01-04) showed the SAME discriminator score gap
(expert ~+7.5, policy ~-7 to -8, separation ~15) essentially saturated since early
training -- classic AMP discriminator-dominance. A confidently-separating
discriminator gives near-zero style-reward *gradient* to the policy regardless of
`style_reward_scale` (scaling ~0 by 2.5x is still ~0), which is exactly why the
weight bump did nothing.

**run_05 (discriminator_gradient_penalty_scale 5.0 -> 15.0, targeting the
dominance diagnosis)**: also REJECTED -- discriminator loss floor was unchanged
(~0.03, identical to every prior run) even at 3x the penalty, and paw lift stayed
at 3.6-6.9mm. This ruled out "insufficient discriminator regularization strength"
as the mechanism, not just as an untried idea.

**run_06 (initial_log_std -2.9 -> -1.0, revert gradient penalty to 5.0) --
BREAKTHROUGH.** Re-examined the agent config and found `fixed_log_std: True` with
`initial_log_std: -2.9` (action std ~0.055 in normalized units, ~0.008 rad of
actual per-step joint noise) carried over unchanged from the prior rev_3-era
tuning, combined with `entropy_loss_scale: 0.0` -- i.e. the policy had essentially
no mechanism to ever explore into a larger-amplitude stepping strategy; it could
only get there via gradient drift from a near-zero-swing initialization, and the
already-established weak/saturated style gradient couldn't supply that. Raising
`initial_log_std` to -1.0 (~0.37 std, ~6.7x more exploration noise) while keeping
`fixed_log_std: True` (so it doesn't collapse back down mid-run) produced a
qualitatively different, genuinely better gait:

| metric (cmd_vx=0.25) | run_02 (best "safe" run) | run_06 (current best) | dog reference |
|---|---:|---:|---:|
| leg joint ptp swing | 0.02-0.13 rad | 0.005-0.75 rad (SP/knee up to 0.4-0.75) | tens of degrees, comparable order |
| paw-lift ptp | 4-8 mm | 18-121 mm (fr=121, others 18-47) | ~40-80mm (expert_buffer_debug.png) |
| dominant gait frequency | 6-12 Hz (nonsensical) | ~3.7 Hz (plausible walking cadence) | -- |
| vx tracking RMSE | 0.007-0.011 | 0.037-0.055 | -- |
| torque saturation | ~5-8% | ~24-27% | -- |
| joint-limit contact | 0.0% | 16.7% | -- |
| roll RMS (deg) | 0.3-0.5 | 3.0 | -- |
| stand survival | 100% | 100% | -- |
| 60s sustained survival | 100% | 100% | -- |

Visually confirmed via a cropped frame-by-frame montage (not just the aggregate
numbers): genuine cyclic leg articulation across the gait cycle, not a frozen pose
(see the "video-inspection pitfall" note above for why this check matters --
squinting at full-scene frames had been misleading twice already this session).
**run_06 is the current best / final deliverable for this pass.** The added torque
saturation and reduced tracking precision are the real cost of a larger, more
natural stride and are not alarming on their own (roll RMS is still only 3 degrees,
survival is unaffected including the 60s sustained hold) -- but they are the
natural next thing to tighten in a follow-up pass (see Blockers).

## Evaluation

Battery: `rl/bingo_rl/scripts/eval_locomotion2_amp.py`, adapted from
`rl/tools/eval_velocity.py`'s methodology (SegmentRecorder pattern, held-command
segments, pass/fail gates) to this env's skrl/DirectRLEnv stack. Segments: stand +
vx in {0.10, 0.20, 0.25, 0.30} x 5s each, + 60s sustained @ 0.25 m/s. One real bug
fixed during first use: `terminated`/`truncated` from the skrl-wrapped env are
shaped `(N,1)`, which silently made downstream boolean masks 3-D and crashed numpy
boolean indexing -- fixed by an explicit `.reshape(-1)`.

Videos: `docs/locomotion2/amp/loop_runs/run_0{1..6}/eval_video/rl-video-step-0.mp4`
(single file per run covering the whole battery back-to-back: stand, then each vx
segment in order, then the 60s sustained hold -- not 3 separate files, but all
requested behaviors are present with known timestamps: stand [0,5)s, vx=0.10
[5,10)s, vx=0.20 [10,15)s, vx=0.25 [15,20)s, vx=0.30 [20,25)s, 60s sustained
[25,85)s). **run_06's video is the one to watch.**

Instrumented gait diagnostic (the more reliable check, see above):
`docs/locomotion2/amp/loop_runs/run_0{1..6}/gait_trace.npz` +
`rl/bingo_rl/scripts/diagnose_gait_cycle.py`.

ONNX export: `docs/locomotion2/amp/export/bingo_locomotion2_amp.onnx` +
`manifest.json`, exported from run_06. Single-step PyTorch-vs-ONNX parity: max abs
diff 1.9e-6 (well under the 1e-4 gate) -- PASS. A 96-step closed-loop rollout
comparison shows 0.23 m/s max divergence between the PyTorch- and ONNX-driven
trajectories; per the brief's own instruction, this is NOT attributed to ONNX
correctness (single-step parity is essentially exact) -- it is normal chaotic
divergence in a contact-rich closed-loop nonlinear system, where a ~1e-6-scale
per-step difference compounds over 96 steps of feedback.

### Comparison against Locomotion 1 (`docs/walk_ref/quality_runs/run_05/`)

Locomotion 1 is a mature result (22 tuning experiments across 3 campaigns, per its
own handoff doc) built by residual correction on a hand-authored reference clip, so
this is a comparison of a first AMP pass against a heavily-refined baseline, not two
equally-mature systems:

| metric | Locomotion 1 (run_05, frozen best) | Locomotion 2 AMP (run_06) |
|---|---:|---:|
| mean vx | 0.193 m/s | 0.243 m/s (cmd 0.25) |
| joint accel RMS | 11.0 rad/s^2 | 40-49 rad/s^2 |
| body-height std (bobbing) | 1.5 mm | 5-6 mm |
| roll RMS | 5.5 deg | 3.0 deg |
| pitch RMS | 2.0 deg | 0.8-0.9 deg (steady-state) |
| survival (60s @ 0.25) | 100% | 100% |
| reference needed at runtime | yes (cyclic reference clip + phase) | no (pure velocity command) |
| style source | hand-tuned reward shaping | dog mocap (AMP) |

Locomotion 1 remains smoother (4x lower joint acceleration, ~4x less bobbing) --
expected, given its extensive refinement history. Locomotion 2 AMP is slightly
better on roll/pitch stability and offers a categorically different capability
Locomotion 1 doesn't have: a pure velocity-command interface with no
hand-authored reference clip, learned from real animal mocap. **Whether the
gait is "actually visually better" than Locomotion 1: no, not yet** -- Locomotion
1 is the smoother, more polished walk. What run_06 achieves for the first time is
a genuinely non-degenerate, dog-style-prior-driven gait that responds to velocity
commands; closing the smoothness gap to Locomotion 1 is follow-up work, not
something this pass claims.

## Blockers / open questions

None fundamental -- the workstream reached a genuinely working, non-degenerate,
command-driven gait (run_06) after diagnosing and fixing three independent real
defects (idle-buffer imbalance, and the exploration-starvation root cause behind
the degenerate-gait defect; the gait-bound and discriminator-regularization
attempts in between were valid hypotheses that were correctly ruled out with
evidence, not abandoned blindly). Remaining, non-blocking follow-up items:

1. **Smoothness gap vs. Locomotion 1**: run_06's joint-acceleration RMS (40-49
   rad/s^2) and torque saturation (~25%) are notably higher than the safe-but-
   degenerate run_02 (and than Locomotion 1). Now that real exploration is
   unlocked, the natural next bounded experiment is re-tuning the action-rate/
   joint-acceleration/torque regularizer WEIGHTS (already present in
   `bingo_dog_amp_env_cfg.py`, currently set low deliberately) upward, now that
   there's an actual stride worth smoothing -- previously these weights were
   irrelevant because the gait had no real amplitude to smooth.
2. **vx tracking RMSE grew** (0.037-0.055 vs. 0.007-0.011) with the larger stride.
   Still passes every gate (>60% of commanded speed), but tightening it (e.g. a
   small increase to `tracking_sigma_vx`'s inverse weighting, or more training
   iterations for the same config to let tracking catch up now that exploration
   found the right gait family) is reasonable follow-up, not a blocker.
3. **No RSI (reference-state initialization)** -- resets always start from the
   default stance; there's no shared joint space to sample a mid-gait pose from
   the dog reference. Noted as a v1 simplification in the Architecture section;
   may help exploration/robustness further if added later.
4. **Turning/lateral/backward motion** is entirely out of scope for this pass
   (per the brief) and untested -- `cmd_vy_range`/`cmd_yaw_range` are pinned to
   (0,0) throughout.
