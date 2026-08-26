# Claude handoff — Timid Stage 4 dynamic feasibility

Date: 2026-08-26  
Repository root: `/home/hassaan/Bingo/Blender`

## Exact goal

Make the corrected Timid v4 reference complete all 180 frames under
`rl/tools/track_v4_physics.py` without changing the robot, physics baseline,
joint limits, Stage 2 contact locking, Stage 3 transfer, RL, or MPC.

Intervention order required by the user:

1. local temporal retiming;
2. small root trajectory correction for balance/ZMP;
3. small local leg corrections only when required to preserve the planted paws.

Preserve Timid's recognizable pose, the contact anchors, collision/ground
validity, and motion outside the local failure as closely as possible.

## Important current-state warning

The result is **close but not robustly complete yet**.

- `stage2/out/timid_retarget.npz` currently contains the latest dynamic candidate.
- `blend_sources/Bingo_Timid_V4_Retargeted.blend` has already been baked from that
  candidate and is the only canonical working v4 Timid blend.
- `motions/timid_v4.npz` is deliberately still the previous contact-locked
  baseline. It was **not** replaced because the final baked dynamics did not pass
  robustly.
- `stage4/out/timid_v4_stage4.npz` and `.csv` are deliberately still the previous
  contact-locked baseline (fall frame 27).
- Do not copy the `/tmp` final into canonical outputs until a baked motion passes
  physics repeatably.

The unbaked candidate completes physics repeatably, but the Blender round-trip,
whose joint difference is only `3.58e-7 rad`, falls at frame 45. Treat this as a
knife-edge dynamics result requiring more stability margin, not a Stage 3 bug.

## Canonical file policy and cleanup already completed

Keep only these Timid project artifacts:

```text
blend_sources/Bingo_Timid.blend                    # Ashley source; NEVER modify
blend_sources/Bingo_Timid_V4_Retargeted.blend      # one v4 working blend
stage2/out/timid_source.npz
stage2/out/timid_contacts.npz
stage2/out/timid_retarget.npz                       # current Stage 2/4 candidate
motions/timid_v4.npz                                # promote only after validation
stage4/out/timid_v4_stage4.npz
stage4/out/timid_v4_stage4.csv
```

Redundant `_ContactLocked`, `_trusted`, `.blend1`, retarget v1-v5, contact-lock
trial NPZs, motion duplicates, and Stage 4 sweep directories were already
checksum-verified and deleted. Do not recreate suffixed routine project files.
Use `/tmp` for trials.

## Trusted baseline before this task

Canonical contact-locked motion: `motions/timid_v4.npz`

```text
180 frames @ 24 Hz
Stage 3: exact
Stage 4 fall: frame 27
joint tracking: mean 0.03003 rad, max 0.59039 rad
root orientation error: mean 93.52 deg, max 118.89 deg
roll error: mean 85.99 deg, max 110.71 deg
actual contacts: mean 0.744 / 4, 99 frames with none
actual planted-paw slip: mean 4.422 mm/frame, max 35.279 mm
reference dynamic ZMP inside: 153/180 = 85%
reference median ZMP margin: +46.3 mm
reference worst ZMP margin: -319 mm
ZMP-infeasible windows: 12-23, 25-36, 89-91
```

Stage 2 contact continuity was already fixed and must not be redone:

```text
stance support step mean, source / old v4 / corrected:
0.037 / 3.493 / 0.035 mm per frame

p95:
0.104 / 10.632 / 0.018 mm

touchdown-to-frame drift mean:
0.148 / 18.864 / 0.019 mm
```

## Diagnosis established during this task

The original failure is not primarily speed. It is an uncontrolled negative base
roll during the low-margin contact transfer. The robot is not commanded in root
space by the physics tracker; only the joint targets are commanded. Consequently,
a root correction helps physics only after leg IK converts it into a physically
different leg trajectory.

The reference has very little lateral stability margin around frames 15-23. The
actual base roll progressively diverges negative. Later, frames 27-36 transfer
support from front paws to hind paws and need the authored fore/aft momentum.
Excessive retiming removes that momentum and creates a different fall.

## Retiming experiments — tested first, then rejected

Tool: `stage4/retime_segment.py`

```text
frames 12-36, factor 1.5:
  192 frames, ZMP inside 81%, worst -245 mm

factor 2.0:
  204 frames, ZMP inside 78%, worst -218 mm
  physics fall frame 28 (baseline 27)

factor 3.0:
  228 frames, ZMP inside 73%, worst -197 mm
```

Slowing reduces acceleration/friction demand but stretches a statically poor
support transfer and removes the momentum needed for the front-to-hind transfer.
The current best candidate therefore uses no retiming. This is intentional and
supported by measurement.

## Code added/changed

### `stage4/balance_adjust.py` (new, reusable)

Applies a smooth local root lateral/roll correction and solves each exact v4 leg
with joint limits. It preserves the authored ankle trajectory and prioritizes the
real orientation-aware collision support for near-ground/contact paws. It also
regenerates physical contacts and records Stage 4 metadata.

Current best command:

```bash
cd /home/hassaan/Bingo/Blender
python3 stage4/balance_adjust.py \
  --motion motions/timid_v4.npz \
  --out /tmp/timid_dynamic_candidate.npz \
  --start 8 --end 42 --ramp 8 \
  --shift-y -0.025 --roll-deg 5.0
```

Candidate geometry/visual-change metrics:

```text
root shift: max 25 mm lateral
body roll correction: max +5 deg
correction is zero outside frames 8-42, with 8-frame smooth ramps
leg correction: mean 0.93 deg, max 17.48 deg
ankle error: mean 0.78 mm, max 12.84 mm
anchored support error: mean 0.28 mm, max 9.72 mm
minimum reference collision z: -2.83 mm
contacts: 533 foot-frames
contact/source schedule agreement: 100.0%
joint limits unchanged
```

Candidate reference audit:

```text
static stable: 166/180 = 92% (baseline 91%)
ZMP inside: 158/180 = 88% (baseline 85%)
median ZMP margin: +46.3 mm
worst ZMP margin: -229 mm (baseline -319 mm)
required mu mean/p90/max: 0.11 / 0.29 / 1.75
```

### `stage4/dynamic_audit.py`

Kept backward-compatible positional motion argument and added:

```bash
python3 stage4/dynamic_audit.py MOTION --frames 8-42
```

This prints frame-by-frame support masks, static/ZMP margins, COM, ZMP, and
support center.

### `stage2/bake_v4_motion.py`

Made idempotent. The canonical v4 blend has already had all helper/control bones
removed, so the baker now tolerates missing `ctrl_Root` and reuses saved
`*_foot_tip_local` armature metadata. This is necessary for repeatedly updating
the single canonical working `.blend`.

## Dynamic iteration results

All use the unchanged physics/controller baseline.

```text
contact-locked baseline                              fall 27
2x retiming only                                     fall 28
2x retiming + y=-25 mm                               fall 52
no retime, y=-25 mm, roll=0, early release           fall 39
no retime, y=-35 mm, roll=0                          fall 52
no retime, y=-30 mm, roll=+2.5 deg                   fall 40
no retime, y=-25 mm, roll=+5 deg, release at 50      fall 61, tips opposite way
no retime, y=-25 mm, roll=+5 deg, release at 42      completes unbaked candidate
```

The successful early-release candidate completes in two separate runs with the
same result:

```text
fallen/collapsed: no
joint tracking mean/max: 0.0196 / 0.3669 rad
root orientation error mean/max: 12.70 / 24.52 deg
max tilt: 44.3 deg
contacts mean: 3.47 / 4
frames with no contact: 1
actual planted-paw slip mean/max: 5.81 / 28.31 mm/frame
actual minimum paw z: -2.9 mm
```

However, this exact Blender-baked equivalent falls at frame 45:

```text
joint tracking mean/max: 0.0273 / 0.5560 rad
root orientation error mean/max: 85.55 / 137.66 deg
fall: frame 45
contacts mean: 0.84 / 4
frames with no contact: 103
```

Numerical equivalence check between candidate and baked NPZ:

```text
max joint difference: 3.5762787e-07 rad
max root-position difference: 4.656613e-10 m
minimum absolute quaternion dot: 0.99999994
```

This sensitivity is why the result must not yet be reported complete.

## Current `/tmp` artifacts

They may survive the handoff session but should be regenerated if absent:

```text
/tmp/timid_dynamic_candidate.npz          # unbaked best candidate; completes
/tmp/timid_dynamic_candidate_stage4.npz   # its successful physics result
/tmp/timid_dynamic_candidate_stage4.csv
/tmp/timid_v4_baked.npz                   # raw Blender round-trip
/tmp/timid_v4_final.npz                   # baked + corrected contacts; falls 45
/tmp/timid_v4_final_stage4.npz
/tmp/timid_v4_final_stage4.csv
```

## Blender/Stage 3 pipeline and exact commands

Blender executable:

```text
/home/hassaan/Bingo/local/blender-5.2.0-linux-x64/blender
```

Bake a candidate onto the canonical physical skeleton:

```bash
cp /tmp/timid_dynamic_candidate.npz stage2/out/timid_retarget.npz

/home/hassaan/Bingo/local/blender-5.2.0-linux-x64/blender \
  -b blend_sources/Bingo_Timid_V4_Retargeted.blend \
  -P stage2/bake_v4_motion.py -- \
  --motion stage2/out/timid_retarget.npz \
  --out blend_sources/Bingo_Timid_V4_Retargeted.blend
```

Export at 24 Hz:

```bash
/home/hassaan/Bingo/local/blender-5.2.0-linux-x64/blender \
  -b blend_sources/Bingo_Timid_V4_Retargeted.blend \
  -P scripts/bake_conform.py -- \
  --rig Bingo_Robot --dof 21 --hz 24 \
  --out /tmp/timid_v4_baked.npz
```

Note: the user's BlenderMCP addon can keep headless Blender alive after the NPZ is
written. If the `[[ wrote ... ]]` line has appeared, interrupt the idle Blender
process and continue; the file is complete.

Restore physical contact fields after exact-match validation:

```bash
python3 stage2/finalize_motion_contacts.py \
  --baked /tmp/timid_v4_baked.npz \
  --stage2 stage2/out/timid_retarget.npz \
  --out /tmp/timid_v4_final.npz
```

Latest exact bake result:

```text
joint mismatch 3.58e-7 rad
root mismatch 4.66e-10 m
orientation mismatch 1.37e-5 deg
contacts 533 foot-frames, source agreement 100%
```

Stage 3 full replay command:

```bash
env TERM=xterm /home/hassaan/robotics/IsaacLab/isaaclab.sh \
  -p rl/tools/replay_v4.py \
  --motion /tmp/timid_v4_final.npz --all --headless
```

Latest Stage 3 result:

```text
180/180 frames PASS
joint as-written max 0
joint after-step max 2.384e-7 rad
root position max 0 mm
root orientation max 0.03466 deg
whole-body FK max 0.0002 mm
```

Stage 4 command:

```bash
env TERM=xterm /home/hassaan/robotics/IsaacLab/isaaclab.sh \
  -p rl/tools/track_v4_physics.py \
  --motion /tmp/timid_v4_final.npz --out /tmp --headless
```

Do not enable `--vel-ff`; it is known to regress because `Kd*qdot_ref` exceeds
actuator effort authority. Do not tune PD, friction, timestep, or armature.

## Exact recommended next step

The best parameter set brackets stability but lacks numerical margin. Do a very
small local sweep around the successful curve, always validating the **baked**
NPZ, not merely the pre-bake candidate:

```text
shift-y fixed near -0.025 m
roll-deg: approximately 4.5, 5.0, 5.5
end/release frame: approximately 40, 41, 42, 43, 44
ramp: 7, 8, or 9
```

Use one `/tmp` candidate path and overwrite it. For promising parameters:

1. run `balance_adjust.py`;
2. run `dynamic_audit.py`;
3. bake/export/finalize through Blender;
4. run Stage 3 once;
5. run Stage 4 at least twice on the baked file;
6. require both runs to complete with comfortable roll/tilt margin, not merely
   one lucky completion.

The 5° correction held too long tips the robot in the positive-roll direction;
without it, the robot tips negative. The likely solution is a slightly adjusted
release curve near the current early release, not a new solver or physics change.

For parameter trials, write temporary blends under `/tmp` rather than creating
new suffixed project files. Only after a baked candidate passes repeatably:

```bash
cp /tmp/timid_v4_final.npz motions/timid_v4.npz
cp /tmp/timid_v4_final_stage4.npz stage4/out/timid_v4_stage4.npz
cp /tmp/timid_v4_final_stage4.csv stage4/out/timid_v4_stage4.csv
```

Then remove the Timid trial files in `/tmp` if desired and verify the final
canonical file list shown above.

## User's required final response format

Report only:

1. files consolidated/deleted and final canonical files;
2. Stage 4 dynamic change made;
3. before/after ZMP, roll, tracking, contacts, and fall result;
4. whether Timid completes;
5. remaining blocker if it does not.

Do not claim completion until the canonical baked `motions/timid_v4.npz` completes
repeatably under the unchanged physics tracker.
