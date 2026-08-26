# Codex handoff — Bingo Cheeky Stage 2 (spatial retarget) finish-up

You are picking up an in-progress task with **no prior context**. This document is
self-contained. Read it fully before running anything.

---

## 0. Current status / next action

Stage 2 (kinematic retarget of Ashley's "Cheeky" animation onto the exact Bingo v4
robot skeleton) was repaired after its first visual validation failed. The corrected
blend, motion NPZ, report, and synchronized comparison video were regenerated on
2026-08-25. Do not use the earlier metrics or frames.

The baked file was reopened in Blender 5.2 and round-tripped through
`scripts/bake_conform.py`. All 21 joint angles match the solved motion within
2.69e-7 rad, root position matches exactly, and root orientation matches within
1.79e-7. The corrected evaluator reports 26.3 mm mean foot-trajectory error, 2.6 mm
planted-foot slide, 100% contact preservation, and 0.0 mm ground penetration. See
`STAGE2_FAILURE_DIAGNOSIS.md` and `CHEEKY_RETARGET_REPORT.md` for root causes/fixes.

**Do NOT** touch the physics/RL side, do NOT modify the physical v4 skeleton, do NOT
re-solve unless a metric below is clearly broken. This stage is kinematic only.

---

## 1. Project context

Bingo is a ~2.5 kg quadruped robot. An animator (Ashley) authored expressive
performances (Cheeky, DeadPan, Timid, …) on a stylized Blender rig. Goal of Stage 2:
reproduce Ashley's **Cheeky** performance on the **exact v4 physical robot skeleton**
(21 actuated joints) purely kinematically, so it can later be baked to Isaac Sim.

- 21 joints = 12 leg (4 legs × SY/SP/knee) + 3 head + 2 tail + 4 ear.
- The v4 URDF is the single source of truth for the physical skeleton.
- The retarget goes through **evaluated world-space semantic motion** (not by copying
  Ashley's joint angles — her rig and the robot have different morphology/pivots).

---

## 2. Environment (exact paths — this machine)

- Working dir: `/home/hassaan/Bingo/Blender`
- Blender (portable, **5.2.0**): `~/Bingo/local/blender-5.2.0-linux-x64/blender`
  - Blender's bundled python has **numpy but NO scipy**.
- System python3: **has scipy 1.15 + numpy 2.2** — the solver runs here.
- v4 URDF: `"URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"`
  (note the **space** in the folder name — always quote it).
- Source animation: `blend_sources/Bingo_Cheeky.blend`
  (its rig `Bingo_Rig` is **linked** from `blend_sources/BingoRig_Latest.blend`).
- Target rig: `blend_sources/Bingo_V4_AnimatorRig.blend` (armature object
  `Bingo_Robot`, 40 bones: physical skeleton + animator controls).
- Baked result: `blend_sources/Bingo_Cheeky_V4_Retargeted.blend`.

### Blender gotchas already handled (don't re-discover these)
- **`os._exit(0)` skips stdout/file flush.** Every script calls
  `sys.stdout.flush()` before `os._exit(0)`. If you add prints and see no output,
  this is why.
- **Blender 5.2 layered Action API**: `action.fcurves` no longer exists. Iterate
  `action.layers[].strips[].channelbags[].fcurves` (see git history if needed).
- **Blender 5.x render output**: set `sc.render.image_settings.media_type='IMAGE'`
  before `file_format='PNG'` (Cheeky's file defaults to VIDEO).
- **Headless rendering**: `bpy.ops.render.opengl` FAILS (no GL context). EEVEE via
  `bpy.ops.render.render(write_still=True)` WORKS headless. RTX/Cycles raytracing is
  intractable on this GPU — use EEVEE only.

---

## 3. The pipeline (all under `stage2/`) — what each script does & how to run

Run order is 1→6. Scripts 1 and 5 run **in Blender**; 2,3,4,6 run **in system
python** (they need scipy or just numpy and read/write `.npz`).

`stage2/v4_kinematics.py` — importable numpy module: exact v4 FK + joint limits from
the URDF. No bpy, no scipy. Used by the solver and evaluator. DOF order (must stay
identical everywhere, matches `scripts/bake_conform.py`):
```
fl_SY_J fl_SP_J fl_knee  fr_SY_J fr_SP_J fr_knee  bl_SY_J bl_SP_J bl_knee
br_SY_J br_SP_J br_knee  head_pitch_joint head_yaw head_roll  tail_pitch tail_yaw
l_ear_pitch l_ear_roll  r_ear_pitch r_ear_roll
```

**1. Extract source motion** (Blender → npz):
```
~/Bingo/local/blender-5.2.0-linux-x64/blender -b blend_sources/Bingo_Cheeky.blend \
  -P stage2/extract_source_motion.py -- --out stage2/out/cheeky_source_keypoints.npz
```
Samples evaluated world-space `def_*` deform-bone positions/quats per frame. Ashley's
rig is biped-style: **front legs = Arm/ForeArm/Hand, back legs = Leg/Shin/Foot**,
`def_Pelvis` body, `def_Head`, `def_Tail.001`, `def_Ear.L/R`. 180 frames @ 24 fps.

**2. Detect contacts** (system python → npz):
```
python3 stage2/detect_contacts.py --keypoints stage2/out/cheeky_source_keypoints.npz \
  --out stage2/out/cheeky_contacts.npz
```
Per-paw PLANTED/SWING from world speed + height with hysteresis.

**3. Solve spatial retarget** (system python, scipy → npz):
```
python3 stage2/solve_spatial_retarget.py \
  --keypoints stage2/out/cheeky_source_keypoints.npz \
  --contacts  stage2/out/cheeky_contacts.npz \
  --urdf "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \
  --out stage2/out/cheeky_v4_retarget.npz
```
Per-leg limb-length scaling → per-frame root pose → contact-anchored per-leg IK
(`scipy.optimize.least_squares`, bounds = exact URDF limits) → head/tail/ear rotation
chains → contact-consistent root de-slip → rate-limit smoothing. Writes
`root_pos`, `root_quat`, `dof_positions[T,21]`, diagnostics.

**4. Bake onto the physical v4 skeleton** (Blender → .blend):
```
~/Bingo/local/blender-5.2.0-linux-x64/blender -b blend_sources/Bingo_V4_AnimatorRig.blend \
  -P stage2/bake_v4_motion.py -- --motion stage2/out/cheeky_v4_retarget.npz \
  --out blend_sources/Bingo_Cheeky_V4_Retargeted.blend
```
Keyframes the **physical** `root` bone (loc+quat) and the 21 joints (rotation about
each bone's local Z). Mutes the head/tail/ear `COPY_ROTATION` constraints so the
physical joints carry the expression (otherwise the at-rest controls zero them).

**5. Render comparison** (Blender → PNG frames) — optional presentation output:
```
~/Bingo/local/blender-5.2.0-linux-x64/blender -b blend_sources/Bingo_Cheeky_V4_Retargeted.blend \
  -P stage2/render_compare.py -- --target robot --outdir stage2/out/frames_robot --every 2
~/Bingo/local/blender-5.2.0-linux-x64/blender -b blend_sources/Bingo_Cheeky.blend \
  -P stage2/render_compare.py -- --target ashley --outdir stage2/out/frames_ashley --every 2
```

**6. Evaluate + report** (system python → md):
```
python3 stage2/evaluate_retarget.py \
  --keypoints stage2/out/cheeky_source_keypoints.npz --contacts stage2/out/cheeky_contacts.npz \
  --retarget stage2/out/cheeky_v4_retarget.npz \
  --urdf "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \
  --report CHEEKY_RETARGET_REPORT.md
```

---

## 4. Current verified state (already done — trust these, re-run only if needed)

The full pipeline (steps 1–4, 6) has been run and validated:

- **Round-trip is exact**: after baking, re-reading the physical joints with
  `scripts/bake_conform.py --rig Bingo_Robot --dof 21` returns the solved angles to
  **0.00000 rad** for all 21 joints, including all 9 expression channels.
- **Global motion is faithful**: yaw matches Ashley frame-by-frame
  (+10/+46/+63/+89/+61/+12° at f31/61/91/121/151/180 = robot exactly); horizontal
  travel 1.382 m (robot) vs 1.378 m (Ashley, scaled).
- Planted-foot slide **2.6 mm/frame during stance**, contact schedule **100% preserved**, ground
  penetration **0 mm**.
- Swing-phase foot direction is positively correlated with Ashley; body-crouch
  vertical amplitude ~preserved (119 vs 104 mm).
- Poses transfer on stills (crouch at f81, play-bow at f121 both clearly match).

### Two non-obvious bugs that were already found and fixed (do not reintroduce)
1. **Ashley's rig is mirrored / left-handed** (forward +Y, left +X vs robot forward
   +X, left +Y). The solver reflects the source across YZ (`F = diag(-1,1,1)`) at
   load. Without this, feet land wrong and direct/mirror Kabsch residuals are equal.
2. **The 4 hips are nearly coplanar** → Kabsch on them is degenerate about the
   vertical and can flip the up-axis (feet then move backwards, swing cosine
   negative). Root **orientation is taken from Ashley's pelvis quaternion**, not from
   Kabsch on hips. Only the frame-0 hip geometry is used to set the source→robot
   axis alignment `A`.

### Known, accepted limitations (record, don't try to "fix" by shrinking motion)
- `SY` (shoulder-yaw/abduction) limit is only **±0.42 rad (~24°)**, far less than
  Ashley's lateral paw splay, so the robot's stance is **wider/splayed** than Ashley
  and the big front-paw raises can't be matched 1:1. Foot IK residual ~26 mm mean is
  mostly these genuinely-unreachable expressive poses. This is documented in
  `CHEEKY_RETARGET_REPORT.md` §5, not hidden. The animation amplitude is **not**
  globally reduced.

---

## 5. Optional comparison-video workflow (not a blocker)

The previous session had just switched `stage2/render_compare.py` from a **static**
wide camera (which made the moving/turning dog wander around frame and look
"sprawled") to a **tracking camera** that follows the character's root each frame and
keeps it centered and same-size. **That new code is written but was never run.**

### Step A — test-render a few frames first (cheap, catches bugs)
```
cd /home/hassaan/Bingo/Blender
B=~/Bingo/local/blender-5.2.0-linux-x64/blender
$B -b blend_sources/Bingo_Cheeky_V4_Retargeted.blend -P stage2/render_compare.py -- \
    --target robot  --outdir stage2/out/tr_robot  --every 30
$B -b blend_sources/Bingo_Cheeky.blend             -P stage2/render_compare.py -- \
    --target ashley --outdir stage2/out/tr_ashley --every 30
```
Then eyeball a couple of paired frames (e.g. `hstack` ashley|robot with ffmpeg and
open the PNG). Confirm: both characters are **centered, same size, upright, same
relative 3/4 view, feet on the ground**. `render_compare.py` uses a `TRACK_TO`
constraint aiming at an empty placed at the root each frame; the camera is placed at
`root + up*0.35*csize + view*2.6*csize`. If framing is off, tune `dist` (currently
`2.6 * csize`), the `view` vector, or `up_off` in `main()`. `csize` is the median
per-frame body diagonal (travel excluded).

### Step B — full render (every 2 frames = 90 each). Takes several minutes; run in
background or with a long timeout. EEVEE, ~540p.
```
$B -b blend_sources/Bingo_Cheeky_V4_Retargeted.blend -P stage2/render_compare.py -- \
    --target robot  --outdir stage2/out/frames_robot  --every 2
$B -b blend_sources/Bingo_Cheeky.blend               -P stage2/render_compare.py -- \
    --target ashley --outdir stage2/out/frames_ashley --every 2
```

### Step C — stitch labeled side-by-side mp4 (12 fps = real-time for every-2 @ 24fps)
```
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
ffmpeg -y -framerate 12 -pattern_type glob -i 'stage2/out/frames_ashley/f*.png' \
       -framerate 12 -pattern_type glob -i 'stage2/out/frames_robot/f*.png' \
  -filter_complex "\
[0:v]drawtext=fontfile=$FONT:text='Ashley Cheeky (source)':x=(w-tw)/2:y=12:fontsize=22:fontcolor=black:box=1:boxcolor=white@0.6[a];\
[1:v]drawtext=fontfile=$FONT:text='v4 retargeted (baked)':x=(w-tw)/2:y=12:fontsize=22:fontcolor=black:box=1:boxcolor=white@0.6[b];\
[a][b]hstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 stage2/out/cheeky_compare.mp4
```
(The `pad=ceil(iw/2)*2...` is required — raw hstack height 405 is odd and libx264
rejects it.)

### Step D — regenerate the report (harmless, keeps numbers current)
Run step 6 above.

### Step E — hand back
Write 3–5 lines: do the poses/turns/crouches/head/tail/ears visibly match Cheeky?
Note the known SY-splay caveat. The human makes the final visual call. **Do not
proceed to Isaac** — that's a later stage.

---

## 6. Deliverables (final expected set)

- `blend_sources/Bingo_Cheeky_V4_Retargeted.blend`  ✅ exists
- `stage2/out/cheeky_source_keypoints.npz`  ✅
- `stage2/out/cheeky_contacts.npz`  ✅
- `stage2/out/cheeky_v4_retarget.npz`  ✅
- `CHEEKY_RETARGET_REPORT.md`  ✅ (regen in step D)
- `stage2/out/cheeky_compare.mp4`  (optional; existing render may use old framing)
- `stage2/*.py` (the 6 scripts)  ✅

## 7. If you must re-solve (only if a metric is broken)
Re-run steps 3→4→6. Never edit `scripts/build_rig.py` /
`scripts/add_animator_controls.py` (they define the verified physical skeleton).
`scripts/bake_conform.py` is the trusted round-trip reader:
```
$B -b blend_sources/Bingo_Cheeky_V4_Retargeted.blend -P scripts/bake_conform.py -- \
    --rig Bingo_Robot --dof 21 --hz 24 --out /tmp/check.npz
```
Compare its `dof_positions` to `cheeky_v4_retarget.npz` — should differ by ~0.
