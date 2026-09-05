# Bingo Expressive Animation Pipeline — Current Technical Memory

Last updated: **2026-09-05**. This replaces the obsolete rev-3/17-joint and old Isaac setup notes.

## Current objective and stage status

Transfer Ashley's expressive Bingo animations to the exact Bingo v4 robot without changing its physical skeleton.

| Stage | Purpose | Current status |
|---|---|---|
| 1 | Exact Blender v4 rig | **Validated** |
| 2 | Ashley → v4 spatial retargeting | **Working; validate each clip** |
| 3 | Exact, no-physics Isaac replay | **Validated** |
| 4 | Full-physics feasibility | **Infrastructure validated; clip-specific** |
| 5 | Residual RL | **Implemented/trained for Timid only; experimental** |

Do not advance a clip when its previous stage fails. Stage 4 passing means more than surviving: the major root translation, hops, yaw and contacts must still resemble Stage 3.

## Exact v4 robot

- Blender rig: `blend_sources/Bingo_V4_AnimatorRig.blend`
- Target armature: `Bingo_Robot`
- Kinematic URDF: `URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf`
- Physics URDF: `URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf`
- Physics configuration: `rl/bingo_rl/bingo_rl/bingo_v4.py`
- 21 actuated joints: 12 leg, 3 head, 2 tail and 4 ear joints.

Canonical joint order:

```text
fl_SY_J fl_SP_J fl_knee
fr_SY_J fr_SP_J fr_knee
bl_SY_J bl_SP_J bl_knee
br_SY_J br_SP_J br_knee
head_pitch_joint head_yaw head_roll
tail_pitch tail_yaw
l_ear_pitch l_ear_roll r_ear_pitch r_ear_roll
```

The v4 legs have `SY → SP → knee`, with **no ankle or paw joint**. The paw is rigid collision geometry attached to the shank/knee body. Never add a paw joint, alter pivots/axes/lengths, loosen limits, or shrink an animation to hide infeasibility.

## Stage 2 — spatial retargeting

Main files:

```text
stage2/extract_source_motion.py
stage2/detect_contacts.py
stage2/solve_spatial_retarget.py
stage2/v4_kinematics.py
stage2/bake_v4_motion.py
stage2/evaluate_retarget.py
stage2/finalize_motion_contacts.py
```

Correct leg mapping:

```text
Ashley: SY → SP → knee → ankle/paw-bone HEAD → toe/contact
v4:     SY → SP → knee → shank tip             → rigid paw support
```

Important fixes already implemented:

- Construct the source leg target from Ashley's **SP pivot**, then apply the real v4 `SY → SP` offset once.
- Use `def_Hand.head` / `def_Foot.head` as the source ankle target, not their tails.
- Track Ashley's toe separately as the lower-priority rigid-paw contact target.
- Leg IK prioritizes ankle, knee shape/plane and contact approximately `5:3:2`, plus temporal continuity.
- Solve the physical leg first; use only small, capped and smoothed root correction afterward.
- During stance, lock the orientation-aware physical paw support point to its touchdown world anchor. During swing, follow the authored ankle/toe trajectory.
- Compute support points from `stage4/out/collision_hulls.npz`; do not restore the obsolete fixed paw-point model.
- Ashley's source rig requires one X reflection, `diag(-1, 1, 1)`. Do not mirror it twice or swap left/right afterward.
- Ear motion comes from `Anim_Ear.L/R`, mapped as a rest-relative orientation separately per side. `def_Ear.L/R` and copied Euler values produced the bad left ear.
- Root orientation comes from Ashley's pelvis. The former hip-only fit could become degenerate.
- Preserve v4 limits. Log morphology failures—especially the narrow SY range and missing ankle—instead of folding the leg or moving the root excessively.
- Regenerate the final `contacts` field from physical paw motion; never reuse stale source contact flags blindly.

Cheeky's corrected Stage 2 reference previously measured approximately: 1.382 m horizontal root travel versus 1.378 m source-scaled, 26.3 mm mean paw-trajectory error, 2.6 mm/frame planted slip, 100% source-contact preservation and no penetration. Remaining local mismatch is primarily the real v4 morphology: SY is about ±0.42 rad and the paw has no articulation.

Canonical file rule: Ashley's original `.blend` is immutable; maintain one current `*_V4_Retargeted.blend` per clip. Hide source/control/helper rigs in viewport and render. Playback must depend only on the physical root plus 21 v4 joints.

## Stage 3 — exact kinematic Isaac replay

- Export: `scripts/bake_conform.py`
- Viewer: `rl/tools/replay_v4.py`
- Output: `motions/<clip>_v4.npz`
- Writes the full root pose and all 21 joints by name, frame-by-frame, with gravity disabled.
- Validated result: zero joint write error, approximately zero root error and approximately exact whole-body FK.

View any Stage 3 clip:

```bash
cd ~/robotics/IsaacLab
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/replay_v4.py \
  --motion ~/Bingo/Blender/motions/<clip>_v4.npz --all --loops 0
```

`--loops 0` repeats forever. `--scene living-room` is the GUI default; use `--scene plain` for the basic ground scene.

Current motion names include:

```text
cheeky_v4.npz                 timid_v4.npz
deadpan_v4.npz                eccentric_v4.npz
enthusiastic_v4.npz           laidback_v4.npz
reactionanimation_test_v4.npz # Yes
reaction_no_v4.npz
reaction_what_v4.npz
```

## Stage 4 — full physics

Main files:

```text
rl/tools/track_v4_physics.py
stage4/contact_model.py
stage4/build_contact_model.py
stage4/dynamic_retarget.py
stage4/dynamic_audit.py
stage4/stand_test.py
```

The validated baseline uses the exact v4 physics model, collision hulls, gravity, contact and `IdealPDActuator`. It interpolates 24 Hz reference joint targets at 120 Hz physics. After reset, it does **not** teleport or prescribe the floating root.

Stage 4 optimization order is:

1. Local temporal adjustment.
2. Contact/force-aware control while preserving stance anchors.
3. Small balance/root-reference correction through physical leg controls—not root teleportation.
4. Small local leg correction only when needed.

Audit root XYZ/orientation/velocities, joint tracking, paw paths, contact timing, penetration, effort/velocity/joint limits, and fall frame. `--vel-ff` stays off: it regressed tracking because `Kd × qdot_ref` could exceed actuator authority.

View a Stage 4 result:

```bash
cd ~/robotics/IsaacLab
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/track_v4_physics.py \
  --motion ~/Bingo/Blender/stage4/out/<motion>.npz --loops 0
```

Validated simple reaction outputs:

| Clip | Canonical Stage 4 input | Result |
|---|---|---|
| Yes | `reaction_yes_v4_dynamic.npz` | 55 frames; completes; 4/4 contacts |
| No | `reaction_no_v4_dynamic.npz` | 19 frames; completes; 4/4 contacts |
| What | `reaction_what_v4_dynamic.npz` | 71 frames; completes; 4/4 contacts |

Yes measured mean/max joint error 0.0057/0.2073 rad, mean/max root position error 0.9/1.0 mm, mean/max root orientation error 0.08/0.09°, and mean/max paw slip 0.04/0.20 mm per frame, with no sustained torque saturation.

Current limitation: Cheeky and Timid can complete physics playback, but their Stage 4 world-space fidelity is not yet a holistic pass. Their latest logs end around 1.405 m/77.8° error for Cheeky and 0.611 m/11.3° for Timid. Survival alone does not make these clips complete.

## Stage 5 — residual RL

Files:

```text
rl/bingo_rl/bingo_rl/stage5/bingo_stage5_env.py
rl/bingo_rl/bingo_rl/stage5/bingo_stage5_env_cfg.py
rl/bingo_rl/bingo_rl/track/agents/skrl_ppo_cfg.yaml
rl/tools/eval_stage5.py
```

- Tasks: `Bingo-Stage5-Timid-Direct-v0` and `Bingo-Stage5-Timid-Play-v0`.
- The policy controls only the 12 leg residuals: `q_target = q_ref + 0.3 × action`.
- The 9 head/tail/ear joints remain reference feed-forward and are not learned.
- Simulation is 120 Hz with decimation 5, producing 24 Hz control and first-order reference interpolation.
- Training RSI samples arbitrary reference frames and initializes root pose plus all 21 joints from that frame. Current config initializes velocities to zero because reference-velocity seeding regressed the baseline.
- A Timid run and checkpoint exist at `logs/skrl/bingo_stage5_timid/2026-08-26_13-59-06_ppo_torch/`.
- The learned Timid policy reduced final root-position error in one evaluation, but worsened joint imitation and produced frequent torque saturation. It is **not** a general or visually validated solution.
- Stage 5 is unnecessary for Yes/No/What because those clips already pass Stage 4.

## Residential presentation scene

- Scene: `assets/environments/bingo_living_room.usda`
- Loader/camera: `rl/tools/indoor_scene.py`
- Selective NVIDIA asset downloader: `tools/fetch_residential_assets.py`
- Vendor assets: `assets/vendor/nvidia_residential/` and ignored by Git.

The composed room uses real NVIDIA Residential Assets furniture plus a lightweight architectural shell. It contains 212 rendered prims and no collision or rigid-body APIs, so it changes presentation only—not robot physics or contact results.

Install the required subset if missing:

```bash
cd ~/Bingo/Blender
python3 tools/fetch_residential_assets.py --preset bingo-living-room
```

The subset is about 1.1 GB; the full NVIDIA archive is about 22.5 GB and is not required. First launch is slower while materials/shaders compile.

## Working conventions and environment

- Project: `~/Bingo/Blender`
- Blender: `~/Bingo/local/blender-5.2.0-linux-x64/blender`
- Isaac Lab: `~/robotics/IsaacLab`
- Isaac Sim: 4.5.0 on the current RTX 2070 machine.
- Correct launcher: `cd ~/robotics/IsaacLab && ./isaaclab.sh -p <script> ...`
- A shell continuation backslash must be the final character on its line. `\ --motion` is invalid and causes the “required: --motion” error.
- Use Blender EEVEE for headless renders.
- Do not change Stage 3 export infrastructure, the physical rig, URDF limits, physics baseline, PD/friction/timestep, or start RL unless a reproducible defect requires it.
- Avoid routine suffixes such as `_v2`, `_fixed`, `_trusted` or `_final_final`. Keep one original source, one current v4 Blender file, one Stage 3 NPZ and one active Stage 4 result per clip.

Detailed history and clip evidence remain in `CODEX_HANDOFF.md` and `CHEEKY_RETARGET_REPORT.md`; this file is the concise current source of truth.
