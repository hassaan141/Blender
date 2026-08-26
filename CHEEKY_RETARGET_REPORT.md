# Cheeky -> v4 spatial retarget - evaluation report

Source: `Bingo_Cheeky.blend`  ->  target: exact v4 physical skeleton (`Bingo_Cheeky_V4_Retargeted.blend`).  Kinematic only (no physics/RL).

## 1. Discovered rig mapping

Ashley's rig is a biped-style quadruped (front legs = Arm/ForeArm/Hand, back legs = Leg/Shin/Foot) driven through `def_*` deform bones. Its anatomical frame (forward +Y, up +Z) is **mirrored/left-handed** vs the robot (forward +X, left +Y, up +Z); the source is reflected across YZ before fitting.

| Ashley (source) | v4 (target) | role |
|---|---|---|
| aFL (front-left) | `fl_SY_J/SP_J/knee` | leg |
| aFR (front-right) | `fr_SY_J/SP_J/knee` | leg |
| aBL (back-left) | `bl_SY_J/SP_J/knee` | leg |
| aBR (back-right) | `br_SY_J/SP_J/knee` | leg |
| def_Pelvis | `root` (floating base) | body |
| def_Head | `head_pitch_joint/head_yaw/head_roll` | head (3 DOF) |
| def_Tail.001 | `tail_pitch/tail_yaw` | tail (2 DOF) |
| Anim_Ear.L / Anim_Ear.R | `l_ear_*` / `r_ear_*` | visible terminal ears (2+2 DOF) |

Per-leg length scale (source units -> metres): `fl`=0.00941, `fr`=0.00941, `bl`=0.00939, `br`=0.00939.

## 2. Source clip

- Frames **1-180** (180 frames), **24 fps**, 7.50 s.
- Method: evaluated world-space `def_*` keypoints -> per-leg limb-length scaling -> per-frame Kabsch root fit -> contact-anchored per-leg IK (scipy `least_squares`, bounds = exact v4 URDF limits) -> head/tail/ear rotation-chain solve -> globally continuous absolute contact-anchor root solve -> velocity-constrained trajectory refinement.

## 3. Contact intervals (source schedule, preserved on the robot)

| foot | duty | planted intervals |
|---|---|---|
| fl (front-left) | 39% | 7-8, 14-16, 22-24, 30-51, 59-60, 65-76, 135-145, 165-180 |
| fr (front-right) | 34% | 10-12, 18-20, 26-28, 35-44, 53-57, 65-75, 104-105, 138-145, 165-180 |
| bl (back-left) | 46% | 1-4, 10-11, 18-20, 26-28, 35-46, 53-57, 65-73, 84-98, 135-145, 149-151, 165-180 |
| br (back-right) | 42% | 6-8, 14-15, 22-24, 30-33, 41-45, 59-73, 85-98, 135-145, 149-151, 165-180 |

Contact schedule preserved (source vs solved, per foot-frame): **100.0%**.

## 4. Quantitative errors

| metric | mean | p95 | max |
|---|---|---|---|
| foot trajectory error (hip-relative, scaled) | 101.5 mm | 177.2 mm | 208.2 mm |
| foot IK residual (target vs achieved) | 28.2 mm | 74.9 mm | 124.8 mm |
| knee trajectory error (hip-relative, scaled) | 75.3 mm | 120.1 mm | 158.2 mm |
| root deviation from scaled Ashley body | 48.4 mm | 92.7 mm | 98.1 mm |
| root orientation error | 0.000 deg | 0.000 deg | 0.000 deg |

- **Planted-foot sliding:** mean 3.3 mm/frame, max 21.0 mm between consecutive planted frames.
- **Ground penetration:** max 10.1 mm below the floor (10 frame(s) with any paw < -1 mm).
- **Joint velocity:** legs max 10.0 rad/s (limit 10), expression max 8.0 rad/s (limit 8).

### Gait phase evidence

| foot | source stance speed | target stance speed | source swing distance | target swing distance |
|---|---:|---:|---:|---:|
| fl | 1.6 mm/s | 62.2 mm/s | 3.785 m | 3.717 m |
| fr | 1.1 mm/s | 46.8 mm/s | 3.894 m | 3.898 m |
| bl | 0.8 mm/s | 44.5 mm/s | 3.606 m | 3.474 m |
| br | 1.8 mm/s | 71.4 mm/s | 3.689 m | 3.409 m |

## 5. Joint-limit saturation (poses v4 could not fully reproduce)

| joint | % frames on limit | frame span | limit (rad) |
|---|---|---|---|
| `fl_SY_J` | 39.4% | 54-155 | [-0.42, 0.42] |
| `fr_SY_J` | 37.2% | 52-180 | [-0.42, 0.42] |
| `fr_SP_J` | 3.9% | 98-130 | [-1.56, 1.56] |
| `bl_SY_J` | 43.9% | 32-180 | [-0.42, 0.42] |
| `bl_knee` | 10.0% | 163-180 | [-1.57, 1.57] |
| `br_SY_J` | 44.4% | 52-133 | [-0.42, 0.42] |
| `br_knee` | 0.6% | 162-162 | [-1.56, 1.56] |
| `head_pitch_joint` | 61.7% | 29-180 | [-0.65, 0.05] |
| `head_roll` | 9.4% | 33-49 | [-0.78, 0.78] |
| `tail_pitch` | 21.7% | 55-96 | [-0.60, 0.60] |
| `l_ear_roll` | 47.8% | 1-153 | [-1.50, 0.00] |
| `r_ear_roll` | 24.4% | 1-153 | [0.00, 1.50] |

Interpretation: the `SY` (shoulder-yaw/abduction) limit is only **+/-0.42 rad (~24 deg)**, so Ashley's wide lateral paw placements and the big front-paw raises are the main gestures v4 physically cannot match one-to-one; these are recorded here rather than hidden, and the amplitude is **not** globally shrunk to make the solver succeed.

## 6. Root-cause diagnosis and changes

- **Gait failure:** leg targets omitted the established source-to-v4 axis matrix `A`; the evaluator repeated the omission. The old cumulative de-slip was then blurred, so it neither enforced stance anchors nor preserved source travel. Fixed by applying `A` once, using absolute per-stance anchors, solving one globally continuous root offset, and refining joints under explicit velocity bounds instead of globally blurring the clip.
- **Left-ear failure:** Stage 2 sampled `def_Ear.*`, while the visible meshes inherit the child `Anim_Ear.*` transforms. It also solved body-relative ear orientation without removing the already-achieved v4 head rotation. Fixed by sampling visible terminal bones and solving left/right separately relative to the achieved head and each target rest frame.
- **Floating bones:** ten `ctrl_*` bones, four `*_ik_end` helpers, and four `*_foot_tip` markers stayed at the rig rest location while the physical root moved. All 18 nonphysical bones are removed from the final baked file. Only physical `root` + 21 joints remain, are keyed, and are required.

## 7. Largest gait-error frames

| frame | worst foot | foot error |
|---|---|---|
| 115 | bl | 208 mm |
| 129 | bl | 206 mm |
| 130 | bl | 203 mm |
| 120 | bl | 203 mm |
| 116 | bl | 202 mm |
| 121 | bl | 202 mm |
| 148 | br | 202 mm |
| 119 | bl | 202 mm |

## 8. Ear mapping and largest ear-error frames

The source is reflected exactly once across X (`F R F`) and then aligned with `A`; ears are not swapped for the selected direct anatomical mapping. Source terminal rest orientation is removed before mapping. Target axes and asymmetric limits come directly from the unchanged v4 URDF.

| side | target chain | pitch axis | roll axis | limits (rad) |
|---|---|---|---|---|
| left | `l_ear_pitch` -> `l_ear_roll` | `[0.028, -0.914, -0.405]` | `[-0.996, 0.01, -0.09]` | [-3.00,3.00], [-1.50,0.00] |
| right | `r_ear_pitch` -> `r_ear_roll` | `[-0.028, -0.913, 0.406]` | `[-0.996, -0.01, -0.09]` | [-3.00,3.00], [0.00,1.50] |

| side | frame | terminal orientation error |
|---|---:|---:|
| left | 77 | 48.9 deg |
| left | 51 | 35.8 deg |
| left | 52 | 33.7 deg |
| left | 76 | 33.5 deg |
| left | 78 | 31.0 deg |
| right | 68 | 28.8 deg |
| right | 69 | 28.4 deg |
| right | 67 | 28.2 deg |
| right | 70 | 27.7 deg |
| right | 74 | 27.3 deg |

Representative mapped Ashley visible-ear deltas and final v4 joint values (`wxyz`, rad):

| frame | Ashley L delta | v4 L pitch/roll | Ashley R delta | v4 R pitch/roll |
|---:|---|---|---|---|
| 1 | `[1.0, 0.0, 0.0, 0.0]` | `[0.0, -0.0]` | `[1.0, -0.0, 0.0, -0.0]` | `[0.0, 0.0]` |
| 60 | `[0.638, -0.339, 0.657, 0.216]` | `[-1.571, -0.221]` | `[0.774, -0.34, 0.473, 0.247]` | `[-1.102, 0.5]` |
| 94 | `[0.966, -0.207, -0.006, 0.157]` | `[0.005, -0.0]` | `[0.988, -0.019, 0.022, 0.15]` | `[0.003, 0.0]` |
| 121 | `[0.926, -0.172, 0.334, -0.008]` | `[-0.037, -0.0]` | `[0.938, -0.051, 0.337, -0.056]` | `[-0.04, 0.0]` |
| 127 | `[0.76, 0.524, 0.306, -0.232]` | `[-0.182, -1.331]` | `[0.67, -0.642, 0.297, 0.225]` | `[-0.16, 1.393]` |
| 128 | `[0.708, 0.594, 0.279, -0.26]` | `[-0.19, -1.476]` | `[0.63, -0.682, 0.29, 0.232]` | `[-0.165, 1.5]` |
| 180 | `[0.936, 0.32, -0.034, -0.142]` | `[-0.013, -0.467]` | `[0.977, -0.157, 0.031, -0.14]` | `[-0.007, 0.503]` |

## 9. Assumptions

- Paw contact point = tip of Ashley's Hand/Foot deform bone; robot foot = shank tip (knee frame + 120 mm along -Z), paw mesh hangs 28 mm below it.
- Front/back leg correspondence is fixed by anatomy; left/right is chosen by the lower hip-layout Kabsch residual (handles the mirrored labels).
- Expressive chains reproduce Ashley's **in-body orientation delta from frame 1**, mapped through target rest frames; ears additionally compensate for achieved head pose.
- Root follows Ashley's body strongly but is allowed to adapt (contact-consistent de-slip + ground placement) so planted paws stay put and the lowest paw rests on z=0.

## 10. Deliverables

- `blend_sources/Bingo_Cheeky_V4_Retargeted.blend`
- `stage2/out/cheeky_source_keypoints.npz`
- `stage2/out/cheeky_contacts.npz`
- `stage2/out/cheeky_v4_retarget.npz`
- `stage2/*.py`
- `CHEEKY_RETARGET_REPORT.md`
- `stage2/out/cheeky_compare.mp4 (side-by-side)`
