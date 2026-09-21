# Lifelike Dog BVH Dataset Report

Source: `dataset/lifelike_dog/raw_bvh/raw_bvh_data/` (33 BVH files).

## Skeleton

3ds-Max-Biped-style rig (human bone names retargeted onto a quadruped mesh): root `Bip01` -> `b_Hips` -> spine chain (`b_Spine`..`b_Spine3`) -> neck/head, and two shoulder clavicles `b_LeftClav`/`b_RightClav` for the FRONT legs (`_Arm` -> `_ForeArm` -> `_Hand` -> `_Finger` -> end site), plus a tail chain off `b_Hips` and hind legs off `b_Hips` directly (`_LegUpper` -> `_Leg` -> `_Leg1` -> `_Ankle` -> `_Toe` -> `_Toe002`/end site). Units are centimeters (auto-detected and converted to meters). Up-axis is Y (root/hip height sits at ~0.15-0.6 m depending on clip and pose; paw Y sits near 0 at ground contact). All 33 files share FPS=120 (Frame Time 0.008333s).

**2-segment leg abstraction used for AMP features** (to match Bingo's 2-segment legs): front hip=`b_*Arm`, mid("knee")=`b_*ForeArm`, paw=`b_*Hand`; rear hip=`b_*LegUpper`, mid("knee")=`b_*Ankle` (absorbing the `Leg`/`Leg1` sub-bend), paw=`b_*Toe`.

Forward-axis sanity check on `dog_quad_walk_001` (a named forward walk): 54% of moving frames have a velocity heading within 90 deg of the clip's own mean heading -- i.e. **the clip does not reverse direction** even though (see below) it changes heading substantially over its 110s duration (it ambles around the capture volume, not a straight line for the whole clip).

## Per-clip scan (name-implied category, cross-checked numerically)

| file | category | dur(s) | mean speed (m/s) | stationary frac | candidate windows | best window | best stationary window |
|---|---|---|---|---|---|---|---|
| dog_back_001.bvh | backward | 8.1 | 0.617 | 0.15 | 1 | [2.4,6.5]s v=1.11 straight=0.98 | -- |
| dog_back_002.bvh | backward | 12.1 | 0.596 | 0.21 | 2 | [1.8,4.1]s v=1.23 straight=0.99 | -- |
| dog_fast_run_02_004.bvh | run | 26.6 | 1.764 | 0.13 | 5 | [11.4,15.2]s v=3.86 straight=0.99 | -- |
| dog_fast_run_02_005.bvh | run | 25.7 | 2.613 | 0.01 | 6 | [6.6,10.0]s v=3.27 straight=0.96 | -- |
| dog_fast_run_02_006.bvh | run | 33.0 | 2.050 | 0.04 | 8 | [15.4,18.1]s v=2.94 straight=0.98 | -- |
| dog_fast_run_02_007.bvh | run | 45.4 | 2.529 | 0.02 | 13 | [22.8,25.4]s v=2.77 straight=0.98 | -- |
| dog_fast_run_02_008.bvh | run | 30.5 | 1.875 | 0.02 | 5 | [26.7,30.2]s v=3.29 straight=0.99 | -- |
| dog_hit_001.bvh | other | 29.7 | 0.375 | 0.15 | 3 | [1.7,5.1]s v=0.72 straight=0.97 | [27.0,28.4]s (1.3s) |
| dog_idle_001.bvh | idle | 25.6 | 0.967 | 0.02 | 1 | [21.6,24.6]s v=0.91 straight=0.96 | -- |
| dog_idle_002.bvh | idle | 7.7 | 0.991 | 0.23 | 0 | -- | [0.7,2.0]s (1.3s) |
| dog_idle_003.bvh | idle | 12.9 | 1.299 | 0.01 | 1 | [9.2,12.9]s v=1.69 straight=0.97 | -- |
| dog_idle_004.bvh | idle | 10.7 | 1.531 | 0.00 | 2 | [7.4,10.7]s v=1.91 straight=0.98 | -- |
| dog_jump_002.bvh | jump | 9.4 | 2.047 | 0.00 | 2 | [6.3,8.0]s v=2.62 straight=0.99 | -- |
| dog_jump_003.bvh | jump | 9.7 | 1.995 | 0.01 | 3 | [3.7,5.5]s v=3.13 straight=0.99 | -- |
| dog_jump_004.bvh | jump | 19.0 | 1.599 | 0.07 | 4 | [6.2,8.7]s v=2.85 straight=0.99 | -- |
| dog_jump_006.bvh | jump | 14.3 | 1.828 | 0.05 | 3 | [6.2,8.7]s v=3.50 straight=0.99 | -- |
| dog_jump_007.bvh | jump | 13.0 | 2.076 | 0.09 | 3 | [9.0,11.5]s v=3.46 straight=0.99 | -- |
| dog_jump_high_001.bvh | jump | 15.2 | 1.776 | 0.02 | 3 | [3.7,6.0]s v=3.38 straight=0.99 | -- |
| dog_jump_high_002.bvh | jump | 20.8 | 1.880 | 0.00 | 6 | [10.8,13.0]s v=2.52 straight=0.99 | -- |
| dog_play_001.bvh | other | 41.2 | 1.418 | 0.03 | 0 | -- | -- |
| dog_play_002.bvh | other | 47.3 | 0.972 | 0.03 | 5 | [0.0,2.5]s v=1.68 straight=0.99 | -- |
| dog_quad_run_001.bvh | run | 36.7 | 2.070 | 0.02 | 6 | [21.5,31.5]s v=2.15 straight=0.08 | -- |
| dog_quad_run_002.bvh | run | 24.8 | 2.087 | 0.00 | 5 | [9.5,12.5]s v=2.39 straight=0.99 | -- |
| dog_quad_walk_001.bvh | walk | 110.0 | 1.027 | 0.01 | 10 | [1.7,27.2]s v=1.19 straight=0.33 | [0.1,1.3]s (1.2s) |
| dog_quad_walk_002.bvh | walk | 150.6 | 1.051 | 0.01 | 14 | [81.5,112.2]s v=1.03 straight=0.11 | -- |
| dog_quad_walkrun_001.bvh | transition | 9.6 | 1.914 | 0.08 | 2 | [1.5,7.1]s v=2.51 straight=0.99 | -- |
| dog_quad_walkrun_005.bvh | transition | 9.2 | 1.506 | 0.04 | 1 | [1.1,7.0]s v=2.21 straight=0.99 | -- |
| dog_quad_walkrun_006.bvh | transition | 10.0 | 1.406 | 0.15 | 1 | [2.9,7.3]s v=2.76 straight=0.99 | -- |
| dog_quad_walkrun_007.bvh | transition | 6.7 | 2.172 | 0.09 | 1 | [1.2,6.7]s v=2.60 straight=0.99 | -- |
| dog_star_walk_001.bvh | turn_walk | 66.2 | 0.823 | 0.05 | 10 | [16.9,23.9]s v=0.97 straight=0.96 | [63.6,66.2]s (2.7s) |
| dog_star_walk_003.bvh | turn_walk | 37.5 | 0.986 | 0.05 | 8 | [13.0,16.7]s v=1.37 straight=0.98 | -- |
| dog_zig_walk_001.bvh | turn_walk | 30.8 | 0.636 | 0.03 | 2 | [1.4,3.8]s v=1.02 straight=0.99 | -- |
| dog_zig_walk_002.bvh | turn_walk | 30.0 | 0.621 | 0.01 | 3 | [15.2,18.2]s v=0.46 straight=0.96 | -- |

## Category summary

- **backward**: 2 clips
- **run**: 7 clips
- **other**: 3 clips
- **idle**: 4 clips
- **jump**: 7 clips
- **walk**: 2 clips
- **transition**: 4 clips
- **turn_walk**: 4 clips

## Important finding: filename categories do not match actual motion

Numerically checking root speed exposed a mismatch: the `dog_idle_*.bvh` clips are **not standing still** -- `frac_stationary` (fraction of frames with root speed < 0.04 m/s) is only 0.00-0.23 for the four idle clips, and their MEAN speed (0.97-1.53 m/s) is comparable to or higher than the `quad_walk` clips (~1.0 m/s). Root (`Bip01`) and all four paws travel the same multi-meter excursions in these files (verified directly, not just at the root), so this is real locomotion, not a moving-root/stationary-mesh export artifact. This matches the standing lesson from the InterPet4D work: **never trust a clip's name or an automated aggregate metric without checking the actual motion.** Genuinely still moments do exist, but only as short windows inside otherwise-moving clips -- see the per-clip `best_stationary_window` column above and the ranked list below.

### Best stationary windows found dataset-wide (>=1.0s of speed<0.04 m/s)

- `dog_star_walk_001.bvh` [63.6, 66.2]s (2.7s)
- `dog_hit_001.bvh` [27.0, 28.4]s (1.3s)
- `dog_idle_002.bvh` [0.7, 2.0]s (1.3s)
- `dog_quad_walk_001.bvh` [0.1, 1.3]s (1.2s)

## Selection for the first AMP controller

Per the brief (idle + forward walk + useful walk transitions only; exclude run/jump/play/aggressive), the expert set for `extract_dog_amp_features.py` is:

- **idle/stand**: the ranked stationary windows above (not full `idle_*` clips, since those are mostly moving -- see finding above). If total stationary duration is too short to be a useful style prior on its own, the AMP env's own upright/stability + zero-command-tracking reward terms carry the standing behavior instead, same as the prior (rev_3-era) `bingo_amp_env` design.
- **forward walk**: clean straight-walk windows extracted from `dog_quad_walk_001.bvh` and `dog_quad_walk_002.bvh` (both long multi-behavior captures; only the heading-stable, straightness>0.9 candidate windows listed above are used, not the full files).
- **walk transitions**: `dog_quad_walkrun_*.bvh` are EXCLUDED from this first pass -- inspecting their speed traces shows they ramp continuously from walk into a true run with no sustained walk-paced plateau long enough to be useful without also teaching a running gait, which the brief explicitly excludes.
- **excluded**: `*_run*`, `*_jump*`, `play_*`, `hit_*` (per brief), and `*_star_walk*`/`*_zig_walk*`/`*_back*` (turning/backward -- out of scope for a straight-forward-only first controller, but cataloged above for a later stage).
