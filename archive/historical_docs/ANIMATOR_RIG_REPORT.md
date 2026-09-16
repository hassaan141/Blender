# Bingo v4 Animator Rig — report

**Deliverable:** `blend_sources/Bingo_V4_AnimatorRig.blend` — Ashley-style animator controls layered
on the **exact v4 physical robot skeleton**, so animation bakes directly onto the real v4 joints for
Isaac Sim with no second morphology-changing retarget.

## The bug that was fixed
In `Bingo_V4_RetargetRig.blend`, rotating `fl_SY_J` changed the value but the leg didn't move.
**Cause:** an always-on IK constraint on the leg overrode manual (FK) rotation. Meshes were parented
correctly; IK was the problem. Fixed by making IK switchable (default off → physical joints drive their
meshes directly; on → foot control drives the leg).

## Two layers (unchanged physical core)
```
ANIMATOR CONTROLS  ->  IK / constraints (driven by switches)  ->  V4 PHYSICAL BONES  ->  meshes  ->  bake -> Isaac
```
The physical layer (21 joints `fl_SY_J … r_ear_roll`, `root`, meshes) is **byte-identical** to the
verified v4 rig — rest matrices differ by 0.000e+00, axes 0.0000°, limits/segment lengths/zero pose
all match the URDF. Only a control layer was added on top.

## Controls added
| Control | Does |
|---|---|
| `ctrl_Root` (master) | moves the **whole dog** (torso + feet-in-root + head/tail/ears) as one |
| `ctrl_Body` (torso) | moves the base **relative to planted feet** (crouch/lean) — parents the physical `root` |
| `ctrl_fl_foot … ctrl_br_foot` | 4 foot **IK** targets driving the real v4 leg joints |
| `ctrl_head`, `ctrl_tail`, `ctrl_l_ear`, `ctrl_r_ear` | drive the physical head/tail/ear joints (local-Z Copy Rotation) |

## Per-leg switches (custom properties on `ctrl_Root`, wired by drivers)
- `ik_<leg>`  — **0 = FK** (rotate the physical joint directly), **1 = IK** (foot control drives the leg). Default 0.
- `footroot_<leg>` — **1 = ROOT/character space** (foot follows `ctrl_Root`), **0 = WORLD/planted** (foot fixed in world). Default 1.

Global vs body movement:
- move `ctrl_Root` → entire character translates (feet included; legs don't stretch).
- move `ctrl_Body` → torso moves relative to feet; with feet planted (WORLD) + IK on, knees bend.

Unreachable poses are respected: the IK hits the real joint limits / leg reach rather than faking
kinematics (e.g., pushing the body forward past the ~35 mm front-reach limit leaves the foot behind — a
true mechanical limit, made visible to the animator).

## Baking (Blender → npz → Isaac)
```
blender -b <animated>.blend -P scripts/bake_conform.py -- --rig Bingo_Robot --dof 21 --hz 120 --out clip.npz
```
Reads the **evaluated** physical joint angles (works with IK), the floating root, and the foot bodies →
the standard 21-DOF `.npz`. Control/helper bones are **not** baked as DOFs. Verified: a crouch keyed on
`ctrl_Body` (IK on) baked the IK-driven knee bend (0.80–0.85 rad) and root drop (0.03 m) correctly.

## Validation tests (all pass, headless)
| Test | Result |
|---|---|
| 1. Physical FK: rotate `fl_SY_J`/`fl_SP_J`/`fl_knee` | foot-tip moves 41/60/59 mm; only downstream pivots move, upstream stay 0.0 mm — **PASS** |
| 2. `ctrl_Root` +50 mm | torso +50, all 4 feet +50, max leg stretch 0.00 mm — **PASS** |
| 3. Crouch (feet WORLD, IK) `ctrl_Body` −30 mm | feet stay 0.2 mm, body −30, knees bend 17.8° — **PASS** |
| 4. Foot WORLD → body move: foot stays 0.8 mm; foot ROOT → root move: foot follows 30 mm | **PASS** |
| 5. Physical layer unchanged | rest matrices identical (0.000e+00), check_rig physical checks all PASS, `fl_ik_end` = URDF shank-tip 0.0 mm | **PASS** |

(check_rig flags the 6 new control bones as "unexpected extra bones" — expected; they are the intended controls.)

## Scripts
- `scripts/build_rig.py` — builds the physical v4 rig from the URDF (added `--exact-limits`, `--rest-pose`).
- `scripts/add_animator_controls.py` — adds the animator control layer (this rig).
- `scripts/test_animator.py` — the 5 validation tests.
- `scripts/bake_conform.py` — bakes control-driven animation → 21-DOF `.npz`.

## Not done yet (by design)
No animation copied on yet — per the plan, this proves **URDF ↔ Blender v4 ↔ Isaac v4 are the same
robot** and the rig is animatable + bakeable, before retargeting Cheeky/etc.
