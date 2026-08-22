# Bingo — Robot Specifications for Animation

Answers to the questions in *Blender Animation to Robot Motion Guide*, section 1.
All values read directly from `bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf`
and verified by forward kinematics, not transcribed from notes.

**One framing correction up front:** Bingo is not a ROS-controlled robot. There is no
JointTrajectory, no controller, no hardware step yet. The animation is training data for a
reinforcement-learning policy in NVIDIA Isaac Lab. That changes a few things (see §6), but
the guide's sections 1–6 all still apply.

---

## 1. Joint names and hierarchy

17 joints total. Base link is `origin` — a **floating base**, so root translation *is*
required (guide p.3 says "only if the robot has a mobile base" — Bingo qualifies).

```
origin  (floating base)
├── fl_SY_J → fl_SP_J → fl_knee      front-left leg
├── fr_SY_J → fr_SP_J → fr_knee      front-right leg
├── bl_SY_J → bl_SP_J → bl_knee      back-left leg
├── br_SY_J → br_SP_J → br_knee      back-right leg
├── head_pitch_joint → head_yaw → head_roll
└── tail_pitch → tail_yaw
```

`SY` = shoulder yaw (sideways swing), `SP` = shoulder pitch (fore/aft swing), `knee` = knee.
Three joints per leg. **There is no ankle, paw, or toe joint** — the knee is the last motor.

## 2–3. Rotation axis and positive direction

Measured by rotating each joint +0.20 rad from zero and recording where the foot moves.
Frame is **+X forward, +Y left, +Z up**.

| Joint | Axis | +0.20 rad moves the foot | In plain terms |
|---|---|---|---|
| `fl_SY_J` | −1 0 0 | Y −40.6 mm | **inward** (toward the body) |
| `fr_SY_J` | +1 0 0 | Y +40.6 mm | **inward** |
| `bl_SY_J` | +1 0 0 | Y +37.8 mm | **outward** |
| `br_SY_J` | −1 0 0 | Y −37.8 mm | **outward** |
| `fl_SP_J` | 0 +1 0 | X −38.9 mm | **backward** |
| `fr_SP_J` | 0 −1 0 | X +39.9 mm | **forward** |
| `bl_SP_J` | 0 +1 0 | X −38.4 mm | **backward** |
| `br_SP_J` | 0 +1 0 | X −38.4 mm | **backward** |
| `fl_knee` | 0 −1 0 | X +23.8 mm | **forward** |
| `fr_knee` | 0 +1 0 | X −23.8 mm | **backward** |
| `bl_knee` | 0 −1 0 | X +23.8 mm | **forward** |
| `br_knee` | 0 +1 0 | X −23.8 mm | **backward** |

⚠️ **The left and right sides are not consistent.** `fr_SP_J` and `fr_knee`/`br_knee` are
flipped relative to their partners, and positive `SY` means *inward* on the front legs but
*outward* on the back. This is a quirk of how the robot was exported, and it is the single
most common source of the "one leg bends backwards" bug.

**You do not have to reproduce this.** Please animate in a consistent, symmetric convention
(all four legs behaving the same way) and we will apply the sign flips on import. If you *do*
choose to mirror the robot's convention instead, say so explicitly in the delivery notes so
we don't apply the flips twice.

Head/tail: `head_pitch_joint` axis `0 −1 0`, `head_yaw` axis `−1 0 0`, `head_roll` axis
`0 0 −1`, `tail_pitch` axis `0 −1 0`, `tail_yaw` axis `−1 0 0`.

## 4. Zero / home pose

**Zero = all joints at 0 rad = all four legs straight down, head and tail neutral.**

But zero is a singularity the robot never actually stands in. The *functional* neutral is a
crouch:

| | SP | knee |
|---|---|---|
| fl | −0.25 rad (−14.3°) | +0.50 rad (+28.6°) |
| fr | +0.25 rad (+14.3°) | −0.50 rad (−28.6°) |
| bl | −0.25 rad (−14.3°) | +0.50 rad (+28.6°) |
| br | −0.25 rad (−14.3°) | −0.50 rad (−28.6°) |

Verified: all four feet land level within 0.6 mm, base at **0.1989 m** — inside the stated
0.19–0.20 m band. **Treat this as the neutral standing pose**, not straight legs.

⚠️ Correction to the internal spec: it lists the crouch as SP −0.30 / knee +0.60, but that
gives 0.1859 m, *below* the 0.19–0.20 m stance height the same document states. The two
numbers disagree. −0.25/+0.50 satisfies the stated stance height and is what the rig ships in.

Straight legs (all joints 0) would stand at 0.2273 m, but that is a kinematic singularity the
robot never uses — some knee bend is correct. 0.1989 m is 87% of maximum extension, a normal
standing posture rather than a crouch.

Heights above are ground-to-base, measured to the **paw surface**. If you measure to the
*shank tip* instead you'll get ~29 mm less, because the paw mesh hangs 28–30 mm below the
tip. Both are correct; they just measure different things.

## 4b. Foot reach — the thing that will surprise you most

The standing pose puts each paw **ahead of its own hip** (front +103 mm, back +44 mm). That
is the robot's real operating stance — it matches the motion data the policy already trains
on — but it means the forward half of the leg's travel is nearly used up before you start.

Measured by IK against the URDF (two independent solvers agree exactly):

| Stance height | Foot can move FORWARD | Foot can move BACK |
|---|---|---|
| 0.210 m | 16 mm | 169 mm |
| **0.199 m (default)** | **35 mm** | **188 mm** |
| 0.190 m | 48 mm | 201 mm |
| 0.180 m | 60 mm | 213 mm |
| 0.170 m | 70 mm | 223 mm |

**Practical rules:**

- Build steps by sweeping the paw **backward** from its neutral position, not forward. There
  is lots of room behind and almost none in front.
- If you need more forward reach, **lower the body** — dropping the stance to 0.18 m roughly
  doubles it.
- Push a foot control further forward than the table allows and the paw simply stops
  following it. That is the robot's real workspace boundary, not a rig bug, and the rig is
  deliberately refusing to fake a pose the machine cannot hit.

## 5. Joint limits

| Joint | Min | Max | Min° | Max° | Please author within |
|---|---|---|---|---|---|
| `*_SY_J` (all 4) | −0.42 | +0.42 | −24.1° | +24.1° | **±22°** |
| `fl/fr/br_SP_J` | −1.56 | +1.56 | −89.4° | +89.4° | **±80°** |
| `bl_SP_J` | −1.57 | +1.57 | −90.0° | +90.0° | **±80°** |
| `fl/fr/br_knee` | −1.56 | +1.56 | −89.4° | +89.4° | **±80°** |
| `bl_knee` | −1.57 | +1.57 | −90.0° | +90.0° | **±80°** |
| `head_pitch_joint` | −0.65 | +0.40 | −37.2° | +22.9° | −33° … +20° |
| `head_yaw` | −0.60 | +0.60 | −34.4° | +34.4° | ±31° |
| `head_roll` | −0.78 | +0.78 | −44.7° | +44.7° | ±40° |
| `tail_pitch` | −0.60 | +0.60 | −34.4° | +34.4° | ±31° |
| `tail_yaw` | −0.60 | +0.60 | −34.4° | +34.4° | ±31° |

**The two that will bite you:**

- **Knee tops out at ±89°** from straight (≈179° of total travel — a normal range, but a
  real dog's stifle folds further). Deep tucks, sits, and lie-downs will hit it. We
  retargeted the *Laidback* clip and the front-left knee sat pinned against this limit for
  **35% of the clip** — that clip is not usable as-is.
- **SY is only ±24°.** Almost no sideways range. Wide stances, crab walks, and lateral
  weight shifts are out of reach.

## 6. Speed and acceleration

| | Max velocity | Max torque |
|---|---|---|
| Leg joints | **10 rad/s** (573°/s) | 3.0 N·m |
| Head / tail | **8 rad/s** (458°/s) | 1.5 N·m |

**Acceleration is not specified in the URDF** — torque is the real constraint. The robot is
only **2.478 kg** with 3.0 N·m per leg joint, which is weak. Jumps, hard accelerations, and
fast recoveries are likely physically impossible. We'll flag anything infeasible on our side.

## 7. Units

- **Radians** for all joint angles (not degrees).
- **Metres** for all positions.
- **+Z up, +X forward, +Y left**, right-handed.
- Ground plane is **z = 0**.

Blender defaults to Y-up on export; the conversion happens on our side, but please state
which convention the delivered file uses.

## 8. Frame rate

- **24 fps is acceptable.** Our importer samples your F-curves directly, so we can resample
  to 120 Hz cleanly regardless of what you keyed on.
- **60 fps is better** if the motion is fast — 24 fps genuinely can't represent detail
  faster than that, and no resampling recovers it.
- **Must be constant.** No retiming, no time-warping, no motion blur on the exported channels.
- **Do not use Clean Curves / keyframe reduction.** Your guide already says "Clean Curves:
  Off initially" (p.6) — please keep it off entirely. We compute velocities by differencing
  positions, so decimation becomes velocity noise.

## 9. Input format — what we actually want

**Simplest: just send the `.blend` file.** We have a Blender script that reads the evaluated
rig directly and bakes what we need. That's how the six existing performances were processed,
and it works. No FBX, no BVH, no CSV needed.

If you'd rather deliver baked data, our order of preference is:

1. **JSON / CSV of joint angles** — your guide's Option C, p.9. Zero ambiguity.
2. **FBX with an FK bake** — your Option B settings on p.8 are correct as written.
3. **BVH** — works, but please state rotation order and axis convention in the header.

What we do **not** need: ROS JointTrajectory (§7 of your guide). Correct for ROS robots,
just not applicable here.

---

## Two things to add that aren't in the guide

### Contact flags

Per frame, per foot, a boolean: **is this paw planted on the ground?**

This is the highest-value thing you could add. Right now we infer it from paw height, which
is a guess, and it feeds the step that reconstructs the body's motion. With real flags that
step becomes exact. A custom property on each foot bone, or a sidecar CSV keyed by frame,
either is fine.

### The contact point is the shank tip

The robot has no paw joint. Its "foot" is a single point **0.120 m down from the knee hinge**,
along the shank. The paw geometry around it is decorative. So paw roll, toe-off, and heel
strike can't be reproduced — animate them if they help the performance read, but know they'll
be dropped.

---

## Notes on the current rig

Findings from processing the six existing performance files — these apply to
`BingoRig_Latest.blend` and the shot files built on it:

1. **The `.L` / `.R` bone names are mirrored.** The character faces −Y, which makes its left
   side +X, but every `.L` bone sits at −X. So `def_Arm.L` is on the character's *right*.
   We correct for it on import, but it's worth knowing before anyone builds on that rig.

2. **The rig's proportions are not the robot's.** Thigh:shank is 0.60 on the rig vs **0.70**
   on the robot, and the hips are about **1.9× narrower**. No single scale factor fixes it,
   which is why we solve foot positions with IK rather than copying joint angles.

3. **Extra articulation the robot doesn't have:** a 2-bone spine, a paw/hand segment on each
   limb, and ear bones. All dropped on import. Motion whose read depends on spine flex will
   look stiffer on the robot.

4. **The rig is not authored at real-world scale** — roughly 1 Blender unit ≈ 9.4 mm. Fine
   for us since we rescale, but if we build a new rig it should be at true metres.

## Recommendation

Rather than hand-building the `ROBOT_EXPORT` armature from §2 of your guide, **we can
generate it directly from the URDF** — bones at the exact joint positions, each bone's roll
set so its local axis matches the real motor axis, hard rotation limits baked in, and IK
controls on top. Names, axes, limits, and zero pose would then be correct by construction
rather than by manual matching, which removes the entire class of problems your §2 and the
axis-validation test on p.4 are designed to catch.

If that's useful, we'll send you the rig and you'd animate directly on it — and retargeting
essentially disappears.

---

# The conform rig — `Bingo_ConformRig.blend`

Generated directly from the URDF by `build_rig.py`, so it is correct by construction rather
than by manual matching. **This replaces section 2 of your guide** — no `ROBOT_EXPORT`
armature to build, and the axis-validation test on p.4 is already satisfied.

## What's in it

- **17 joint bones**, named exactly as the robot's joints, at the exact joint positions,
  at true real-world scale (Bingo is ~19 cm at the shoulder — scene units are centimetres).
- **Each bone's local Z axis is that motor's rotation axis**, verified exact to 0.0000°.
- **X and Y rotation are locked** on every joint bone. A 1-DOF hinge cannot be animated as
  a ball joint even by accident.
- **Limit Rotation constraints** at the robot's real limits with margin. Poses outside the
  robot's range are not authorable.
- **The robot's own 18 STL meshes**, attached to the correct bones. You're animating Bingo.
- **4 `*_foot_tip` marker bones** — zero DOF, these mark the actual ground-contact points.
- Opens in the **crouch**, which is the true neutral standing pose.

Verified: in that crouch all four feet land level within **0.29 mm**, and foot positions match
the URDF's own forward kinematics exactly.

## How to work with it

- **Animate the joint bones in Pose Mode.** Only the Z rotation channel will move; the other
  two are locked deliberately.
- **`root`** carries the whole robot's position and orientation in the world. Locomotion goes
  here — never in the legs.
- **Do not rename bones, change bone roll, or remove the constraints.** Those are the parts
  that make the rig match the robot.
- **Ground is z = 0.** The paw surface should rest on it; note the `*_foot_tip` markers sit
  ~29 mm *above* the ground when the paw is planted, because the tip is inside the paw.
- Keep the scene at a **constant frame rate**, and leave keyframe reduction / Clean Curves off.

## What it does not have yet

**No IK foot controls.** This first version is FK only — you pose the joints directly. IK is
very doable (foot targets plus knee poles) and if posing legs in FK is painful, say so and
we'll add it. We left it out of v1 because IK controls must not end up in the exported
chain, and we wanted the exact part correct first.

No spine, no ears, no paw/toe joints — the robot doesn't have those motors.

## Delivery

Just send the `.blend` back. Our importer reads the evaluated rig directly, so there is no
export step, no format decision, and no unit conversion for you to get right.

If you can also include per-foot contact flags (planted / not planted, per frame) as a custom
property or a sidecar CSV, that measurably improves the result — see above.
