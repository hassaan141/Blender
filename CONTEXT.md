# Bingo — Project Context & Verification Brief

**Written 2026-08-21.** This file exists to give an AI agent (or a new human) complete,
unembellished context on the Bingo animation→robot project so it can independently verify
the central technical question below.

**Provenance is labelled on every claim:**

- `[STATED]` — told to me by Hassaan; not independently checked.
- `[MEASURED]` — computed this session directly from files on disk. Method is given so it can be re-run.
- `[FROM DOC]` — recorded in an earlier project document ([MEMORY.md](MEMORY.md),
  [docs/BingoMocapPipelineSpec.md](docs/BingoMocapPipelineSpec.md)); those documents state they
  were verified by running them, but that has not been re-checked here.
- `[UNVERIFIED]` — asserted somewhere, with no evidence on disk supporting or refuting it.
- `[UNKNOWN]` — genuinely open.

---

## 1. End goal

`[STATED]` Train **cute, expressive locomotion and gestures** on the Bingo quadruped using
Blender animations of that same robot. The animation is deployed in **Isaac Sim** first, as
close to the authored performance as possible, with the **intent of putting it on the real
robot** afterward.

`[STATED]` The near-term scope is narrower than the end goal: build the **retargeting
pipeline** and get the animations running on the robot in Isaac Sim. That is the current
deliverable.

---

## 2. Organisation — who owns what

`[STATED]` Three groups, in sequence:

| Group | Location | Role |
|---|---|---|
| **Design team + founders** | San Francisco | Made the initial robot design — the simple quadruped form. Own the industrial/visual design. |
| **Engineering team** | Berlin | Turned that design into the **CAD assembly** with real motors and housings. Own the mechanical assembly and the URDF exports. |
| **Hassaan + Neil** | Canada | Contracted after the CAD assembly existed. **Neil = mechanical**, future improvements. **Hassaan = the entire software stack**. |

`[STATED]` The animators are a separate party who authored the performance clips. They
animated against the **simple STL model** originating from the design team, not against the
Berlin CAD assembly.

`[MEASURED]` **However — §5.5 shows the geometry they animated is dimensionally identical to
the Berlin CAD assembly.** Whatever the provenance of the STL they were handed, it carried the
engineering assembly's exact dimensions.

---

## 3. The claim under dispute — this is the thing to verify

### 3.1 What happened

`[STATED]` While trying to deploy the Blender animations onto the robot, Claude (in an earlier
conversation) diagnosed that **retargeting could not be done** because the animation rig was
*bigger*, had *more joints*, and was structurally different from the engineering assembly.

`[STATED]` Hassaan relayed that diagnosis to the team and worked under that assumption. In
response, he built a Blender rig generated directly from the URDF
(`blend/conform/Bingo_ConformRig.blend`) to hand to the animator, so future animation would be
correct by construction.

`[STATED]` **He now believes that diagnosis was wrong.** After consulting the original
animators and receiving their latest model, the animation model appears **very close** to the
engineering assembly. The animators confirmed that although they referenced the simple STL
model, that STL was **very much the same as the engineering assembly**.

### 3.2 The two problems to solve

`[STATED]` **Problem 1.** If the animation rig is genuinely based on the actual robot and only
differs in minor ways, then retargeting **should work**, and the premise that blocked it was
false.

`[STATED]` **Problem 2.** This must be **verified properly this time**, so the same mistake is
not made twice — and to establish whether the robot is actually ready to be trained on mocap
data.

### 3.3 Status of the original diagnosis

> **RESOLVED 2026-08-21 — see §5.5.** The animation model was measured directly against the
> STEP CAD assembly and matches it **exactly** (1 Blender unit = 10.0000 mm, agreeing to 0.003%
> on two independent axes). The "bigger / different model" diagnosis is **disproved**. What is
> genuinely off is narrower: the *bone pivots* inside that correct model sit ~22% too narrow
> laterally, with SP and knee matching within 3%.

`[UNVERIFIED]` No document on disk contains the "the animation rig is bigger / has more joints"
finding, the measurements behind it, or the reasoning. It does not appear in
[MEMORY.md](MEMORY.md) or [docs/BingoMocapPipelineSpec.md](docs/BingoMocapPipelineSpec.md).
**Treat it as an unsourced claim, not a finding.**

`[FROM DOC]` What the project documents *do* say is weaker and different: MEMORY.md §1 states
the art rig "is **not** the robot" and that Path A is "Lossy: amplitude had to be cut to
50–70%", and §8 says "The old rig is a Rigify character rig built around the art model; it was
never derived from the URDF, which is why Path A is lossy." These are statements about the rig
not being *derived from* the URDF — which is true regardless — not about it being oversized.

`[MEASURED]` **A caution about one number that may have caused the confusion.** The
0.16 / 0.35 ≈ 0.457 leg-length ratio in spec §A.2 and §B.1.5 refers to the **Peng et al. dog
mocap clips** (a real dog, 0.35 m leg), *not* to the animators' Blender rig. If that figure was
read as describing the animation rig, it would produce exactly the mistaken "the rig is more
than twice the size" conclusion. The animators' rig has its own, much closer proportions —
measured in §5.2 below.

---

## 4. Asset inventory

### 4.1 On disk now

| Path | What it is | Provenance |
|---|---|---|
| `bingo_urdf v4_w_ear_joints/` | **Latest engineering URDF**, with ear joints. To be used in Isaac Sim. | `[STATED]` this is the latest |
| `bingo_urdf_rev_3/` | Previous URDF revision. All existing pipeline work targets this. | `[FROM DOC]` |
| `BS Bingo_Final_Export.stp` | 70 MB STEP CAD assembly. Header: generated by HOOPS Exchange 25.8.0, `D:\Projects\BS Bingo\BS Bingo_Final_Export.stp`, dated **2026-05-13**, author `cbres`. | `[STATED]` this is the model recently sent |
| `blend/legacy_art_rig/` | The animators' rig + **the 6 expressive performances**: DeadPan, Laidback, Cheeky, Enthusiastic, Eccentric, Timid, plus `BingoRig_Latest.blend` and `Bingo_MasterScene_01.blend`. All dated **Jul 22**. | `[STATED]` these are the animation clips |
| `blend/conform/Bingo_ConformRig.blend` | The URDF-derived rig Hassaan generated as a workaround. | `[FROM DOC]` |
| `raw/deadpan_raw.json`, `raw/laidback_raw.json` | World-space Cartesian bakes of the animators' rig. **These contain the rig's actual measured geometry** and are the evidence base for §5.2. | `[MEASURED]` |
| `motions/*.npz` | Packed motion files from previous retarget attempts. | `[FROM DOC]` |
| `scripts/` | The existing pipeline (see [README.md](README.md)). | — |

### 4.2 Not on disk / not confirmed present

- `[UNKNOWN]` **The simple STL model the animators used as reference**, as a standalone file.
  It may be embedded inside the `.blend` files; this has not been checked. This asset matters
  because the whole Problem-1 argument rests on "that STL was very close to the engineering
  assembly," and that comparison cannot be made without it.
- `[UNKNOWN]` Whether a USD conversion of the v4 URDF exists for Isaac Sim. Only rev_3 has one
  (`bingo_urdf_rev_3/urdf/bingo_scene_real_values_3/`).

---

## 5. Measured facts (computed this session)

### 5.1 v4 URDF vs rev_3 URDF

Method: parsed both URDFs with `xml.etree`, compared every non-fixed joint's type, parent,
child, axis, limits, and origin, plus every link mass.

**The legs are completely unchanged.**

- `[MEASURED]` Every one of the 17 rev_3 joints is present in v4 with an **identical axis
  vector, identical parent, identical child, and identical position limits**. Joint origins are
  identical to within floating-point printing differences (max movement **0.00 mm**). No `rpy`
  changed.
- `[MEASURED]` **Consequence: every leg-side result from the rev_3 work carries over to v4
  unchanged** — the A.3 sign map, the FK, the reach analysis, the retargeter. The leg retarget
  does not need to be redone for v4.

**What did change:**

- `[MEASURED]` **4 new ear joints**, bringing movable joints from **17 → 21**:

  | Joint | Type | Parent | Limits |
  |---|---|---|---|
  | `l_ear_pitch` | `continuous` | `head_roll` | −3 … 3 |
  | `l_ear_roll` | `revolute` | `l_ear_pitch` | −1.5 … 0 |
  | `r_ear_pitch` | `continuous` | `head_roll` | −3 … 3 |
  | `r_ear_roll` | `revolute` | `r_ear_pitch` | 0 … 1.5 |

  Both ears hang off `head_roll`, so ear motion inherits the full head chain.
  `[MEASURED]` The ear axes are **not axis-aligned** — e.g. `l_ear_pitch` axis is
  `(0.0277, −0.9137, −0.4055)`. Every other joint on the robot uses a clean ±X/±Y/±Z axis.
  `[MEASURED]` The `*_ear_roll` limits are **asymmetric and mirrored** (left −1.5…0, right
  0…1.5).
  `[MEASURED]` The `*_ear_pitch` joints are declared `continuous` yet carry `−3…3` limits —
  contradictory, since a continuous joint is by definition unlimited. An importer may honour
  either.

- `[MEASURED]` ⚠️ **Every joint in v4 has `effort="0"` and `velocity="0"`.** rev_3 had 3.0 N·m /
  10 rad/s (legs) and 1.5 / 8 (head, tail). v4 is a raw export with placeholder zeros.
  `[FROM DOC]` MEMORY.md §2 already flagged rev_3's numbers as "probably placeholders —
  identical round numbers on every joint."
  **Decision taken `[STATED]`: carry rev_3's values forward so sim runs, and flag loudly that
  they are unverified placeholders.**

- `[MEASURED]` ⚠️ **`head_pitch_joint` upper limit reduced from +0.40 to +0.05 rad.** This is
  the only position-limit change anywhere in the robot — an **87% reduction in upward head
  pitch**. This directly constrains expressive head-lift, which matters for "cute". It is
  unknown whether this is a deliberate mechanical change or an export artefact.

- `[MEASURED]` Mass: **2.4781 kg → 2.4472 kg**. Per-link changes are confined to the head/ear
  region: `head_roll` 0.7949 → 0.7077, `head_pitch` 0.0000 → 0.0103, `origin` 0.2923 → 0.3024,
  new `l_ear_roll` / `r_ear_roll` 0.0180 each.
  `[MEASURED]` `l_ear_pitch` and `r_ear_pitch` links have **mass = 0**, matching the existing
  zero-mass problem MEMORY.md §2 noted for `head_pitch`/`head_yaw`/`tail_pitch` in rev_3.

### 5.2 The animators' rig vs the URDF — the actual measurement

This addresses Problem 1 directly. Method: `raw/deadpan_raw.json` is a bake of the animators'
rig (`Bingo_DeadPan.blend`, rig object `Bingo_Rig`) in **raw, unscaled Blender units**, and it
records `rest_bone_lengths` per leg. Compared against URDF segment lengths from spec §A.2.

**Leg chain topology matches exactly.** `[MEASURED]` The exported leg chain is
`sy → sp → kn → tip` — the same 3-joint-plus-tip structure as the robot's
`SY → SP → knee → shank tip`. Not more joints, not fewer.

**Scale.** `[MEASURED]` Blender units here are arbitrary, so "bigger" is not meaningful on its
own. A single uniform scale of **0.009972 m per Blender unit** maps the rig's front-leg chain
onto the URDF's. The meaningful question is whether *proportions* match under that one scale.

**Proportions** (front leg, each segment as a share of the `sy+sp+kn` chain):

| Segment | Rig | URDF | Relative difference | Implied per-segment scale |
|---|---|---|---|---|
| `sy` | 16.3 % | 21.0 % | **−22.4 %** | 0.012856 |
| `sp` (thigh) | 31.5 % | 32.5 % | **−3.1 %** | 0.010288 |
| `kn` (shank) | 52.3 % | 46.6 % | **+12.2 %** | 0.008885 |

`[MEASURED]` If the rig were a perfect uniform scale of the robot these three per-segment
scales would be identical. They are not: **max/min = 1.447**. So the rig is *not* a uniform
scale of the robot, but the deviation is a **moderate proportion mismatch concentrated in the
shortest segment (`sy`) and the shank**, with the thigh matching to 3%.

`[MEASURED]` Front vs back legs are near-identical on the rig (`sp` 8.136 vs 8.085,
`kn` 13.505 vs 13.605 Blender units).

`[MEASURED]` **This is consistent with Hassaan's revised position and inconsistent with the
original "much bigger, many more joints" diagnosis** — at least for the leg chain.

**Extra bones beyond the robot's DOF.** `[MEASURED]` The bake also exports `spine_front`,
`spine_back`, `neck`, `head`, `tail1`, `tail2`. The robot has **no spine and no neck joint**
(spec §A.2, §B.5.2). `head`/`tail1`/`tail2` have robot counterparts. So the "extra joints" part
of the original claim is **partly true — a 2-segment spine and a neck** — but this is the
normal, expected difference between a character rig and a robot, and the retargeter already
drops it.

⚠️ **Critical caveat on §5.2.** `[MEASURED]` These are **only the bones `bake_motion.py` chose
to export**. A Rigify rig typically carries many more bones (IK, poles, `MCH-*`, deform). This
data **cannot prove** the rig has no other bones, and it says nothing about the *mesh* the
animators used. It proves only that the exported leg chain is structurally right and
proportionally close.

### 5.5 Animation model vs the CAD assembly — DIRECT COMPARISON (2026-08-21)

This is the measurement that settles Problem 1. Method, all re-runnable:

- **STEP**: `BS Bingo_Final_Export.stp` parsed as raw ISO-10303-21 text. It is a **flattened
  single-product assembly** — 1 `PRODUCT`, 1 `AXIS2_PLACEMENT_3D`, 18 `MANIFOLD_SOLID_BREP`,
  **0** `NEXT_ASSEMBLY_USAGE_OCCURRENCE` — so all geometry is already in one global coordinate
  system and a bounding box over its 755,071 `CARTESIAN_POINT`s is meaningful. Units:
  `SI_UNIT(.MILLI.,.METRE.)` = **millimetres**.
- **URDF v4**: assembled at zero pose by forward kinematics, each link's STL transformed into
  world space.
- **Animation model**: `Bingo_DeadPan.blend` opened headless in Blender 5.2, armature forced to
  **REST** position, world-space bounds taken over only the meshes parented to `Bingo_Rig`
  (excludes `Stage`, camera, and control objects).

**Result 1 — the STEP and the v4 URDF are the same machine.** `[MEASURED]`

| | width | other | other |
|---|---|---|---|
| STEP assembly | **179.334 mm** | 370.733 | 394.251 |
| URDF v4 @ zero pose | **179.283 mm** | 365.0 | 403.0 |

The width axis agrees to **0.05 mm (0.03%)**. The other two axes differ because the two are in
different poses. **Consequence: the URDF's STL meshes are a valid proxy for the CAD**, so the
part-level comparisons below are comparisons against the engineering assembly.

**Result 2 — the animation model IS the CAD geometry, exactly.** `[MEASURED]`

Animation model rest-pose bounds: **17.934 × 37.072 × 50.265 Blender units.**

| Axis | Animation (BU) | STEP (mm) | mm per Blender unit |
|---|---|---|---|
| width | 17.934 | 179.334 | **10.0000** |
| length | 37.072 | 370.733 | **10.0003** |
| height | 50.265 | 394.251 | 7.844 — *pose differs, see below* |

**Two independent axes land on exactly 10 mm per Blender unit, to within 0.003%.** That is not
a coincidence and not a similarity — the animation model is the engineering CAD, imported at
1 Blender unit = 1 cm. The height axis disagrees only because the rig's rest pose hangs the
legs lower than the pose the CAD was exported in; height is the one pose-dependent extent.

`[MEASURED]` Corroborating detail: individual parts match once you account for the URDF
combining shank + paw into a single `*_knee` link (0.1743 m) while the animation model splits
them into `lowerleg_*` (0.1534 m) + `foot_*`. The ~21 mm difference is exactly the paw drop
MEMORY.md §2 documents as "the paw mesh hangs 28–30 mm below the shank tip."

`[MEASURED]` The mesh part names are mechanical, not organic — `hip_ball`, `shoulder_ball`,
`neck_rod`, `tail_rod`, `tail_closeout`, `nose_cap`, `upperleg_*`, `lowerleg_*` — consistent
with parts imported from a CAD assembly rather than a sculpted dog.

**Conclusion: the original "the animation rig is bigger / a different model" diagnosis is
disproved.** The model is dimensionally identical to the engineering assembly.

**Result 3 — but the SKELETON inside that model is not on the robot's joint axes.** `[MEASURED]`

Using the established 10.000 mm/BU scale, comparing pivot positions (rig `def_*` bone heads at
rest vs URDF joint origins by FK):

| Pivot | Measure | Anim rig | URDF v4 | Diff |
|---|---|---|---|---|
| SP (hip pitch) | front-to-back spacing | 172.0 mm | 172.0 mm | **0.0%** |
| SP → knee | segment, front | 81.4 mm | 83.6 mm | **−2.7%** |
| SP → knee | segment, back | 80.8 mm | 83.6 mm | **−3.3%** |
| knee | front-to-back spacing | 172.7 mm | 177.4 mm | −2.6% |
| SP (hip pitch) | left-right separation | 108.7 mm | 138.0 mm | **−21.2%** |
| knee | left-right separation | 127.2 mm | 165.4 mm | **−23.1%** |
| SY → SP | segment | 42.0 mm | 54.2 mm | **−22.5%** |
| SY (abduction) | left-right separation | 24.7 mm | 44.0 mm | **−43.9%** |
| SY (abduction) | front-to-back spacing | 172.0 mm | 118.1 mm | **+45.7%** |

Read that as a pattern, not a list: **everything in the fore-aft and vertical plane matches
within ~3%** — the SP-to-SP spacing is *exact* — while **everything lateral is systematically
~22% narrow**, and the SY pivot is placed quite differently (the rig puts it directly inboard
of SP; the real robot insets it 26.9 mm fore-aft and sits it further outboard).

`[MEASURED]` **The lateral narrowness is structural, not a rest-pose artefact.** The rig's
SY pivot sits 12.34 mm off centreline with a 42.0 mm purely-lateral bone to SP, so SP can reach
at most 54.35 mm lateral at any SY angle — it cannot reach the URDF's 69.0 mm by rotation. No
choice of rest pose fixes this.

**Interpretation.** The animator imported the correct CAD parts and placed them correctly — the
overall width is exact — then placed bones inside them by eye, landing inboard of the true
motor axes. So the *model* is Bingo; the *rig* is an approximation of Bingo's kinematics.

**What this means for retargeting.** This is a **minor, correctable discrepancy, not a
blocker**, and it is favourably distributed:

- The **SP and knee joints carry essentially all the locomotion and posture content**, and they
  match within 3%.
- The error is concentrated in **SY (abduction)** — the joint with the smallest range on the
  robot (±0.42 rad) and which the policy additionally caps ×0.3 to ≈±0.126 rad. It is the least
  consequential axis to be wrong on.
- A retargeter that solves for **foot-tip position** (spec §B.3 Stage 3) rather than copying
  joint angles absorbs this mismatch by construction.

### 5.3 Frame rate

`[MEASURED]` `raw/deadpan_raw.json` records `authored_fps: 24`, resampled to `sample_hz: 120`,
1876 frames.
`[FROM DOC]` Spec §B.1.7 requires authoring at **≥ 60 fps, 120 preferred**, with no retiming or
keyframe reduction, because velocities are finite-differenced. **Animation authored at 24 fps
and upsampled to 120 Hz does not meet the spec's own requirement**, and upsampling does not
recover the missing information.

### 5.4 Axis convention

`[MEASURED]` The animators' rig is **−Y forward, +Z up, character LEFT is +X**
(`axis_note` in the bake). The robot is **+X forward, +Y left, +Z up** (spec §B.1.4). A
conversion is required and `bake_motion.py` already records that Stage 2 owns it.

---

### 5.6 Can the six existing clips be retargeted and trained? (2026-08-21)

Method: every clip in `blend/legacy_art_rig/` baked headless (`scripts/bake_motion.py`, 120 Hz)
and retargeted with the existing `scripts/retarget.py` **against the v4 URDF**. Outputs kept in
`motions/v4_retarget_test/` and `raw/v4_retarget_test/`.

**First: the rig and pipeline are structurally healthy.** `[MEASURED]`

- All six clips carry real animation — ~81 s total, all authored at **24 fps**.
- The rig is **linked** from `BingoRig_Latest.blend`; each performance is a **library override**
  with its own action. One rig, six actions — a clean setup. (Note for tooling: this means two
  objects are named `Bingo_Rig`; the animated one is the override, the linked one is inert.
  Code that does `bpy.data.objects['Bingo_Rig']` is ambiguous and must select the override.)
- **The ears are animated** in all six (`Anim_Ear.L/R`, `ctrl_Ear.L/R`, `def_Ear.L/R`) — content
  that v4's 4 new ear joints can now actually drive, and which the 12-DOF schema discards.
- `retarget.py` runs against **v4** unmodified and all six produced valid A.7 `.npz`.
- **All six pass the foot-slip gate** (<5 mm/stance) and the **velocity limit** (max 9.3 rad/s
  vs the 10 rad/s limit, **zero** frames over) after the contact-consistent root solve.

**Second: the blocker is joint range, not geometry.** `[MEASURED]`

The robot's foot cannot come closer than **147.0 mm** to its own hip (SP) pivot — that floor is
set by the ±1.56 rad knee limit (max reach is 203.6 mm). The animations demand far less:

| Clip | % of frames demanding a reach the knee cannot make (fl / fr / bl / br) | Closest demanded |
|---|---|---|
| **Eccentric** | **0.0 / 0.0 / 0.0 / 0.0** | 181 mm |
| **DeadPan** | **0.6 / 0.0** / 38.4 / 38.3 | 121 mm |
| Cheeky | 29.9 / 31.6 / 17.1 / 10.4 | 90 mm |
| Enthusiastic | 17.4 / 23.9 / 33.3 / 34.9 | **65 mm** |
| Laidback | 22.3 / 5.2 / 62.3 / 65.4 | 77 mm |
| Timid | 42.4 / 44.8 / 49.7 / 54.4 | 82 mm |

**Cause: the animation rig has no joint limits.** The animator folded the legs to whatever read
well; the real knee stops at ±89°. This is exactly the constraint spec §B.1.6 says must be
authored as hard rig constraints so violations are *impossible to create*.

**Third: what the retargeter does about it — and why the clips look weaker than authored.**
`[MEASURED]` `retarget.py` searches amplitude downward (1.00 → 0.50) until limit-clamping drops
below 0.5%. **0.50 is the floor of that search, not a chosen value** — so an amplitude of 0.5 in
the log means *"even at half amplitude the gate never cleared."*

| Clip | Amplitude applied | Frames clamped at a limit after retarget | IK foot error (mean) | Duty factor |
|---|---|---|---|---|
| **DeadPan** | 0.5 front / 0.7 back | **fr_knee 1.4%** — clean | **0.01 mm** | 0.86–0.93 |
| Eccentric | **1.0** | none >0.5% | 11.2 mm | fl/fr 0.75, **bl/br 0.00** |
| Cheeky | 0.5 (floor) | fl_knee 17.5%, fr_knee 5.0% | 5.9 mm | **0.20–0.30** |
| Enthusiastic | 0.5 (floor) | all four knees 10–20% | 6.4 mm | 0.67–0.76 |
| Laidback | 0.5 (floor) | fl_knee 35.3%, fr_SY 13.4% | 1.3 mm | 0.76–0.97 |
| Timid | 0.5 (floor) | all four knees **39–48%** | 6.5 mm | 0.79–0.94 |

Two clips fail for reasons unrelated to limits: **Eccentric** has duty factor **0.00 on both
hind legs** — the hind feet never touch the ground, it is a sustained rear-up, a balance problem
rather than a tracking one. **Cheeky** has duty 0.20–0.30, i.e. airborne 70–80% of the time — a
bound, and on a 2.478 kg robot with 3.0 N·m joints almost certainly infeasible (untestable
until real motor data arrives, see §9.4).

**Fourth: independent confirmation of the §5.5 pivot finding.** `[MEASURED]` The retargeter
reports a **constant base-fit residual of 23.4–24.9 mm** on every clip, labelled in its own
output as "hip-layout mismatch." That is the same lateral pivot offset §5.5 measured
geometrically (~15–19 mm per side). Two independent methods, same conclusion.

**Answer to "can we easily train this?"**

- **Retargeting is not the blocker and geometry is not the blocker.** The pipeline runs
  end-to-end on v4 today, and the model is dimensionally identical to the CAD (§5.5).
- **DeadPan is usable now** — 0.01 mm IK error, 1.4% clamping, clean velocities.
- **The other five need work**, and the work is *upstream in the animation*, not in the
  retargeter: the poses ask for knee flexion the robot does not have.
- **This is the strongest argument yet for the conform rig** (`blend/conform/`). Its hard
  Limit Rotation constraints make an unreachable pose impossible to author, which is precisely
  the one real defect found. §8's "undecided" status can now be decided on evidence: keep the
  animator's *model* (it is correct), give her a rig that *cannot* exceed the robot's limits.

---

### 5.7 Engineering clarification on model currency (2026-08-21)

`[STATED]` The Berlin engineers clarified, verbatim:

> "the assembly you're working with is the most up-to-date from an engineering standpoint.
> however, the model you have has an old version of the tail, whereas the updated design has a
> slot here, and the tail in this version has 2 ranges of motion versus 1 in the old version.
> also, the leg joints in the new model are slightly less bulging than the old bone-like ones.
> let us know if it can affect the work at the current stage. we are going to update the CAD
> assembly to fully reflect those design changes. the blender animation rig that Ashley provided
> was done for visuals and shouldn't be used as a reference for your area of work."

**Impact assessment:** `[MEASURED]`

- **Tail — no impact on the URDF.** Both rev_3 and v4 already declare **2 tail DOF**
  (`tail_pitch` about −Y, `tail_yaw` about −X). The URDF already reflects the new 2-range tail;
  only the STEP *geometry* carries the old single-range tail. Nothing in the kinematic pipeline
  needs to change.
- **Leg-joint bulge — cosmetic.** A change to mesh surface, not to joint origins or axes. It
  affects visual meshes and, marginally, collision geometry — **not** link lengths, joint
  positions, or limits, which are what the retarget consumes.
- **⚠️ When the updated CAD lands, re-run the §5.5 comparison.** The current "exact match"
  conclusion is against *this* STEP. A revised tail changes the fore-aft extent, which is one of
  the two axes that produced the 10.0000 mm/BU result.
- **"Ashley's rig was for visuals, don't use it as a reference" — independently confirmed by
  measurement.** §5.5 found exactly this split: her *model* is dimensionally the CAD (exact),
  but the *bone pivots inside it* are ~22% narrow laterally and the SY pivot is misplaced. The
  engineers' instruction and the measurement agree. **The model is trustworthy; the rig is not.**

---

### 5.8 The conform rig actually works — Ashley's walk cycle measured (2026-08-21)

`[STATED]` Ashley reported problems with `Bingo_ConformRig.blend`: feet sinking into the ground
when she moves the body, inability to move the character up and down, and "legs do funny
things." She asked whether her own rig could be kept and "additional proper bones" added to bake
from. She delivered a test walk cycle: `Bingo_Walking_04 (1).blend`.

**Measured result of that walk cycle** (`scripts/bake_conform.py`, 120 Hz, 499 frames / 4.16 s):

| Metric | Value | Verdict |
|---|---|---|
| Authored frame rate | **60 fps** | ✅ meets spec §B.1.7 (her other clips are 24 fps) |
| Joint range used | **68–96% headroom** on every joint (max 0.505 of 1.56 rad) | ✅ nothing near a limit |
| Frames clamped at a limit | **zero** | ✅ vs 39–48% on her own rig (§5.6) |
| Duty factor | 0.57 / 0.62 / 0.57 / 0.64 | ✅ a real walk |
| L/R foot travel symmetry | 17.2 / 19.0 / 18.9 / 17.2 mm — within **1.8 mm** | ✅ well under the 10 mm gate |
| Foot-tip ground penetration | none (min tip z **+23.9 mm**) | ✅ |
| Paw ground penetration | **−4.9 mm** (tip +23.9 less the 28.8 mm paw drop) | ⚠️ minor |
| Foot vertical travel | 17–19 mm | ❌ spec gate wants 30–60 mm — steps too shallow |
| Root height variation | **exactly 0.0000 m** | ❌ spec gate wants 10–30 mm of bob |
| Max joint velocity | 20.85 rad/s | ❌ limit is 10; **19 frames (3.8%)** over, in 6 short bursts |

**This is the best motion data in the project.** Compare §5.6: her own rig demanded knee flexion
the robot cannot reach on 39–48% of frames; on the conform rig she used at most 32% of the
available range.

**Root cause of every complaint she raised — a single, correctable mistake.** `[MEASURED]`

She animated the body at a root height of **0.2265 m**. The maximum physically possible root
height, with the paw touching z = 0 and the legs *perfectly straight*, is **0.2273 m**. She was
working at **99.7% of full leg extension** — 27.6 mm above the 0.1989 m crouch the rig ships
with.

That single fact explains all three symptoms:

1. **"I can't move the character up"** — she was already 0.8 mm from the mechanical maximum.
2. **"Legs do funny things"** — a fully-extended leg is the **straight-leg singularity**, which
   spec §B.1.3 explicitly warns "never occurs in real motion." IK is ill-conditioned there and
   pops. The 6 velocity bursts (16–21 rad/s, all in the knees) are those pops.
3. **"Feet sinking into the ground"** — at full extension the paw sits 4.9 mm below z = 0,
   because the paw mesh hangs 28.8 mm below the shank tip (MEMORY.md §2).

**None of these is a rig bug.** They are the robot's real kinematics, encountered because the
work started from the extended pose rather than the crouch. Spec §B.1.3 already states the rule:
*"Animators should be told the crouch is the neutral, and zero is a straight-leg singularity
that never occurs in real motion."* That instruction did not reach her.

---

## 5A. Strategy: how much should we depend on animators? (decided 2026-08-21)

`[STATED]` Concern raised: relying on animators to hand-author motion that exactly matches the
robot means they do most of the work, which defeats the point of using RL. Reference:
[mixamo-llm-mocap](https://github.com/squall01337/mixamo-llm-mocap) (video -> GVHMR pose
estimation -> FK retarget onto Mixamo humanoids; deterministic, no physics, humanoid-only).

**The boundary, stated precisely.**

RL *does* absorb, so verbatim animation is NOT required:
- morphology mismatch (different limb proportions, different skeleton)
- a physically invalid reference (floating, sliding, dynamically impossible)
- style vs. exact trajectory — an adversarial discriminator matches a *distribution*
- a different species entirely (the v9 result came from **dog** mocap)

RL *cannot* absorb — this is physics, not learning:
- **joint limits** (knee stops at ±1.56 rad; a reference needing more gets something else)
- **torque** (3.0 N·m on 2.478 kg)
- **missing DOF** (no spine, no ankle)

**The actual diagnosis.** Every failure measured in §5.6 — 39–48% clamping, amplitude auto-halved,
knees crossing zero — comes from treating animation as *a trajectory to reproduce* (tracking,
which demands a feasible reference) rather than *a style distribution to match* (AMP, which
degrades gracefully). The project already has AMP; it has been used as if it were a tracker.

**Decided architecture — split the robot in two:**

| Layer | DOF | Needs | Animator dependency |
|---|---|---|---|
| **Expression** | head 3 + tail 2 + ears 4 = **9** | almost nothing — generous limits, low inertia, no ground contact, no balance coupling | can be played near-verbatim from any source |
| **Locomotion** | 12 leg | RL + feasibility filtering | **none** — dog mocap, procedural gaits, or generated video |

This decoupling is the answer to the dependency question: expression is where "cute" lives and
it is cheap and forgiving; locomotion is hard but fully automatable without an animator.

**Longer term** — [Uni-Mo, arXiv 2606.28237](https://arxiv.org/html/2606.28237) is the closest
published match: LLM proposes prompts -> fine-tuned video diffusion generates video *of the target
robot* -> ViTPose keypoints lifted to 3D -> PPO tracking, with a tracking-error gate discarding
what will not track. 7,488 synthetic motions, 96.7% real-robot success on a Unitree Go2. It
sidesteps morphology mismatch by never leaving the robot's own morphology. Its stated *limitation*
— biased toward in-place expressive behaviour rather than traversal — **is exactly this project's
target**. Note it still bootstraps from **190 designer-choreographed sequences**: the animator is
not eliminated, she is amplified ~40x, and her output becomes seed data rather than the
deliverable. Caveat: the Go2 is ~15 kg with far stronger actuators; Bingo's feasibility gate will
reject considerably more.

**Order of work:** (1) extend the schema 12 -> 21 DOF; (2) run the existing clips through AMP as
*style* references, not tracking targets; (3) have the animator prioritise head/ears/tail
performance over leg precision; (4) revisit generative expansion only once 1–3 work.

---

### 5A.1 Implemented: the 21-DOF expression schema (2026-08-21)

Step 1 of the above is **done on the data-production side**. The RL training code
(`bingo_rl/`) is **not in this repo** — it lives on the remote box — so the training-side change
(obs dims, `robot_ctrl_indexes`) still has to be made there.

**Canonical 21-DOF order — a strict extension of the 12-DOF one:**

```
 0-11  fl/fr/bl/br x (SY_J, SP_J, knee)      <- unchanged, same indices
12-14  head_pitch_joint, head_yaw, head_roll
15-16  tail_pitch, tail_yaw
17-20  l_ear_pitch, l_ear_roll, r_ear_pitch, r_ear_roll
```

Indices 0–11 are untouched, so **a 12-DOF consumer slicing the first 12 entries stays correct
against a 21-DOF file.** Verified numerically, not assumed.

**Changes made:**

| File | Change |
|---|---|
| `scripts/bake_conform.py` | `--dof 12\|21` flag (**default 12**, so nothing breaks). Emits `dof_names`/`dof_positions`/`dof_velocities` at the chosen width, plus a new `ear_positions` key always. Rigs lacking ear bones bake those channels as zeros with a printed note. |
| `scripts/build_rig.py` | Ear joints added to the bone order; `MARGIN` extended with ear limits (the `*_ear_roll` limits are **asymmetric and mirrored**, so they are tuples, not symmetric scalars). Joints absent from an older URDF are dropped with a note, so rev_3 still builds. |
| `scripts/build_rig.py` | **`package://` URI support.** rev_3 used relative mesh paths; v4 uses ROS package URIs, which silently produced a rig with **zero meshes**. `resolve_mesh()` handles both, resolving against the nearest ancestor holding `package.xml`. |
| `scripts/check_rig.py` | Expected mesh count now derived from the URDF instead of hardcoded `18` (v4 has 22). |
| `scripts/test_ik.py` | **Fixed a long-standing false failure.** The "forward" case asked for 60 mm of forward reach; only ~35 mm exists at the 0.199 m stance (MEMORY.md §3 — the envelope is strongly asymmetric, ~35 mm forward vs ~188 mm back). All four legs "failed" by 11.2 mm on every run while off-axis stayed 0.00000 deg and limits held — i.e. the IK was clamping correctly and the *test* was wrong. Now 30 mm. |

**New artefact:** `blend/conform/Bingo_ConformRig_v4.blend` — built from the v4 URDF.
**34 bones** (30 + 4 ear), **22 meshes**, axis alignment worst **0.0000 deg**, stance 0.1989 m.

**Verification performed:**

- `check_rig.py`: **ALL CHECKS PASSED** on the v4 rig; rev_3 rig still passes (no regression).
- `test_ik.py`: **33/33 PASS** on both rigs.
- Byte-identical regression: re-baking Ashley's walk at `--dof 12` reproduces the pre-change
  `.npz` exactly (`dof_positions`, `dof_velocities`, `body_positions`, `body_rotations`,
  `contacts`, `dof_names` all `array_equal`).
- Extension guarantee: the 21-DOF file's first 12 columns are `array_equal` to the 12-DOF file.
- End-to-end expression test: all 9 channels driven on the v4 rig round-trip through the bake.

**⚠️ Finding from that end-to-end test — v4's head limit is brutally tight.** An authored head
pitch of +0.30 rad comes back as **+0.05**: **83% of the head-lift is clipped**, because v4
reduced `head_pitch_joint`'s upper limit from +0.40 to +0.05 (§5.1). Head-lift is a primary
"alert / happy / curious" cue, so this directly caps expressiveness. **Confirm with Berlin
whether +0.05 rad is a real mechanical change or an export artefact** — it is the single most
expressive-motion-limiting number in the v4 URDF.

Ashley's existing walk cycle carries **no** head/tail performance — those bones are keyed but
never moved off zero. The expression channels are currently empty content, not just an empty
schema.

---

## 6. What has NOT been verified — the actual work for Problem 2

Ordered by how much each would change the plan.

1. ~~**The mesh comparison has not been done.**~~ **DONE — §5.5.** The animation model is the
   CAD assembly, exactly, at 1 BU = 10 mm.
2. ~~**The full bone list of the animators' rig is unknown.**~~ **DONE.** `Bingo_DeadPan.blend`
   has **55 bones** in `Bingo_Rig`, in three families: `def_*` (the deform chain that gets
   baked), `ctrl_*` (animator controls, including `ctrl_Knee_Pole` / `ctrl_Elbow_Pole` IK poles
   and `ctrl_Root`), and `Anim_Ear.L/R`, plus one stray bone named `Bone`. The baked leg chain
   maps `def_Shoulder`→`sy`, `def_Arm`→`sp`, `def_ForeArm`→`kn`, `def_Hand`→`tip` (front) and
   `def_Hip`/`def_Leg`/`def_Shin`/`def_Foot` (back). Confirmed by bone lengths matching
   `rest_bone_lengths` in `raw/deadpan_raw.json` exactly. **Still to check: the other 5 blends.**
3. **Joint axes of the animators' rig were never compared to the URDF axes.** Proportions
   matching does not mean rotation axes match. This is the failure mode the docs call the
   "one leg bends backwards" bug (spec §A.3, §B.1.3).
4. **Whether the rig's rest pose corresponds to the URDF zero pose** (spec §B.1.3). If not, an
   offset table is needed and joint values cannot be read directly.
5. **Whether the 6 performances are physically feasible at all.** `[FROM DOC]` MEMORY.md §5
   says only **DeadPan** retargets cleanly (0.01 mm IK error, 1.4% clamping); **Laidback fails**
   (crouches below the knee limit, 35% of frames clamped); **Cheeky/Enthusiastic are bounds
   with flight phases**; **Eccentric is reared with hind feet never contacting**. These
   findings are about the *robot's* limits, not about rig mismatch, so **fixing the retarget
   will not fix them.**
6. **Nothing has been run against v4 yet.** Every existing motion file, script default, and
   result targets rev_3.
7. **No torque feasibility check exists** `[FROM DOC]` (MEMORY.md §6) — and v4 supplies zeros,
   so it still cannot be done from the URDF alone.

---

## 7. Constraints that cap what is achievable

`[FROM DOC]` These are robot properties, independent of any rig question, from spec §B.5 and
MEMORY.md §3:

1. **No ankle, paw, or toe joint.** The knee is the last actuated joint; contact is the shank
   tip. Paw roll, toe-off, and heel strike cannot be reproduced.
2. **No spine.** Bounding and galloping will read as stiff no matter what.
3. **`SY` is only ±0.42 rad**, and the policy caps its authority ×0.3 (≈±0.126 rad).
   `[FROM DOC]` MEMORY.md §6 notes IK already spends up to 0.30 rad holding feet planted —
   an unresolved overrun.
4. **The paw mesh hangs 28–30 mm below the shank tip** — resting the tip on z=0 buries the paw.
5. **Foot reach is the biggest authoring constraint.** At the default 0.199 m stance each paw
   is already ahead of its own hip (front +103 mm, back +44 mm), leaving only **35 mm of
   forward travel** versus 188 mm backward. Rule for animators: **sweep paws backward, not
   forward.**
6. **The motion schema is 12 DOF (legs only)** `[FROM DOC]`. Head and tail are baked but
   ignored; ears do not exist in it at all. **For expressive work this is the binding
   constraint** — with v4 the robot has 3 head + 2 tail + 4 ear = **9 expressive DOF that the
   pipeline currently discards.** Extending 12 → 21 is described as a contained change
   (spec §B.5.6).

---

## 8. Decisions already taken

- `[STATED]` **Torque/velocity limits:** carry rev_3's values (3.0 N·m / 10 rad/s legs,
  1.5 / 8 head+tail) forward into v4 work so simulation runs, and record clearly that they are
  unverified placeholders.
- `[STATED]` **The conform rig (`blend/conform/Bingo_ConformRig.blend`) is undecided** — keep
  or abandon based on what verification actually measures. Do not assume either way.

---

## 9. Open questions

1. `[UNKNOWN]` Do the expressive clips need to **translate the robot** (locomotion), or are they
   **in-place gestures** layered on a locomotion policy? Spec §B.7.5 flags this as unsettled,
   and it changes the training problem substantially. MEMORY.md §5 indicates the existing 6 are
   a mix — some are bounds with flight phases.
2. `[UNKNOWN]` Is `head_pitch_joint`'s reduction from +0.40 to +0.05 rad intentional? Ask
   Berlin. It constrains expressive head-lift.
3. `[UNKNOWN]` Are the ear joints' odd definitions — non-axis-aligned axes, `continuous` type
   carrying limits, zero-mass pitch links — deliberate or export artefacts? Ask Berlin.
4. `[UNKNOWN]` What are the **real motor specs**? Blocks all feasibility analysis.
5. `[UNKNOWN]` Does a USD conversion of v4 exist for Isaac Sim, and if it is regenerated, will
   it reproduce the drive-gain problem spec §A.5 documents (CLI conversion produced drives ~55×
   weaker than the GUI import)?
6. `[UNKNOWN]` Will the animators re-author at ≥ 60 fps, or must the existing 24 fps material be
   used as-is?
7. `[UNKNOWN]` Can the animators supply **per-foot contact flags**? Spec §B.1.8 / §B.3 Stage 4:
   contact is what makes root motion physically consistent. Currently inferred from paw height
   and speed.

---

## 10. Instruction to the verifying agent

Do not accept §3.1's original diagnosis, and do not accept its reversal either. §5 establishes
that the **leg chain is structurally correct and proportionally close (within ~12–22% per
segment)**, and that **v4 changed nothing about the legs**. That is real but partial evidence:
it covers the exported leg bones only.

The open items in §6 — the mesh comparison, the full bone list, and the **joint-axis
comparison** — are what would actually settle whether these animations can be retargeted 1:1.
Measure them from the files. Report numbers, not conclusions, and state clearly what each
measurement does and does not cover.
