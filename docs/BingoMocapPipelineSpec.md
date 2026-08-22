# Bingo — Simulation Setup & 1:1 MoCap Training Specification

**Purpose.** Two things in one document:

1. **Part A — what exists today**: the exact robot model, joint conventions, sim rates, observation/reward design, and motion-data schema currently driving AMP training. Anyone touching the rig, the mocap capture, or the retargeter needs these numbers.
2. **Part B — what a 1:1 mocap pipeline requires**: how the animation/mocap rig must be built (bones, axes, limits, units, export), what has to be captured, what code we still need to write, and what the training loop must change to go from *style imitation* (what AMP does now) to *1:1 motion tracking*.

Scope note: this document is about **skeleton / joint / motion-data requirements**. It deliberately says nothing about the contents of any specific art file — only about the structure a rig must have to be trainable.

---

# PART A — CURRENT SIMULATION SETUP

## A.1 Robot asset

| Item | Value |
|---|---|
| Source URDF | `bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf` |
| Sim asset (USD) | `bingo_urdf_rev_3/urdf/bingo_scene_real_values_3/bingo_scene_real_values_3.usd` |
| Isaac Lab cfg | `bingo_rl/bingo_rl/improved_walking_cfg.py` → `BINGO_IMPROVED_CFG` |
| Base / root link | `origin` |
| Total mass | **2.478 kg** (base 0.292 kg; each thigh 0.163 kg, each shank 0.153 kg) |
| Total DOF | **17** (12 leg + 3 head + 2 tail) |
| Actuated in RL today | **12** (legs only; head/tail pinned at default) |
| Nominal standing base height | **0.19–0.20 m** |

`rev_1` is the legacy import and exists only because its GUI-imported drive gains were copied forward (see A.5). **All new work uses rev_3.**

## A.2 Kinematic tree

```
origin (floating base)
├── fl_SY_J  → fl_shoulder_yaw  → fl_SP_J → fl_shoulder_pitch → fl_knee → fl_knee(link)   [+ virtual shank tip]
├── fr_SY_J  → fr_shoulder_yaw  → fr_SP_J → fr_shoulder_pitch → fr_knee → fr_knee(link)   [+ virtual shank tip]
├── bl_SY_J  → bl_shoulder_yaw  → bl_SP_J → bl_shoulder_pitch → bl_knee → bl_knee(link)   [+ virtual shank tip]
├── br_SY_J  → br_shoulder_yaw  → br_SP_J → br_shoulder_pitch → br_knee → br_knee(link)   [+ virtual shank tip]
├── head_pitch_joint → head_pitch → head_yaw → head_yaw(link) → head_roll → head_roll(link)
└── tail_pitch → tail_pitch(link) → tail_yaw → tail_yaw(link)
```

**3 DOF per leg: SY (abduction/adduction) → SP (hip pitch) → knee. There is no ankle, no toe, no paw joint, and no spine chain.** The knee child link (`*_knee`) is the terminal segment; the contact point is a fixed offset down that link.

### Link geometry (from URDF, metres)

| Segment | Vector (parent frame) | Length |
|---|---|---|
| `origin` → front SY | (+0.0591, ±0.0220, ~0) | — |
| `origin` → back SY | (−0.0590, ±0.0220, ~0) | — |
| SY → SP | (±0.0269, ±0.0470, 0) | 0.054 |
| SP → knee hinge (thigh) | (−0.0255, ±0.0137, −0.0785) | **0.0837** |
| knee hinge → contact point (shank) | (0, 0, −0.12) | **0.120** (`SHANK_LEN`) |
| Hip-to-hip fore/aft (wheelbase) | 0.0591 − (−0.0590) | **0.1181** |
| Hip lateral half-width (at SP) | 0.0220 + 0.0470 | **0.069** |
| Max leg reach, SP → contact | thigh + shank | **~0.204** |

For retarget scaling we currently use a leg-length ratio of **0.16 / 0.35** (Bingo / reference dog) ≈ **0.457**, applied to horizontal displacement and root bob.

## A.3 Joint axes, limits, and the sign map ⚠️

The rev_3 export has **inconsistent axis directions between left and right legs**. This is the single most error-prone fact in the project.

| Joint | Axis | Lower | Upper | Effort (N·m) | Vel (rad/s) |
|---|---|---|---|---|---|
| `fl_SY_J` | −1 0 0 | −0.42 | +0.42 | 3.0 | 10 |
| `fl_SP_J` | 0 +1 0 | −1.56 | +1.56 | 3.0 | 10 |
| `fl_knee` | 0 −1 0 | −1.56 | +1.56 | 3.0 | 10 |
| `fr_SY_J` | +1 0 0 | −0.42 | +0.42 | 3.0 | 10 |
| `fr_SP_J` | 0 **−1** 0 | −1.56 | +1.56 | 3.0 | 10 |
| `fr_knee` | 0 **+1** 0 | −1.56 | +1.56 | 3.0 | 10 |
| `bl_SY_J` | +1 0 0 | −0.42 | +0.42 | 3.0 | 10 |
| `bl_SP_J` | 0 +1 0 | −1.57 | +1.57 | 3.0 | 10 |
| `bl_knee` | 0 −1 0 | −1.57 | +1.57 | 3.0 | 10 |
| `br_SY_J` | −1 0 0 | −0.42 | +0.42 | 3.0 | 10 |
| `br_SP_J` | 0 +1 0 | −1.56 | +1.56 | 3.0 | 10 |
| `br_knee` | 0 **+1** 0 | −1.56 | +1.56 | 3.0 | 10 |
| `head_pitch_joint` | 0 −1 0 | −0.65 | +0.40 | 1.5 | 8 |
| `head_yaw` | −1 0 0 | −0.60 | +0.60 | 1.5 | 8 |
| `head_roll` | 0 0 −1 | −0.78 | +0.78 | 1.5 | 8 |
| `tail_pitch` | 0 −1 0 | −0.60 | +0.60 | 1.5 | 8 |
| `tail_yaw` | −1 0 0 | −0.60 | +0.60 | 1.5 | 8 |

**Canonical → URDF sign map** (verified by analytical FK, used by the retargeter and the default pose):

```
SY:   fl −1   fr +1   bl +1   br −1
SP:   fl +1   fr −1   bl +1   br +1
knee: fl +1   fr −1   bl +1   br −1
```

So a bilaterally **symmetric** crouch is `fl(SP −0.30, knee +0.60)`, `bl(SP −0.30, knee +0.60)`, `fr(SP +0.30, knee −0.60)`, `br(SP −0.30, knee −0.60)`. A uniform pose across all four legs is **wrong** and produces the "back-right leg inverted" look.

Note the **SY range is only ±0.42 rad (±24°)** — a hard constraint on any lateral/abduction content in a mocap clip. Additionally the policy caps SY authority by ×0.3 in the AMP env to stop splay-crouching.

## A.4 Canonical DOF and body ordering

Everything (motion files, obs, actions) uses these orders:

```python
DOF_ORDER (12) = [fl_SY_J, fl_SP_J, fl_knee,
                  fr_SY_J, fr_SP_J, fr_knee,
                  bl_SY_J, bl_SP_J, bl_knee,
                  br_SY_J, br_SP_J, br_knee]

BODY_NAMES (5)  = [origin, fl_knee, fr_knee, bl_knee, br_knee]
```

The env maps these onto the robot's 17-DOF articulation via `robot_ctrl_indexes`; head/tail stay at defaults. **Any new motion file must use exactly these names/order** or the loader silently mismatches.

## A.5 Actuators and control rates

| Item | Value |
|---|---|
| Drive type | position (implicit PD) |
| Stiffness | SY 1.15, SP 1.82, knee 2.10 |
| Damping | SY 0.092, SP 0.146, knee 0.166 |
| Head/tail | k 1.5, d 0.12 |
| Physics dt | 1/120 s |
| Decimation | 4 |
| **Control rate** | **30 Hz** |
| Action | 12-dim, scaled to joint limits (`offset + scale·a`), SY scale ×0.3 |
| Episode length | 10 s |
| Termination | base height < 0.15 m |

⚠️ Gain gotcha: the URDF→USD CLI conversion produced drives ~55× weaker than the GUI import. The gains above are the **rev_1 GUI values re-applied by hand** in `improved_walking_cfg.py`. Any re-export must re-check them, or the robot collapses under its own crouch.

## A.6 AMP environment

Files: `bingo_rl/bingo_rl/amp/bingo_amp_env.py`, `bingo_amp_env_cfg.py`, `agents/skrl_amp_cfg.yaml`. Direct workflow + skrl AMP agent (the only AMP implementation available in this Isaac Lab install).

**AMP / discriminator observation — 49 dims** (`compute_obs`):

| Block | Dims |
|---|---|
| `dof_positions` (12 leg joints) | 12 |
| `dof_velocities` | 12 |
| root height (world z) | 1 |
| root orientation as tangent+normal vectors | 6 |
| root linear velocity (world) | 3 |
| root angular velocity (world) | 3 |
| 4 × foot position **relative to root** | 12 |
| **total** | **49** |

`num_amp_observations = 2` → the discriminator sees 98 values (two consecutive frames).

**Policy observation** = 49 style features **+ 2 velocity commands `[vx, yaw]`** = **51**. The command is deliberately **not** in the discriminator obs — style must be command-agnostic.

**Foot convention (important).** The `*_knee` body origin is the knee **hinge**; knee flexion rotates *about* it, so its height barely changes (~8 mm) no matter how the leg steps. Both the reference and the policy therefore report the **shank tip** = `hinge + R_knee · (0, 0, −0.12)`. `SHANK_LEN = 0.12` is duplicated in `retarget_dog_to_bingo.py` and `bingo_amp_env.py` — **these must stay equal**, otherwise the discriminator separates real from fake on that channel for free and the style gradient collapses.

**Reset** uses `reset_strategy="random"`: sample a random time from the motion, write the reference root pose/velocity and joint state into the robot (Reference State Initialisation), and prime the AMP obs buffer from the reference.

**Reward (current, v14-style):** pure velocity tracking
`exp(−(vx − cmd_vx)²/0.25) + 0.5·exp(−(yaw − cmd_yaw)²/0.30)`.
The **best result to date (v9)** instead used `vel_rew × trot_factor × height_factor` with a single trot clip; see A.9.

**Agent config:** policy/value/discriminator all `[1024, 512]` ReLU, lr 5e-5, rollouts 16, `discriminator_loss_scale 5.0`, gradient penalty 5.0, `task_reward_scale 1.0 / style_reward_scale 1.5`.

## A.7 Motion-file schema (`.npz`)

This is the interface every retargeter must produce. `T` = frames, `D` = 12 DOF, `B` = 5 bodies.

| Key | Shape | Notes |
|---|---|---|
| `fps` | scalar | constant frame rate |
| `dof_names` | (D,) str | must equal `DOF_ORDER` |
| `body_names` | (B,) str | must equal `BODY_NAMES` |
| `dof_positions` | (T, D) f32 | radians, URDF sign convention, within limits |
| `dof_velocities` | (T, D) f32 | rad/s |
| `body_positions` | (T, B, 3) f32 | **world** metres; feet are shank **tips** |
| `body_rotations` | (T, B, 4) f32 | **wxyz** quaternions, world |
| `body_linear_velocities` | (T, B, 3) f32 | m/s |
| `body_angular_velocities` | (T, B, 3) f32 | rad/s |

The loader interpolates linearly between frames and slerps rotations, so the clip must be **temporally uniform** (no variable frame timing).

## A.8 Current motion source and how it is produced

- **Data:** Peng et al. `motion_imitation` dog clips (`dog_trot.txt`, `dog_pace.txt`) — Laikago-retargeted, 60 fps, 19 floats/frame = root_pos(3) + root_quat xyzw(4) + 12 joint angles. Lineage: Zhang et al. 2018 dog MoCap (research licence only).
- **Retargeter:** `bingo_rl/scripts/retarget_dog_to_bingo.py`, pure NumPy. It performs a **joint-space affine map**, not a true retarget:
  - leg mapping FR→fr, FL→fl, RR→br, RL→bl;
  - each dog joint is mean-centred and rescaled by hand-tuned gains (`G_SY 0.15`, `G_SP` 0.36 front / 0.60 back, `G_KN` 0.51 front / 0.85 back), then offset onto Bingo's crouch (`SP0 −0.30`, `KN0 +0.60`) and multiplied by the sign map;
  - root translation is **synthesised** (constant forward speed + scaled vertical bob), root yaw dropped, pitch/roll currently gained to 0 (`G_ROOT_ROT = 0`);
  - feet placed by analytical FK to the shank tip;
  - optional L/R symmetrisation; clip tiled ×`cycles`.

**This is the core limitation for 1:1 work.** The dog's *phasing* survives; its actual joint trajectories, stride length, foot placement, and contact timing do not. The amplitude gains are free parameters tuned by eye. Nothing guarantees the reference feet are on the ground when they should be — there is no contact model in the reference at all.

## A.9 Where results stand

- **v9 is the best gait produced so far** (`output/bingo_amp_FINAL.mp4`): single trot clip, knee-hinge feet, flat root, reward `vel × trot_factor × height_factor`, task/style 1.5/1.5. Clean diagonal trot, fl·br +0.98, tilt 0.008, vx 0.57.
- v10–v15 (shank-tip foot lift, injected body bob, front/back reach balance, L/R symmetrisation, command-conditioned multi-gait rebuild, task-weight tuning) each traded crispness away without a net visual win. Iteration on *tuning this pipeline* is closed.
- **The conclusion that matters for this document:** the remaining quality gap is not a reward-tuning problem. It is that the reference motion is a hand-scaled approximation of someone else's retargeted dog. Getting further requires **real motion data captured or authored against Bingo's own kinematics** — which is what Part B specifies.
- Repo code currently reflects the v14 command-conditioned config, not v9.

---

# PART B — WHAT 1:1 MOCAP TRAINING REQUIRES

## B.0 What "1:1" actually means here

Two distinct targets, often conflated:

| | **Style imitation (AMP — what we do now)** | **1:1 motion tracking (DeepMimic-style — what "1:1" means)** |
|---|---|---|
| Objective | match the *distribution* of reference states | reproduce *this specific clip*, frame by frame |
| Reference needs | phasing + rough posture | exact joint angles, contact timing, root trajectory |
| Reward | discriminator score + task | per-joint / per-body tracking error vs. a phase-indexed target |
| Failure mode | plausible but generic motion | drift/divergence; needs RSI + early termination |
| Retarget quality needed | moderate | **high — this is the whole ballgame** |

A 1:1 pipeline needs **both**: a tracking objective for fidelity, and (optionally) an adversarial term for robustness off-distribution. Everything below assumes we are building for tracking-grade data.

## B.1 Rig specification — hard requirements

The rig that produces training data must be a **kinematic twin of the robot**. Not "a dog rig that looks like Bingo" — the same joints, in the same places, with the same limits.

### B.1.1 Bone topology (mandatory)

Exactly one bone per robot joint, in the robot's hierarchy. No extra deform bones in the chain, no spine chain, no IK poles or control bones *inside* the export chain.

```
root                    (world transform of the base — see B.1.4)
└── origin              (base link; the reference body)
    ├── fl_shoulder_yaw → fl_shoulder_pitch → fl_knee → fl_foot_tip*
    ├── fr_shoulder_yaw → fr_shoulder_pitch → fr_knee → fr_foot_tip*
    ├── bl_shoulder_yaw → bl_shoulder_pitch → bl_knee → bl_foot_tip*
    ├── br_shoulder_yaw → br_shoulder_pitch → br_knee → br_foot_tip*
    ├── head_pitch → head_yaw → head_roll
    └── tail_pitch → tail_yaw
```

`*_foot_tip` is a **leaf marker bone only** — zero DOF, fixed 0.120 m down the shank's −Z. It exists so contact and foot position are exportable without re-deriving FK. It is not a joint.

**Bone names must match the URDF link names exactly** (case-sensitive). Any renaming happens in a single explicit mapping table in the importer, never implicitly.

### B.1.2 Degrees of freedom (mandatory)

- Each leg bone is a **1-DOF hinge**. `*_shoulder_yaw` rotates about X only; `*_shoulder_pitch` and `*_knee` rotate about Y only. Rotation on the other two axes must be **exactly zero** in every exported frame.
- Head: pitch (Y), yaw (X), roll (Z) — one axis each, in that chain order.
- Tail: pitch (Y), yaw (X).
- **No translation channels on any bone except `root`/`origin`.** No scale channels anywhere. No non-uniform scale, ever.
- Total animated DOF: **17 rotational + 6 root** = 23.

If the animation rig uses IK, poles, constraints, drivers, or correctives to author the motion, that is fine — but the **export must be an FK bake** of the chain above, with all constraints resolved and removed. Exported data containing an IK target or a pole vector is not usable.

### B.1.3 Rest pose and joint zeros (mandatory)

- The rig's **rest/bind pose must equal the URDF zero pose**: every joint at 0 rad = legs fully extended straight down, head and tail neutral. Joint values are then read directly with no offset table.
- Bone roll/orientation must be set so that the joint's rotation axis matches the URDF axis **including sign** (table in A.3). The right-side legs have flipped SP/knee axes — the rig must either replicate that flip, or export in a **canonical (all-legs-identical) convention** and let the importer apply the sign map from A.3. **Pick one and document it in the file.** Silent mismatch here is the classic "one leg bends backwards" bug.
- The functional reference pose (what standing looks like) is the crouch: SP −0.30, knee +0.60 canonical, base at 0.19 m. Animators should be told the crouch is the *neutral*, and zero is a straight-leg singularity that never occurs in real motion.

### B.1.4 Root and world convention (mandatory)

- **Units: metres. Up: +Z. Forward: +X. Left: +Y.** (Right-handed, matching URDF/Isaac.) If the DCC is Y-up or centimetre-scaled, the conversion happens once, in the exporter, and is asserted on import.
- The **root bone carries all world motion**: translation and full orientation of `origin`. Locomotion must **not** be baked into the hips or applied as a scene/object-level offset.
- Ground plane is **z = 0**. The rig must be authored standing on it — no floating, no sub-floor penetration.
- Root height when standing: **0.190–0.200 m**. If the rig's proportions put the base anywhere else, the retarget will fight it every frame.

### B.1.5 Proportions (strongly recommended)

For true 1:1 the rig should use **Bingo's link lengths verbatim** (A.2): thigh 0.0837, shank 0.120, hip wheelbase 0.1181, hip half-width 0.069. If the rig is authored at a different scale (e.g. a real dog's proportions for capture), then:

- proportions must be **uniformly scalable** to Bingo's by a single documented factor per segment, and
- the retargeter must do foot-position IK rather than raw joint copying (B.3), and
- the file must record the source skeleton's segment lengths so the scale factor is computed, not guessed.

Current placeholder ratio in use: **0.457** (Bingo 0.16 m leg vs. source dog 0.35 m).

### B.1.6 Joint limits (mandatory)

Animation must stay inside the robot's limits, with margin. Put these as hard rotation constraints on the rig so violations are impossible to author:

| Joint | Rig limit to enforce | Robot limit |
|---|---|---|
| `*_SY` | **±0.38 rad (±22°)** | ±0.42 |
| `*_SP` | **±1.40 rad (±80°)** | ±1.56 |
| `*_knee` | **±1.40 rad (±80°)** | ±1.56 |
| `head_pitch` | −0.58 … +0.36 | −0.65 … +0.40 |
| `head_yaw` | ±0.54 | ±0.60 |
| `head_roll` | ±0.70 | ±0.78 |
| `tail_pitch` / `tail_yaw` | ±0.54 | ±0.60 |

Also, **rate limits**: no joint may exceed **10 rad/s** (legs) or **8 rad/s** (head/tail) at the export frame rate. A clip that clips against a position limit or blows the rate limit produces a reference the robot physically cannot follow, and tracking rewards will punish the policy for the animator's frame.

**Acceptance rule:** < 0.5 % of frames may sit within 2 % of any position limit; **zero** frames may exceed a rate limit.

### B.1.7 Frame rate and timing (mandatory)

- Capture/author at **≥ 60 fps**, constant. **120 fps preferred** (the sim steps at 120 Hz; clean velocities come from clean sampling).
- **No** keyframe reduction, curve simplification, time-warping, retiming, or motion blur on the exported channels. Velocities are finite-differenced from positions — decimation shows up as velocity noise the discriminator/tracker will chase.
- Constant frame duration; the loader assumes uniform `dt`.
- Loopable gait clips: **first and last frame must match in joint space and in phase** (root translation excluded), so tiling doesn't inject a discontinuity.

### B.1.8 Contact annotation (strongly recommended)

Per frame, per foot, a boolean (or 0–1 confidence) "this foot is planted". Either an exported custom channel on the `*_foot_tip` bones, or a sidecar CSV/JSON keyed by frame.

Why it matters: contact is what makes root motion physically consistent. With contact flags we can (a) enforce zero foot velocity during stance to kill sliding, (b) derive root translation from foot kinematics instead of synthesising it, (c) compute duty factor and validate gait type, (d) feed a contact-matching reward. Without them, all of that is guesswork — and the current pipeline's synthesised root is exactly that guesswork.

### B.1.9 What must NOT be in the exported data

- IK targets, pole vectors, control/`ctrl_*` bones, `MCH-*` mechanism bones
- Extra articulation the robot does not have: spine segments, ear bones, jaw, individual toes, scapula float
- Scale or translation animation on any bone but the root
- Constraints, drivers, NLA layering, or actions that require evaluation to resolve
- Cameras, stage geometry, or scene-level transforms affecting the character

Expressive channels the robot *does* have — head (3) and tail (2) — **should** be animated and exported (see B.5.3). Ears and anything else with no motor are ignored by the loader; keep them out of the export chain.

### B.1.10 Export format

Preference order:

1. **Direct joint export (best).** JSON or NPZ: `fps`, per-frame root position (3) + root quaternion (4, state the convention) + 17 joint angles in radians in a documented order, plus contact flags. Zero ambiguity, trivial importer, no DCC dependency.
2. **FBX with FK bake.** Baked every frame, constraints removed, Z-up metres, one take per clip, take names = clip names.
3. **BVH.** Acceptable; note BVH is Euler-with-explicit-rotation-order and Y-up by default — the exporter must state the rotation order and axis convention in the header comment.

Every delivery must ship a **manifest**: clip name, fps, frame count, duration, gait type, nominal forward speed, whether it loops, contact channel present y/n, skeleton version, axis/unit convention, rotation convention.

## B.2 What to capture — clip list

For a controllable, natural quadruped policy, this is the minimum useful dataset. Speeds are **Bingo-scale** (m/s at the robot's size); if capturing a real animal, record the actual speed and body length so it can be scaled.

| # | Clip | Content | Length |
|---|---|---|---|
| 1 | Walk, slow | 4-beat lateral sequence, ~0.15–0.25 m/s | ≥ 6 cycles |
| 2 | Walk, medium | ~0.3 m/s | ≥ 6 cycles |
| 3 | Trot | diagonal pairs, ~0.4–0.6 m/s | ≥ 8 cycles |
| 4 | Trot, fast | ~0.7–0.9 m/s | ≥ 8 cycles |
| 5 | Pace | lateral pairs | ≥ 6 cycles |
| 6 | Turn in place | left and right, ~±0.5 rad/s | ≥ 4 s each |
| 7 | Walking turn | arcs left and right at walk/trot speed | ≥ 4 s each |
| 8 | Start / stop | stand → walk → stand transitions | 3 takes |
| 9 | Gait transitions | walk→trot→walk, trot→pace | 2 takes |
| 10 | Stand / idle | weight shifts, small head motion | ≥ 10 s |
| 11 | Sit → stand, lie → stand | if in scope for behaviour work | 2 takes each |
| 12 | Expressive idles | per-personality head/tail/posture, standing | as needed |

Notes:
- Straight-line clips should travel **straight along +X** or be yaw-normalised on import.
- Capture on **flat, level, known ground** with a calibrated z = 0 plane.
- More cycles per clip beats more clips — AMP and tracking both benefit from clean, long, uniform-speed segments.
- Keep raw takes; do not deliver only the trimmed loops.

## B.3 Retargeting pipeline we still need to build

Current retargeter is a joint-space affine map with hand-tuned gains. For 1:1 it needs to be replaced by a proper solver. Proposed stages:

**Stage 1 — Importer.** FBX/BVH/JSON → intermediate representation: bone hierarchy, rest-pose offsets, per-frame local rotations (quaternions), root pose, contact flags, fps. Assert units, axis convention, rotation order, and constant dt at this boundary. Emit a validation report.

**Stage 2 — Skeleton normalisation.** Compute per-segment scale from source skeleton to Bingo's link lengths (A.2). Scale root translation and foot targets accordingly. Record the factors in the output metadata.

**Stage 3 — IK retarget (the core).** Per frame, solve Bingo's `[SY, SP, knee]` per leg to minimise:

```
w_foot · Σ‖p_tip_bingo − p_tip_target‖²          (foot tip position, dominant term)
+ w_root · ‖root_pose_bingo − root_pose_target‖²  (height + pitch/roll; yaw handled separately)
+ w_smooth · ‖q_t − q_{t−1}‖²                     (temporal smoothness)
+ w_reg · ‖q − q_crouch‖²                         (stay near the nominal crouch)
subject to: joint limits (B.1.6), rate limits
```

Because the solve works backwards from foot position, it absorbs the morphology mismatch that the affine map cannot. Where a target is unreachable (source stride longer than Bingo's 0.204 m reach), clamp to the reachable workspace and **log the clamp** — clip regions with heavy clamping are candidates for rejection.

**Stage 4 — Contact-consistent root.** Rather than synthesising a constant-speed root: with contact flags, solve the root trajectory such that planted feet have zero world velocity. Removes foot sliding and makes the reference's forward speed physically earned. Fall back to the current synthesis only when contact data is absent, and mark such clips as lower-grade.

**Stage 5 — Feasibility filter.** Check rate limits, quasi-static torque feasibility against the 3.0 N·m leg effort, ground penetration, and self-collision. Report per-clip; reject or trim what fails.

**Stage 6 — Pack + dataset.** Write the A.7 `.npz` schema (extend it with `contacts (T,4)` and a `phase (T,)` channel for tracking). Support multi-clip datasets with per-clip weights.

**Stage 7 — Visual + numeric validation, before any GPU time.** Replay the packed motion on the robot in kinematic mode and inspect. Numeric gates:

| Metric | Gate |
|---|---|
| Foot-tip vertical travel (p2p) | 0.03–0.06 m per foot, matched within ~10 mm across all four |
| Foot slip during stance | < 5 mm per stance phase |
| Root height p2p | 0.01–0.03 m, non-zero |
| Root pitch/roll | non-zero, < 0.10 rad p2p |
| Diagonal phase corr. (trot) | fl·br and fr·bl > +0.9; fl·fr and fl·bl < −0.9 |
| L/R foot-tip amplitude difference | < 10 % |
| Frames clipped at a joint limit | < 0.5 % |
| Max joint velocity | < 10 rad/s (legs) |
| Duty factor | walk > 0.6, trot ~0.4–0.5 |

The "measure the reference before training" rule is load-bearing: every failed iteration in this project's history was visible in the reference numbers first.

## B.4 Training-side changes for 1:1

### B.4.1 Objective

Add a **tracking** mode alongside AMP:

- Reference is indexed by a **phase/time variable** carried in the policy obs (either normalised phase `[sin φ, cos φ]` or the next *k* reference frames as target features).
- Tracking reward, DeepMimic weighting as a starting point:

| Term | Content | Weight |
|---|---|---|
| joint position | `exp(−2 · Σ‖q − q_ref‖²)` | 0.50 |
| joint velocity | `exp(−0.1 · Σ‖q̇ − q̇_ref‖²)` | 0.05 |
| end-effector | `exp(−40 · Σ‖p_tip − p_tip_ref‖²)` | 0.20 |
| root pose | `exp(−20 · ‖p_root − p_root_ref‖² − 10·‖θ_err‖²)` | 0.15 |
| contact match | fraction of feet whose contact state matches | 0.10 |

- **Reference State Initialisation** (already implemented via `reset_strategy="random"`) is mandatory for tracking — it is what makes the hard parts of the clip reachable.
- **Early termination on tracking error** (e.g. root position error > 0.15 m, or orientation error > 0.6 rad) in addition to the height floor. Without it the policy learns to survive rather than track.
- Keep the AMP discriminator as a **secondary** term (style/robustness) at low weight, or run tracking-first then AMP-finetune. Do **not** run a hand-crafted gait-shaping reward alongside tracking — this project has already demonstrated that hand gait terms cause emergent leg-sacrifice asymmetry.

### B.4.2 Observation changes

| Change | Reason |
|---|---|
| + phase `[sin φ, cos φ]` or target frames (next 1–3 reference states) | the policy must know *where in the clip it is* |
| + per-foot contact booleans (4) | contact-aware tracking; cheap and available from the sensor |
| + previous action (12) | standard for smooth sim2real policies |
| keep discriminator obs command-agnostic | proven structural requirement (v14) |

### B.4.3 Rates and fidelity

- Control at 30 Hz vs. reference at 60–120 fps: the loader interpolates, which is fine, but **consider raising control to 50–60 Hz** (decimation 4 → 2) for tracking-grade fidelity. Higher control rate costs throughput; measure before committing.
- Motion clips must exceed the episode length (10 s) or be tiled cleanly.

### B.4.4 Domain randomisation (for eventual hardware)

Currently mostly off in the AMP env. For sim2real add: link mass ±15 %, base mass ±0.2 kg, ground friction 0.4–1.0, PD gains ±20 %, actuator latency 0–20 ms, observation noise on joint pos/vel and IMU, random pushes, and initial-state noise. Add these **after** tracking works in nominal sim, not before.

### B.4.5 Acceptance metrics for a 1:1 result

| Metric | Target |
|---|---|
| Mean per-joint tracking error | < 0.10 rad |
| Mean foot-tip position error | < 0.02 m |
| Root position drift over a 10 s episode | < 0.10 m |
| Contact-state match | > 85 % of frames |
| Survival rate on full clip | > 95 % |
| Phase correlations | within 0.05 of the reference's own |

## B.5 Constraints that cap achievable fidelity

Honest limits, so the rig and capture aren't specced against motion the robot can't perform:

1. **No ankle, no paw, no toe.** The knee is the last actuated joint; the shank tip is the contact point. Paw roll through contact, toe-off push, and heel strike **cannot be reproduced**. Mocap containing them is fine — the retargeter must simply drop the distal chain, and reviewers must not expect it back.
2. **No spine.** Bounding, galloping, and any motion whose realism depends on spinal flexion will read as stiff no matter what.
3. **SY ±0.42 rad and ×0.3 policy authority.** Wide-stance, lateral, or crab motions are out of range.
4. **Weak drives: 3.0 N·m per leg joint on a 2.478 kg robot.** Jumps, hard accelerations, and deep-crouch recoveries are likely infeasible; the feasibility filter (B.3 Stage 5) exists to catch this before training.
5. **Small scale.** Stride length ≤ ~0.20 m and stance height ~0.19 m. Dog clips must be time- and space-scaled; a real dog's cadence at the same Froude number is different from ours.
6. **Head (3 DOF) and tail (2 DOF) are currently unused by RL** — held at defaults. For expressive/personality work these are the highest-value cheap win available: they exist in hardware, they're in the URDF, and any rig will already animate them. Extending the motion schema from 12 to 17 DOF is a contained change (`DOF_ORDER`, obs dims, `robot_ctrl_indexes`) and would let captured head/tail performance transfer directly.

## B.6 Gap summary — what has to happen, in order

| # | Item | Owner | Blocking? |
|---|---|---|---|
| 1 | Rig rebuilt / conformed to B.1 (topology, axes, zeros, limits, units) | animation | **yes** |
| 2 | Axis-sign convention chosen and written into the delivery manifest | animation + RL | **yes** |
| 3 | Contact annotation added to the export | animation | high value |
| 4 | Clip list (B.2) captured/authored at ≥ 60 fps, constant, FK-baked | animation | **yes** |
| 5 | Importer + validation report (B.3 S1) | RL | **yes** |
| 6 | IK retargeter replacing the affine map (B.3 S3) | RL | **yes** |
| 7 | Contact-consistent root solve (B.3 S4) | RL | high value |
| 8 | Feasibility filter + validation gates (B.3 S5/S7) | RL | **yes** |
| 9 | Tracking objective + phase obs + early termination (B.4) | RL | **yes** |
| 10 | Extend schema 12 → 17 DOF for head/tail | RL | optional, cheap |
| 11 | Domain randomisation for hardware transfer | RL | later |

## B.7 Open questions to settle before capture

1. **Rig proportions** — authored at Bingo's exact link lengths, or at natural dog proportions with a documented scale factor? (Exact lengths make the retarget nearly trivial; dog proportions make the performance more natural to author.)
2. **Axis convention in the export** — replicate the URDF's flipped right-side axes, or export canonical and let the importer apply the sign map? Recommend **canonical + importer sign map**, so animators never see the asymmetry.
3. **Contact channels** — can the export carry per-foot booleans, or do we detect contact from foot height/velocity heuristics on import?
4. **Data licensing** — the current dog clips are research-use-only (Zhang et al. lineage via Peng). Anything commercial needs the reference data replaced with our own capture. This is an independent reason to build B.2.
5. **Locomotion vs. performance** — do the expressive/personality performances need to translate the root (locomotion), or are they in-place gestures layered on top of a locomotion policy? These are different training problems and the answer changes the clip list.
6. **Control rate** — stay at 30 Hz or move to 60 Hz for tracking fidelity?

---

## Quick reference card

```
Base link ......... origin           Total mass ........ 2.478 kg
DOF ............... 17 (12 leg + 3 head + 2 tail); 12 actuated in RL
Per leg ........... SY (X axis, ±0.42) → SP (Y, ±1.56) → knee (Y, ±1.56)
Thigh / shank ..... 0.0837 m / 0.120 m       Reach ....... 0.204 m
Wheelbase ......... 0.1181 m                 Hip width ... ±0.069 m
Stance height ..... 0.19–0.20 m              Foot ........ shank tip, knee + (0,0,−0.12)
Effort / vel ...... 3.0 N·m / 10 rad/s (legs); 1.5 / 8 (head-tail)
Gains ............. k = 1.15 / 1.82 / 2.10   d = 0.092 / 0.146 / 0.166
Physics ........... 120 Hz, decimation 4 → 30 Hz control
Units / axes ...... metres, Z-up, X-forward, right-handed, ground z=0
Motion fps ........ ≥ 60 (120 preferred), constant, FK-baked
Sign map .......... SY: fl−1 fr+1 bl+1 br−1 | SP: fl+1 fr−1 bl+1 br+1
                    knee: fl+1 fr−1 bl+1 br−1
Symmetric crouch .. fl(−0.30,+0.60) bl(−0.30,+0.60) fr(+0.30,−0.60) br(−0.30,−0.60)
```
