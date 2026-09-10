# Bingo free locomotion — design report

Required by the task brief before any environment code is written. Four sections:
the Isaac Lab files to reuse, the current Bingo physics/action interface, the Task-1
observation/action/reward specification, and which Task-2 claims the existing
personality data actually supports.

**Status of the evidence in this document.** Everything marked `[MEASURED]` was
computed in this session from files in this repository (URDF, collision hulls,
Stage-3 `.npz`). Everything marked `[FROM REPO]` is read directly out of working
Bingo code that has already run against the local Isaac Lab. Everything marked
`[INFERRED]` is a conclusion about the local Isaac Lab API that could **not** be
checked here, because `~/robotics/IsaacLab` does not exist in this container and
neither does Isaac Sim. Every `[INFERRED]` item is checked by
`rl/tools/verify_locomotion_api.py`, which must be run on the training machine
**before** the first training run. Nothing in this design is trusted on my say-so.

---

## 1. Exact local Isaac Lab files to reuse

The brief says to port the local Unitree/ANYmal velocity task rather than upstream
main. That is already what this project did once: `rl/bingo_rl/bingo_rl/env_cfg.py`
subclasses the stock velocity task, and `improved_walking_cfg.py` specialises it
into a gait that the README records as **the original RL walking success**. So the
architecture to port is not hypothetical — it is in this repo, working, against the
exact Isaac Lab version installed on the training machine.

### Modules imported by working Bingo code `[FROM REPO]`

| Import | Used by | What we take |
|---|---|---|
| `isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg` → `LocomotionVelocityRoughEnvCfg`, `RewardsCfg`, `TerminationsCfg` | `env_cfg.py:7`, `improved_walking_cfg.py:28-31` | the whole manager-based velocity task |
| `isaaclab_tasks.manager_based.locomotion.velocity.mdp` (as `mdp`) | `improved_walking_cfg.py:27` | reward / event / termination / observation terms |
| `isaaclab.managers` → `RewardTermCfg`, `EventTermCfg`, `TerminationTermCfg`, `CurriculumTermCfg`, `SceneEntityCfg` | `improved_walking_cfg.py:20-25` | term wiring |
| `isaaclab.sim` → `SimulationCfg`, `PhysxCfg` | `stage5/bingo_stage5_env_cfg.py:52` | 120 Hz sim config |
| `isaaclab.actuators` → `IdealPDActuatorCfg`, `ImplicitActuatorCfg` | `bingo_v4.py`, `bingo.py` | the validated actuator model |
| `isaaclab_rl.rsl_rl` → `RslRlOnPolicyRunnerCfg`, `RslRlPpoActorCriticCfg`, `RslRlPpoAlgorithmCfg` | `agents.py:5` | PPO runner config |
| `isaaclab.terrains`, `TerrainGeneratorCfg` | `improved_walking_cfg.py` | (rough terrain, not used in Task 1) |

### `mdp` functions known to exist in the local version `[FROM REPO]`

Confirmed because working Bingo configs reference them:
`mdp.is_terminated`, `mdp.feet_slide`, `mdp.push_by_setting_velocity`,
`mdp.root_height_below_minimum`, `mdp.bad_orientation`, `mdp.terrain_levels_vel`.

Reward terms confirmed present on the stock `RewardsCfg` because `env_cfg.py` and
`improved_walking_cfg.py` assign to them: `track_lin_vel_xy_exp`,
`track_ang_vel_z_exp`, `lin_vel_z_l2`\*, `ang_vel_xy_l2`\*, `dof_torques_l2`,
`dof_acc_l2`, `action_rate_l2`, `feet_air_time`, `undesired_contacts`,
`flat_orientation_l2`, `dof_pos_limits`\*.
(\* = present in the stock class, not overridden by Bingo code, so name confirmed by
the upstream class rather than by a Bingo assignment — flagged for verification.)

Config attributes confirmed by assignment:
`actions.joint_pos.joint_names` / `.scale`,
`commands.base_velocity.ranges.{lin_vel_x,lin_vel_y,ang_vel_z}`,
`commands.base_velocity.{heading_command,rel_heading_envs,rel_standing_envs}`,
`observations.policy.{height_scan,enable_corruption}`,
`events.{push_robot,add_base_mass,base_external_force_torque,reset_robot_joints,reset_base,base_com,physics_material}`,
`terminations.base_contact`, `curriculum.terrain_levels`,
`scene.{robot,terrain,height_scanner,num_envs,env_spacing}`.

### Version markers `[INFERRED]`

The local version is **older than current Isaac Lab main**, and these names pin it:

- Events are called `add_base_mass` and `base_com`. Current main renamed these to
  `randomize_rigid_body_mass` / `randomize_rigid_body_com`. **Do not** use the new
  names.
- Actuator configs expose `effort_limit_sim` / `velocity_limit_sim` alongside the
  older `effort_limit` / `velocity_limit` (`bingo_v4.py` reads both with `getattr`
  fallbacks, which is why the fallback exists).
- `RslRlPpoActorCriticCfg` takes `actor_obs_normalization` / `critic_obs_normalization`
  (`agents.py:14-15`), which is a relatively recent rsl_rl field — so rsl_rl itself
  is not ancient.

**Rule adopted:** the new module reuses only names that already appear in working
Bingo code, plus a small set of stock-class names listed above for verification. No
API is invented from memory of newer Isaac Lab.

---

## 2. Current Bingo physics and action interface

### The robot `[FROM REPO]`

`rl/bingo_rl/bingo_rl/bingo_v4.py` → `BINGO_V4_CFG`, spawned from
`rl/v4_usd/bingo_v4.usd`. This is the validated Stage-4 physics model and is
**reused unchanged**: same USD, masses, collision geometry, joint limits, actuator
model and gains.

Actuators are **explicit** `IdealPDActuatorCfg`, not implicit. That matters and is
not a detail: with `ImplicitActuatorCfg` the PhysX drive keeps the USD's authored
`maxForce` (3.4e38) and `effort_limit` only scales the *reported* torque — leg
effort 3.0 / 8.0 / 15.0 N·m produced byte-identical motion. `IdealPDActuator`
computes `tau = clip(Kp*e - Kd*qd, ±effort)` in Python, so the ceiling actually
binds. Any torque-based reward or saturation metric is only meaningful because of
this.

| group | joints | Kp | Kd | effort | armature |
|---|---|---|---|---|---|
| legs | `SY` | 40.0 | 0.90 | 3.0 N·m | 0.01 |
| legs | `SP`, `knee` | 120.0 | 1.60 | 3.0 N·m | 0.01 |
| head_tail | `head_*` | 60.0 | 1.80 | 6.0 N·m | 0.06 |
| head_tail | `tail_*` | 8.0 | 0.20 | 6.0 N·m | 0.02 |
| ears | `*_ear_*` | 0.6 | 0.05 | 1.0 N·m | 0.0005 |

Leg `Kd = 1.60` is **capped by the standing gate, not by tracking**: at 2.20
`stage4/stand_test.py` fails on sustained torque saturation while merely standing.
Do not raise it for locomotion without re-running that gate.

### Joint order `[MEASURED]`

Canonical 21-DOF order (`v4_kinematics.DOF_ORDER`, identical to
`bake_conform.DOF_ORDER_21`):

```
fl_SY_J fl_SP_J fl_knee   fr_SY_J fr_SP_J fr_knee
bl_SY_J bl_SP_J bl_knee   br_SY_J br_SP_J br_knee
head_pitch_joint head_yaw head_roll   tail_pitch tail_yaw
l_ear_pitch l_ear_roll   r_ear_pitch r_ear_roll
```

**Isaac orders DOFs breadth-first, which is not this order.** Every existing Bingo
env maps by name (`track_v4/bingo_track_v4_env.py:33` uses
`robot.data.joint_names.index(n)`), and positional indexing scrambles the legs. The
new module does the same and never assumes an ordering.

Measured limits from `URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf`:

| joint | limit (rad) | limit (deg) |
|---|---|---|
| `*_SY_J` | ±0.42 | ±24.1 |
| `*_SP_J`, `*_knee` | ±1.56 (bl ±1.57) | ±89.4 |
| `head_pitch_joint` | −0.65 … +0.05 | −37.2 … +2.9 |
| `head_yaw`, `tail_pitch`, `tail_yaw` | ±0.60 | ±34.4 |
| `head_roll` | ±0.78 | ±44.7 |
| `l_ear_roll` | −1.50 … 0 | `r_ear_roll` 0 … +1.50 |

**SY is the binding constraint on the whole robot**: ±0.42 rad total, against ±1.56
for every other leg joint. MEMORY.md already flags SY overrun as an open issue.

### The standing pose — a defect found and worked around `[MEASURED]`

`BINGO_V4_CFG.init_state.joint_pos` still carries the **rev_3** pose
(`SP ±0.3, knee ±0.6`). Measured against the real collision hulls, that pose is not
a valid v4 stance:

| pose | paw heights in base frame | spread | support polygon x |
|---|---|---|---|
| `STAND_SOLVED` (stage4) | −0.180, −0.180, −0.180, −0.180 | **0.00 mm** | −0.059 … +0.059 |
| rev_3 legacy (in `bingo_v4.py`) | −0.154, −0.154, −0.162, −0.162 | 8.14 mm | **+0.024 … +0.207** |

The legacy pose puts one paw 207 mm in front of the base while another is at 24 mm —
a 183 mm skew on a robot whose whole support polygon should be 118 mm long. This
independently reproduces the note in `stage4/stand_test.py`, which records that the
inherited pose "drove `fr_SP_J` into its +1.56 limit and jammed the leg against the
floor". The sign pattern is the giveaway: the solved stance is
`fl +0.81 / fr −0.81 / bl +0.39 / br +0.39`, so front legs mirror across L/R but the
back legs do **not**, and the legacy pose gets both wrong.

Drawn out (`rl/tools/plot_stance_and_action_scale.py` →
`docs/locomotion/stance_and_action_scale.png`) the consequence is starker than the
numbers suggest: the legacy support polygon spans x = +0.024 … +0.207 m, so **the
base origin at x = 0 lies entirely outside it**. The robot's body is not over its
feet at all, in any orientation — it is a pose that must topple backwards the
instant gravity is applied, which is consistent with everything Stage 4 recorded
about it.

**Decision:** `bingo_v4.py` is *not* modified — it is validated Stage-4 physics and
Stage 5 depends on it (RSI overwrites the init state every reset, so the bad pose is
harmless there). The locomotion env overrides `init_state.joint_pos` with
`STAND_SOLVED` locally. Nominal base height is then **0.182 m** (lowest paw hull
2 mm above the floor).

### Control timing `[FROM REPO]` — preserved

`sim.dt = 1/120`, `decimation = 5` → **24 Hz policy**, matching Stage 4 and Stage 5
(`stage5/bingo_stage5_env_cfg.py:52-58`). The brief requires this be held fixed
while the baseline is established, and it also keeps every existing Bingo timing
convention intact. Note this is *slower* than the 50 Hz typical of Isaac Lab
locomotion tasks; reward weights that assume 50 Hz (notably `action_rate_l2` and the
air-time threshold) are therefore **not** transferable unchanged, and are flagged in
§3.

---

## 3. Task-1 specification

### Action space — 12 leg joints

```
q_target[leg] = q_stand[leg] + scale[leg] * action     (12 actions)
q_target[expr] = q_stand[expr] = 0                     (9 joints, PD-held)
```

`scale` is **per joint type**, derived rather than inherited `[MEASURED]`. With
`soft_joint_pos_limit_factor = 0.9` the usable travel from `STAND_SOLVED` to the
soft limit is:

| joint | q_stand | headroom down | headroom up | binding |
|---|---|---|---|---|
| `*_SY_J` | 0.0000 | 0.378 | 0.378 | 0.378 |
| `fl_SP_J` | +0.8100 | 2.214 | 0.594 | **0.594** |
| `fr_SP_J` | −0.8109 | 0.593 | 2.215 | **0.593** |
| `bl/br_SP_J` | +0.393 | 1.80 | 1.01 | 1.010 |
| `*_knee` | ±0.893 | — | — | **0.510** |

Choosing `scale` so that **|action| = 3 reaches exactly the soft limit** lets the
policy use the full range of its Gaussian without ever commanding a target outside
the soft limits:

| joint | scale | reach at \|a\|=3 | binding headroom |
|---|---|---|---|
| `*_SY_J` | **0.125** | 0.375 | 0.378 |
| `*_SP_J` | **0.195** | 0.585 | 0.593 |
| `*_knee` | **0.170** | 0.510 | 0.510 |

This is deliberately **not** Stage 5's 0.3. At 0.3, `|a|=3` asks SY for 0.9 rad
against 0.378 of headroom — 2.4× past its limit — which is precisely the SY-overrun
failure MEMORY.md records as warned-but-unsolved. A single scalar scale cannot be
right for this robot because SY's range is 3.7× smaller than SP's.

### Observation space — 66 dims

| term | dims | note |
|---|---|---|
| `base_lin_vel` | 3 | body frame |
| `base_ang_vel` | 3 | body frame |
| `projected_gravity` | 3 | |
| `velocity_commands` | 3 | `[vx, vy, yaw_rate]` |
| `joint_pos_rel` | **21** | all joints, relative to `q_stand` |
| `joint_vel_rel` | **21** | all joints |
| `last_action` | 12 | legs only |
| **total** | **66** | |

All 21 joint states are observed although only 12 are controlled, so that the Task-2
expressive joints are already in the leg policy's observation and the policy can
learn to balance against head/tail/ear motion without an observation-shape change
later.

Sanity check on this layout `[MEASURED]`: `improved_walking_cfg.py` records the flat
obs dim as **58** on the 17-joint rev_3 robot. 3+3+3+3+**17**+**17**+12 = 58 ✓. The
same layout on 21 joints gives 3+3+3+3+21+21+12 = **66**.

No global world XYZ is exposed to the actor, per the brief.

### Command sampling

The stock `UniformVelocityCommand` samples uniformly and would almost never produce
exact zero or pure turn-in-place. The brief requires those explicitly, so the module
adds a command term that samples a **category first, then values within it**:

| category | share | `vx` | `vy` | `yaw_rate` |
|---|---|---|---|---|
| stand (exact zero) | 10% | 0 | 0 | 0 |
| forward | 25% | +0.10…+0.40 | 0 | 0 |
| backward | 10% | −0.25…−0.10 | 0 | 0 |
| lateral | 10% | 0 | ±0.10…0.25 | 0 |
| diagonal | 10% | ±0.10…0.30 | ±0.10…0.20 | 0 |
| turn in place | 15% | 0 | 0 | ±0.4…1.2 |
| turn + translate | 20% | +0.10…+0.35 | 0 | ±0.3…1.0 |

Ranges are Bingo-scaled, not Go1-scaled: a 0.18 m, 2.5 kg quadruped. The existing
`env_cfg.py` already narrowed the stock ranges to ±0.5 / ±0.3 / ±1.0 for exactly
this reason. Stage 1 of the curriculum starts at the *forward* category only.

### Rewards

Ported structurally from the local velocity task, with weights **inherited from
`improved_walking_cfg.py` where that config measured them on Bingo** and flagged
where they must be re-measured.

| term | weight | basis |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.5, std 0.25 | `[FROM REPO]` measured: at 2.0/std 0.2 PPO learned a tilted 3-legged drag that tracked `vx` perfectly |
| `track_ang_vel_z_exp` | +0.75 | `[FROM REPO]` |
| `lin_vel_z_l2` | −2.0 | stock |
| `ang_vel_xy_l2` | −0.05 | stock |
| `flat_orientation_l2` | −2.5 | `[FROM REPO]` measured: kills an 18° tilt |
| `dof_torques_l2` | −5.0e-5 | `[FROM REPO]` softened while gait forms |
| `dof_acc_l2` | −2.5e-7 | `[FROM REPO]` |
| `action_rate_l2` | −0.005 | `[FROM REPO]` **re-measure**: was tuned at 50 Hz, we run 24 Hz |
| `dof_pos_limits` | −1.0 | raised from stock; SY has only ±0.42 |
| `feet_air_time` | +0.25, threshold 0.15 s | `[FROM REPO]` |
| `feet_slide` | −0.4 | `[FROM REPO]` |
| `air_time_over` | −1.0, cap 0.35 s | `[FROM REPO]` anti-tripod, custom term |
| `contact_time_over` | −1.0, cap 0.35 s | `[FROM REPO]` anti-prop, custom term |
| `undesired_contacts` | −1.0 | any non-paw link touching |
| `termination_penalty` | −100.0 | `[FROM REPO]` |
| `stand_still` | −0.5 | **new**, only active on the zero command |

`air_time_over` / `contact_time_over` are carried over deliberately. They are not
stock Isaac Lab — they were written for Bingo to kill two measured failure modes
(v3 held one leg up permanently and dragged on three; v6 parked the front-left foot
grounded 85% of the time as a static prop). Those failure modes are properties of
the robot, not of the old config, so the terms come along.

The foot body is `.*_knee` `[FROM REPO]`: v4 has **no ankle or paw joint**, the paw
is rigid collision geometry on the shank/knee link. The base link is `origin`.

### Terminations

| term | value | basis |
|---|---|---|
| `base_contact` | any force on `origin` | stock |
| `low_base` | z < 0.14 m | 77% of the measured 0.182 m stance |
| `bad_orientation` | tilt > 0.8 rad (45.8°) | `[FROM REPO]` |

### Training progression

Staged exactly as the brief orders, with randomisation **off** until a gait exists:
(1) flat, nominal physics, forward-only, narrow commands → (2) reliable
forward/stand → (3) wider velocities → (4) turning → (5) reverse/lateral →
(6) start/stop → (7) obs noise → (8) friction → (9) mass/COM → (10) actuator
variation → (11) pushes. Stages 1–6 are command/reward curriculum; 7–11 are the
event terms, each enabled one at a time.

### Evaluation battery

`rl/tools/eval_velocity.py` runs fixed command trajectories — stand, forward,
reverse, turn L, turn R, turn-in-place, lateral, diagonal, start→move→stop — and
reports commanded vs measured `vx/vy/yaw`, RMSE, survival, root tilt, foot slip,
joint-limit hits, mean/max torque, torque-saturation %, and action smoothness, plus
a rendered video. **Task 1 is complete only when Bingo is actually driven by the
command** — surviving, or oscillating around one pose, is not success.

---

## 4. Which Task-2 claims the data actually supports

Measured by `rl/tools/analyze_style_motions.py` on the Stage-3 robot-space motions.
Full report: `docs/locomotion/STYLE_DATA_AUDIT.txt`. Four independent criteria, each
reported separately so it is visible which one failed.

| clip | verdict | feet down | peak v | Froude | why |
|---|---|---|---|---|---|
| **deadpan** | **USABLE** | 2.48 | 0.18 | 0.02 | 7 cycles, 0.531 m |
| **laidback** | **USABLE** | 2.56 | 0.21 | 0.03 | 6 cycles, 0.509 m |
| timid | MARGINAL | 2.96 | 1.16 | 0.76 | only 2 cycles |
| cheeky | NOT LOCOMOTION | **1.05** | **3.31** | **6.20** | airborne most of the clip |
| enthusiastic | NOT LOCOMOTION | 1.63 | 3.24 | 5.96 | travels 0.045 m net (in place) |
| eccentric | NOT LOCOMOTION | 0.50 | 0.45 | 0.11 | hind paws never touch (authored sit) |
| reaction_* | NOT LOCOMOTION | 4.00 | 0.00 | — | seated/gestural by construction |

### What this means, stated plainly

**The six personality clips are not a locomotion dataset.** That is not a new
finding — README says so ("an expressive layer, not a locomotion dataset — gait
still needs the capture in spec §B.2") and MEMORY.md records Eccentric as an
authored sit and Cheeky as a trot/bounce whose flight phases Stage 4 cannot hold.
This audit puts numbers on it.

Consequences for Task 2B, recorded now rather than discovered during training:

**1. The "two maximally different personalities" the brief asks to start from do not
exist in this data as gaits.** Only Deadpan and Laidback are physically plausible
walking clips, and both travel at ~0.03 m/s mean over the clip. They are not
maximally different *gaits*; they are two slow shuffles.

**2. The posture difference does not survive measurement either.** A clip-wide mean
height is misleading when the clip contains a postural transition — Laidback's base
sweeps 50 → 194 mm because the character lies down and gets up, so its 112.8 mm clip
mean describes a transition, not a stance. Restricting to frames where the body is
actually travelling with ≥2 paws down `[MEASURED]`:

| clip | walking frames | height (mm) | pitch (deg) | speed (m/s) |
|---|---|---|---|---|
| deadpan | 224 (60%) | 173.2 **± 28.9** | −7.5 ± 15.9 | 0.074 |
| laidback | 149 (31%) | 138.5 **± 44.3** | −4.9 ± 12.4 | 0.083 |
| timid | 78 (43%) | 157.6 ± 16.5 | −9.6 ± 13.4 | 0.090 |
| enthusiastic | 44 (21%) | 162.8 ± 26.8 | −1.1 ± 17.0 | 0.140 |
| cheeky | 13 (7%) | 164.3 ± 28.0 | +8.0 ± 9.0 | 0.061 |
| eccentric | 2 | — | — | — |

The largest between-clip height difference (Deadpan 173.2 vs Laidback 138.5 =
**34.7 mm**) is *smaller than the within-clip standard deviation of either clip*
(±28.9 and ±44.3 mm). Walking speed is effectively constant across all of them
(0.061–0.140 m/s). **Posture during locomotion is not a separable style axis in this
data.** An earlier draft of this report claimed it was, on the clip-wide means; the
walking-segment measurement withdraws that claim.

**3. What *is* separable is expressive-channel activity** `[MEASURED]`. RMS joint
rate over the clip, in rad/s:

| clip | tail | ear L | ear R | head |
|---|---|---|---|---|
| cheeky | **1.80** | 1.89 | 1.99 | 1.05 |
| timid | 0.48 | 1.41 | 1.57 | 0.86 |
| deadpan | 0.64 | 0.70 | 0.87 | 0.69 |
| laidback | **0.35** | 1.03 | 1.15 | 0.40 |

That is a 5× spread on the tail and 3× on the head — well outside measurement noise,
and it is what a human actually reads as "personality" in these clips.

**4. Cheeky's 3.31 m/s peak and 1.05 feet down are authored animation, not
locomotion.** Froude 6.2 is past galloping for a 0.18 m quadruped. A style reward
imitating Cheeky's root velocity would ask the policy to reproduce something the
robot cannot do — the failure Stage 4 already measured (`recipes/cheeky.conf`:
contact-wrench feasible on only 46% of frames, 39 frames needing friction above 0.5,
20 needing μ > 1).

### Recommendation

- **Task 2A is supported and should be built first**, but its parameter ranges must
  be established experimentally on the trained Task-1 policy, **not** anchored on
  the personality clips — §4.2 shows those clips do not contain a separable posture
  signal to anchor on. The brief already says "do not copy Go1 ranges"; this adds
  that Ashley's clips cannot supply them either.
- **Task 2B cannot honestly claim personality *gait* transfer from this data.** The
  defensible version of Task 2B with what exists today is style conditioning on the
  **expressive channels** (head/tail/ear activity, §4.3), which are measurably
  distinct, layered on the Task-1 locomotion policy. That is a real and visible
  result, and it is not the same claim as "Cheeky walks differently from Laidback".
- **Additional authored locomotion is required** for a genuine personality-gait
  dataset. Quantified request, per personality: a straight-line walk of ≥5 s at
  roughly constant speed, all four paws cycling, ≥3 clean touchdown-to-touchdown
  cycles per foot, mean feet-down ≥1.5, peak root speed under 2.3 m/s (Froude 3).
  This is the same capture the pipeline spec already asks for in §B.2 — now with
  acceptance criteria attached. Until it exists, personality-conditioned
  *locomotion* is not measurable, and Task 2B should not be reported as complete.

---

## Verification gate

Before the first training run, on the training machine:

```sh
cd ~/robotics/IsaacLab
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/verify_locomotion_api.py
```

It checks every `[INFERRED]` item above — module paths, `mdp` function names, config
attributes, the actuator model, the joint-name mapping, the resolved observation
dimension, and that `STAND_SOLVED` is inside the joint limits — and prints a
pass/fail line for each. If it fails, fix the config, not this document's
assumptions.
