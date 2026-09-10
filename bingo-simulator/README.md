# Bingo Interactive Physics Simulator

A browser simulator for Bingo, the expressive 21-DOF quadruped: real MuJoCo physics
compiled to WebAssembly, the real v4 robot model, the real expression controllers, and
the real Stage-4-validated motions.

**It is not a website that plays Blender animations.** Everything visible on screen is
MuJoCo state.

Architecture reference: Pollen Robotics' [Microduck Sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-simulator).
Design report: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Read this before evaluating it

**Bingo does not walk in this build, and that is deliberate.**

No Bingo locomotion policy has been trained. The Task-1 environment exists (branch
`claude/bingo-locomotion-rl`) but has never been run — it has not even been
API-verified against Isaac Lab, because that needs a GPU. There is no `.onnx` in the
repository.

The brief forbids the alternatives, and it is right to:

> DO NOT play prerecorded walking animations instead of running the policy
> DO NOT move the floating root manually to fake locomotion

So the simulator ships everything up to the policy boundary and shows a `NO_POLICY`
banner. Press W and nothing happens, by design: the skill manager refuses to enter
`WALK` without a policy and says so. When a policy is trained, drop the `.onnx` in
`app/public/policies/`, add it to `policy_manifest.json`, and it runs behind an
interface that is already validated.

What **does** work today: full 21-DOF physics, the expression system, the three
Stage-4-validated gestures, push/reset, contacts, falls, and the diagnostic HUD.

---

## Quick start

```sh
cd bingo-simulator/app
npm install                  # .npmrc sets legacy-peer-deps for R3F v9 + React 19
../tools/vendor_runtime.sh   # copy MuJoCo + ONNX Runtime out of node_modules
npm run dev                  # http://localhost:5173
```

Regenerating the robot assets (only needed if the URDF changes):

```sh
python3 tools/export_bingo_mujoco.py --check   # URDF  -> MJCF
python3 tools/export_render_model.py           # meshes -> collision hulls + render
python3 tools/export_motions.py                # Stage-4 motions -> JSON
python3 tools/validate_isaac_mujoco.py         # 11 parity checks
```

## Controls

| | |
|---|---|
| `W` / `S` | forward / backward |
| `A` / `D` | turn left / right |
| `Q` / `E` | strafe |
| `1`–`6` | personality |
| mouse / right stick | head aim |
| `Space` | reset |
| `P` | push |
| `H` | toggle engineering HUD |

Gamepads use the standard mapping: left stick drives, right stick aims the head,
triggers turn, B recover, X push, Start reset, LB/RB cycle personality.

---

## Architecture

```
app/src/game/          framework-independent core - no React, no Three.js
  constants.js         the control contract (24 Hz, 21 joints, 66-dim obs)
  physics/mujoco.js    MuJoCo WASM boot, model load, name-based index maps
  policies/loader.js   ONNX loading + manifest validation (refuses mismatches)
  controllers/         the 9-joint expression layer
  skills/              deterministic skill router + robot-space motion playback
  controls/            keyboard, gamepad, merged controller
  runtime/sim.js       the fixed-step loop
app/src/scene/         React Three Fiber canvas + the render rig
app/src/ui/            HUD (public + engineering) and skill bar
app/public/robot/      MJCF, collision hulls, render meshes, kinematics.json
app/public/motions/    Stage-4 validated motions
app/public/policies/   .onnx files (empty - none trained yet)
```

Physics runs on its own wall-clock accumulator, independent of rendering. A dropped
frame cannot change the simulation.

```
120 Hz physics  ·  decimation 5  ·  24 Hz policy
```

That is Bingo's own Stage-4/Stage-5 rate, **not** Microduck's 50 Hz. Microduck runs
50 because that is *its* training rate; copying the number instead of the principle
would silently mis-feed any policy trained here.

### Observation and action

```
66 = base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + command 3
   + joint_pos_rel 21 + joint_vel 21 + last_action 12

q_target[j] = stance[j] + action_scale[j] * action[j]      (12 leg joints)
```

All 21 joints are observed while only 12 are actioned, so the expressive joints are
already in the policy's observation and Task 2 needs no observation-shape change.
Action scale is per joint (SY 0.125 / SP 0.195 / knee 0.170) — SY's range is ±0.42 rad
against ±1.56 for every other leg joint, so one scalar cannot serve them.

The loader **refuses** a policy whose manifest or ONNX graph disagrees with the
runtime, rather than running it. On a quadruped a silently reordered joint vector
does not throw — it walks wrong, and looks like a bad policy instead of a loading bug.

---

## The physics model

Generated from `bingo_urdf_w_ear_joints_physics.urdf` — the same URDF Isaac consumes —
by `tools/export_bingo_mujoco.py`. Masses, inertia tensors, joint origins, axes,
limits and collision geometry are copied; morphology is untouched.

Two things had to be handled rather than worked around:

**Four collision meshes are degenerate.** `tail_pitch`, `head_yaw`, `l_ear_pitch` and
`r_ear_pitch` have exactly 12 triangles and 0.0000 mm³ of volume, so MuJoCo's URDF
importer refuses the model outright. These are the same four near-massless
intermediate links `bingo_v4.py` documents as the articulation-conditioning problem.
The generator uses the URDF's explicit `<inertial>` for every link, so mesh volume is
never needed.

**Self-collision must be off.** Isaac spawns this robot with
`enabled_self_collisions=False`, so the validated Stage-4 physics has never had
robot-vs-robot contacts — and MuJoCo's default is on. With it on, the head rested
against the torso and `head_yaw` sat pinned at its 6 N·m ceiling against a −8 N·m
contact force, reaching 0.014 rad of a commanded 0.200. Encoding Isaac's setting via
`contype`/`conaffinity` moved standing joint error from 0.1538 to **0.0043 rad**.

### Validation

`tools/validate_isaac_mujoco.py` — **11/11 pass**
([report](docs/VALIDATION_REPORT.txt)):

| check | result |
|---|---|
| FK leg positions vs `V4Kin` | max 0.0001 µm over 500 poses |
| FK chain orientation | max 2.4e-06 deg |
| joint axes and limits | all 21 match the URDF |
| total mass | 2.459195327 vs 2.459195326 kg |
| stance paw heights | spread 0.0007 mm, all four level |
| standing 3 s | sank 0.6 mm, tilt 0.19°, joint error 0.0043 rad |
| gravity drop 50 mm | lands, settles to 0.0000 m/s |
| step response | 3 joints within 0.003 rad |
| contacts | 4/4 paws, 7 contact points |

That 0.0043 rad standing error sits almost exactly on Isaac's own documented
`stand_test` figure of 0.0041 rad.

`app/tools/check_rig_vs_physics.mjs` (`npm run check:rig`) drives a real browser and
compares each rendered link's world position against MuJoCo's — **worst disagreement
0.10 mm**. That check exists because a rig which has quietly drifted into being a
decorative animation still looks fine in a screenshot.

**Not verified:** the Isaac ↔ MuJoCo command-trace comparison. Isaac needs a GPU and
could not run here. `validate_isaac_mujoco.py --isaac-trace <npz>` accepts a trace
recorded by `rl/tools/eval_velocity.py` on the training machine; until then the model
is verified against the URDF, which catches conversion errors but does not prove
dynamic agreement.

---

## Expression and personality

Nine expressive joints (head 3, tail 2, ears 4) are driven by a deterministic
controller writing PD targets. The leg policy observes them, so it can learn to
balance against their motion.

**The personality selector changes head, tail and ear behaviour — not gait**, and the
UI says so. Per [the Task-2B audit](../docs/locomotion/STYLE_DATA_AUDIT.txt), the six
authored clips do not contain separable personality gaits: only Deadpan and Laidback
are physically plausible walking clips, and during walking the between-clip body-height
difference (34.7 mm) is *smaller* than the within-clip standard deviation of either
(±28.9 and ±44.3 mm).

What *is* separable is expressive-channel activity — RMS joint rate per clip:

| clip | tail | ear L | ear R | head |
|---|---|---|---|---|
| cheeky | **1.80** | 1.89 | 1.99 | 1.05 |
| timid | 0.48 | 1.41 | 1.57 | 0.86 |
| deadpan | 0.64 | 0.70 | 0.87 | 0.69 |
| laidback | **0.35** | 1.03 | 1.15 | 0.40 |

A 5× spread on the tail. The presets are built from those measured rates, so the
difference a viewer sees is a difference the data actually contains.

## Which skills are genuinely validated

Stage-4 status is authoritative. A Stage-3 animation existing is **not** sufficient.

| skill | status | evidence |
|---|---|---|
| **Yes** | ready | Stage 4: 55 frames, completes, 4/4 contacts |
| **No** | ready | Stage 4: 19 frames, completes, 4/4 contacts |
| **What** | ready | Stage 4: 71 frames, completes, 4/4 contacts |
| Cheeky | withheld | completes, but 1.405 m / 77.8° error — not a holistic pass |
| Timid | withheld | completes, but 0.611 m / 11.3° error |
| DeadPan, Eccentric, Enthusiastic, Laidback | withheld | no Stage-4 pass recorded |
| Walk | needs a policy | none trained |
| Recover | not built | no get-up policy trained for Bingo |

Withheld skills are **shown disabled with the reason**, not hidden — the list of what
Bingo can really do should be visible.

Gestures execute as joint targets tracked by the PD controllers under gravity and
contact. The floating root is never teleported. If Bingo cannot hold a motion
physically, it falls, which is the honest outcome.

---

## Known limitations

- No locomotion policy, so no walking (above).
- No Isaac trace comparison (above).
- Recovery is a reported `FALLEN` state, not a get-up.
- No props (ball, box, ramp) yet — the brief puts them after flat-ground validation.
- Render meshes are decimated to a 90k-triangle budget, which leaves the head
  visibly faceted. Raise it with
  `export_render_model.py --render-budget 250000`.
- Desktop only; touch controls are not built.

## Attribution

Architecture studied from Pollen Robotics' Microduck — the sandbox (via the
`Liyucheng1997/318_lab-microduck-simulator` fork of the
`pollen-robotics/microduck-simulator` Space), `pollen-robotics/microduck_rl`
(Apache-2.0) and `pollen-robotics/microduck`. **No Microduck code or asset is copied
into this project.** The sandbox fork carries no LICENSE file and the official 3D
assets are CC BY-SA-NC, so the runtime is reimplemented against Bingo's own model,
meshes and conventions.

Third-party runtime: [MuJoCo](https://mujoco.org) (Apache-2.0),
[ONNX Runtime Web](https://onnxruntime.ai) (MIT),
[Three.js](https://threejs.org) / [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) (MIT).
