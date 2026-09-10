# Bingo Interactive Physics + RL Simulator — architecture report

Required before implementation. States what is consumed, how the URDF becomes MJCF,
how parity is tested, how expression and skills work, and — most importantly — what
is genuinely available today versus what is blocked.

Evidence tags: `[MEASURED]` computed in this session from repo files; `[FROM REPO]`
read from Bingo code/docs; `[FROM MICRODUCK]` read from the cloned Microduck source;
`[BLOCKED]` cannot be done in this environment and why.

---

## 0. The headline constraint — read this first

**There is no trained Bingo locomotion policy.** The Task-1 environment exists
(`rl/bingo_rl/bingo_rl/locomotion/`, branch `claude/bingo-locomotion-rl`) but has
never been trained: it has not even been API-verified against Isaac Lab, because
this environment has no GPU, no Isaac Sim and no Isaac Lab. There is no `.onnx`
anywhere in the repository `[MEASURED]`.

The brief is explicit and correct that walking must come from real policy inference,
and that faking it is forbidden:

> DO NOT play prerecorded walking animations instead of running the policy
> DO NOT move the floating root manually to fake locomotion

**So this simulator will not walk yet, and it must not pretend to.** What gets built
now is everything up to and including the policy boundary — the physics twin, the
render rig, the ONNX runtime, the manifest contract, the observation/action plumbing
and the HUD — with an explicit `NO_POLICY` state that says so on screen. The moment a
trained policy exists, it drops in behind an already-validated interface.

This is a deliberate scope decision, not an omission. Every other deliverable in §18
of the brief is buildable today.

---

## 1. Exact Bingo files consumed

| file | role | note |
|---|---|---|
| `URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf` | **authoritative** physics model | 22 links, 21 joints, 2.45920 kg total `[MEASURED]` |
| `URDF/bingo_urdf v4_w_ear_joints/meshes/*.STL` | collision + visual geometry | 40 files `[MEASURED]` |
| `stage4/out/collision_hulls.npz` | convex hulls used by Stage 2/4 | 15 links; cross-check for MJCF contact geometry |
| `rl/bingo_rl/bingo_rl/bingo_v4.py` | actuator gains, armature, effort limits | Stage-4 validated; source for MJCF actuators |
| `stage4/stand_test.py` → `STAND_SOLVED` | the validated standing pose | all four paws at −0.180 m, spread 0.00 mm `[MEASURED]` |
| `stage2/v4_kinematics.py` | independent FK implementation | the reference the MJCF is checked against |
| `motions/*_v4.npz` | Stage-3 robot-space motions | expression skills; see §7 |
| `docs/locomotion/TASK1_DESIGN.md` | obs/action/scale spec | the contract a future policy must satisfy |

**Not used:** the Ashley art rig, any `blend_sources/*.blend`, and the rev_1/rev_3
URDFs. Per the brief, the animation skeleton is never the physics robot.

---

## 2. What Microduck actually does (verified from source, not assumed)

Hugging Face is blocked by this environment's egress policy, so the Space could not
be cloned directly `[BLOCKED]`. The sandbox source was obtained instead from a
GitHub fork of it (`Liyucheng1997/318_lab-microduck-simulator`, whose
`package.json` names the project `microduck-sandbox`), plus the upstream training
repo `pollen-robotics/microduck_rl` (Apache-2.0) and the robot runtime
`pollen-robotics/microduck`. All three cloned and read `[FROM MICRODUCK]`.

### Their stack, as it really is

```
app/src/game/          framework-independent game core (no React imports)
  game.js              MuJoCo + ONNX loop, 1905 lines
  duck.js              Three.js render rig built from kinematics.json
  constants.js         the entire policy contract as plain constants
  controls/            keyboard.js, gamepad.js, touch.js, controller.js
app/src/scene/         React Three Fiber canvas
app/src/ui/            HUD and overlays
app/src/store.js       Zustand
app/public/robot/mjlab/  MJCF + meshes + microduck.glb
app/public/policies/     the .onnx files
```

MuJoCo and ONNX Runtime are **not bundled**. They are dynamically imported from a
CDN with `@vite-ignore`, because their `.wasm` sidecars must resolve relative to the
CDN URL `[FROM MICRODUCK]`:

```js
const MUJOCO_URL = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js";
const ORT_URL    = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
```

### Their control loop — the part worth copying exactly

```js
async function controlStep() {
  const feeds = { obs: new ort.Tensor("float32", buildObs(), [1, OBS_SIZE]) };
  const out = await activeSession().run(feeds);
  const act = out.actions.data;
  lastAction.set(act);
  for (let j = 0; j < NUM_JOINTS; j++)
    ctrl[j] = DEFAULT_POSE[j] + act[j] * ACTION_SCALE;   // position targets
  for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
}
```

driven by a wall-clock accumulator that is **independent of rendering**, and which
explicitly refuses to spiral when it falls behind `[FROM MICRODUCK]`:

```js
next += CTRL_DT * 1000;
const wait = next - performance.now();
if (wait > 0) await new Promise((r) => setTimeout(r, wait));
else next = performance.now();   // fell behind: don't spiral
```

### Their observation, and why it matters for us

61 floats = gyro 3 + projected gravity 3 + (qpos − default) 14 + qvel 14 +
lastAction 14 + command 13 `[FROM MICRODUCK]`. Timing is `TIMESTEP 0.005`,
`DECIMATION 4` → 50 Hz, which is their **training** rate, not a browser convenience.

Their `publish/manifest.py` hard-codes `OBS_LEN = 61`, `ACTION_LEN = 14`,
`control_hz = 50` and the runtime *refuses* a policy whose manifest disagrees, and
refuses at load a network whose graph disagrees. That refusal behaviour is the part
§8 of the brief asks for, and we copy the idea, not the numbers.

### Licensing

`microduck_rl` is Apache-2.0 (verified: `LICENSE` is the Apache 2.0 text). The
sandbox fork carries **no LICENSE file**, and the brief notes the official 3D model
assets are CC BY-SA-NC. **Consequence: no Microduck code or asset is copied into
this project.** The architecture is reimplemented from what was read, against
Bingo's own model and meshes, and this report is the attribution record.

---

## 3. URDF → MuJoCo conversion

### Why the stock importer is not enough `[MEASURED]`

MuJoCo 3.13 refuses the v4 URDF outright:

```
Error: mesh volume is too small: head_yaw . Try setting inertia to shell
```

Investigating rather than working around it found a real property of the model.
**Four collision meshes are degenerate — exactly 12 triangles and 0.0000 mm³ each:**

| link | mass (kg) | mesh volume (mm³) | triangles |
|---|---|---|---|
| `tail_pitch` | 0.00300 | 0.0000 | 12 |
| `head_yaw` | 0.00500 | 0.0000 | 12 |
| `l_ear_pitch` | 0.00200 | 0.0000 | 12 |
| `r_ear_pitch` | 0.00200 | 0.0000 | 12 |

These are exactly the four links `bingo_v4.py` already documents as the
near-massless-intermediate-link problem — "head_yaw is 5 g driving the 0.708 kg head,
a 140:1 mass ratio the articulation solver cannot condition, so the joint jams
instead of moving" — which is why that file carries hand-tuned armature values. The
degenerate mesh is the same defect seen from the geometry side. Setting
`discardvisual` does not help: they are collision meshes.

### The approach: a generator, not a hand-rebuild

`tools/export_bingo_mujoco.py` reads the URDF and emits MJCF deterministically. It
never invents geometry and never edits morphology. Specifically:

- **Inertials come from the URDF verbatim** — mass, centre-of-mass origin and the
  full 3×3 inertia tensor per link — so MuJoCo never needs a mesh volume. This is
  what makes the four degenerate meshes a non-issue rather than a hack.
- Meshes are declared `inertia="shell"` as a second belt-and-braces guard.
- Joint origins, axes and limits are copied from the URDF.
- Joint declaration order is the canonical `DOF_ORDER`, so `qpos`/`ctrl` indices are
  stable and match `bake_conform.DOF_ORDER_21`. Index mapping is still done **by
  name** at runtime, never positionally.
- A `freejoint` on `origin` (the floating base).
- 21 `position` actuators carrying the Stage-4 gains from `bingo_v4.py`
  (`kp` legs SY 40 / SP,knee 120; head 60; tail 8; ears 0.6, with matching `kv` and
  `armature`), plus `forcerange` from the effort limits (legs ±3.0, head/tail ±6.0,
  ears ±1.0 N·m).
- A keyframe holding `STAND_SOLVED` at base height 0.182 m.

Two MJCF files are produced: `bingo_v4.xml` (the robot alone, the parity subject) and
`bingo_scene.xml` (robot + floor + arena + props, what the browser loads).

---

## 4. Isaac ↔ MuJoCo parity — what can and cannot be tested here

`[BLOCKED]` Isaac Sim cannot run here (no GPU, no `isaacsim`, no `isaaclab`), so the
brief's literal "run identical command traces in Isaac and browser MuJoCo" cannot be
executed in this environment.

What **is** available is stronger than it first appears. The URDF is the artefact
Isaac itself consumes — `bingo_v4.py` spawns a USD converted from it — and
`stage2/v4_kinematics.py` is an independent, already-trusted FK implementation of
that same URDF, used by the whole Stage-2/4 pipeline. So MJCF can be checked against
the *authority both simulators derive from*, on CPU, right now:

`tools/validate_isaac_mujoco.py` compares MuJoCo against `V4Kin` and the collision
hulls on:

1. FK body positions/orientations at N random joint configurations, all 21 joints
2. joint limits, axes and origins, joint by joint
3. link masses, total mass, and centre-of-mass position
4. the `STAND_SOLVED` rest pose and its paw heights (target: all four at −0.180 m)
5. standing equilibrium under gravity — does it hold the stance, and how far does it sink
6. gravity drop from a height — does it land and settle
7. single-joint step responses against the configured PD gains
8. contact positions vs `stage4/out/collision_hulls.npz`

Every check emits numbers and a pass/fail against a stated tolerance. Items 1–4 are
pure kinematics/mass-properties and are the ones that would catch a conversion
error; 5–8 are dynamics and are compared for *behavioural* agreement, since MuJoCo
and PhysX will never match bit-for-bit.

**The Isaac-side trace comparison remains a required, un-run deliverable.**
`tools/validate_isaac_mujoco.py --isaac-trace <npz>` accepts a trace recorded on the
training machine by `rl/tools/eval_velocity.py` and does the comparison the brief
asks for. Until that file is produced on a GPU box, parity is verified against the
URDF only, and this report says so rather than implying otherwise.

---

## 5. Visual model, driven by physics

Physics and rendering are separated exactly as the brief requires:

```
MuJoCo qpos/qvel  →  render rig (one Three.js Group per body)  →  Bingo visual meshes
```

`tools/export_render_model.py` emits `kinematics.json` (body tree, joint axes and
origins, mesh assignment per body) plus a merged `bingo.glb` from the 40 STLs. The
rig sets each body's local rotation from the corresponding MuJoCo joint angle and
the root from the free joint. **No independent animation of the visual model at any
time** — if physics is paused, the robot is still.

---

## 6. Policy interface and the manifest

`policy_manifest.json` mirrors Microduck's contract idea with Bingo's numbers, and
the loader **refuses** rather than silently running a mismatched policy — the failure
mode the brief calls out and Microduck's daemon is built to prevent.

Derived from `docs/locomotion/TASK1_DESIGN.md` `[FROM REPO]`:

| field | Bingo value |
|---|---|
| `observation_dim` | 66 |
| `observation_schema` | `base_lin_vel 3, base_ang_vel 3, projected_gravity 3, command 3, joint_pos_rel 21, joint_vel 21, last_action 12` |
| `action_dim` | 12 (legs only) |
| `action_scale` | per joint: SY 0.125, SP 0.195, knee 0.170 |
| `control_hz` | 24 |
| `physics_hz` | 120 |
| `decimation` | 5 |
| `joint_order` | canonical 21-DOF `DOF_ORDER` |
| `default_joint_pose` | `STAND_SOLVED` + 9 expressive zeros |

Two differences from Microduck that must not be copied over:

- **24 Hz, not 50 Hz.** Bingo's Stage-4/Stage-5 stack is 120 Hz physics /
  decimation 5. Microduck runs 50 Hz because that is *its* training rate; copying
  the number rather than the principle would be exactly the error the brief warns
  about.
- **Bingo observes all 21 joints while acting on 12.** Microduck acts on all 14 of
  its joints. Bingo's 9 expressive joints are observed so the leg policy can
  compensate for head/tail/ear motion — that is what makes Task-2 Mode A possible
  without an observation-shape change.

These values are **provisional until a policy is exported**: the manifest is
generated from the trained checkpoint by an exporter, and the browser trusts the
manifest, not this document. Where the two disagree, the manifest wins and the
loader refuses.

---

## 7. Expression, skills, and what is honestly ready

### Expressive joints (Mode A, buildable now)

The 9 expressive joints are driven by a deterministic controller writing PD targets,
independent of the leg policy, exactly as §9 Mode A describes. This needs no trained
network and works today. Mode B (learned personality-conditioned locomotion) is
gated behind a policy that does not exist.

**The personality selector will not fake style.** Per the Task-2B audit already in
this repo (`docs/locomotion/STYLE_DATA_AUDIT.txt`), the six clips do not contain
separable personality *gaits*, and during walking the between-clip posture
difference is smaller than the within-clip spread. What *is* measurably separable is
expressive-channel activity — RMS tail rate spans 5× (cheeky 1.80 → laidback
0.35 rad/s). So the personality selector drives **head/tail/ear behaviour**, labelled
as such in the UI, and does not claim to change the gait.

### Which motions are actually browser-ready `[FROM REPO]`

From `MEMORY.md`, Stage-4 status is authoritative — a Stage-3 animation existing is
**not** sufficient:

| clip | Stage 4 | simulator status |
|---|---|---|
| Yes (`reactionanimation_test_v4`) | passes, 55 frames, 4/4 contacts | **ready** |
| No (`reaction_no_v4`) | passes, 19 frames, 4/4 contacts | **ready** |
| What (`reaction_what_v4`) | passes, 71 frames, 4/4 contacts | **ready** |
| Cheeky | completes but 1.405 m / 77.8° error — not a holistic pass | **not exposed** |
| Timid | completes but 0.611 m / 11.3° error | **not exposed** |
| DeadPan, Eccentric, Enthusiastic, Laidback | no Stage-4 pass recorded | **not exposed** |

Three expression skills ship. The rest are listed in the UI as unavailable with the
reason, rather than hidden or silently faked.

Skills execute as **joint-target references tracked by the PD controllers** — never
by teleporting the floating root and never by driving the visual rig directly.

### Skill manager

Deterministic state machine, no learned router, per the brief:

```
STAND ⇄ WALK
  ↓        ↓
GESTURE   FALL → RECOVER → STAND
```

Entry/exit conditions use hysteresis and physical state (base height, tilt, contact
count, settle time), not raw key events. `RECOVER` is stubbed until a recovery policy
exists — a fall currently ends in a reported `FALLEN` state rather than a fake
get-up.

---

## 8. Deliverables status

| # | deliverable | status |
|---|---|---|
| 1 | browser application | buildable now |
| 2 | Bingo MJCF model | buildable now |
| 3 | Isaac↔MuJoCo validation tool + report | tool + URDF-side report now; **Isaac side blocked** |
| 4 | ONNX policy loader | buildable now (no policy to load) |
| 5 | versioned policy manifest | buildable now |
| 6 | locomotion controller integration | interface now, **behaviour blocked on training** |
| 7 | expression/personality controller | Mode A now; Mode B blocked |
| 8 | skill manager | buildable now |
| 9 | keyboard/gamepad controls | buildable now |
| 10 | debug HUD | buildable now |
| 11 | README | buildable now |
| 12 | validated-skill list | done — §7, three skills |

---

## Attribution

Architecture studied from Pollen Robotics' Microduck: the sandbox
(via the `Liyucheng1997/318_lab-microduck-simulator` fork of the
`pollen-robotics/microduck-simulator` Space), `pollen-robotics/microduck_rl`
(Apache-2.0), and `pollen-robotics/microduck`. No code or asset from those projects
is copied into this repository; the runtime is reimplemented against Bingo's own
model, meshes and conventions.
