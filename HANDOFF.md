# Handoff — Bingo browser simulator

**Branch:** `claude/bingo-web-simulator`
**Read this first**, then `bingo-simulator/ARCHITECTURE.md` and
`bingo-simulator/README.md`. Do not re-derive what is already measured here.

---

## 1. What this is

A browser simulator for Bingo, an expressive 21-DOF quadruped: real MuJoCo physics
compiled to WebAssembly, the real v4 robot model, real expression controllers, real
Stage-4-validated motions.

```
user command -> controller/policy -> 21 joint targets -> MuJoCo WASM -> rendered robot
```

**It is not a website that plays Blender animations.** Every pixel of robot motion
comes out of the physics solver. The architecture reference is Pollen Robotics'
[Microduck Sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-simulator),
studied from source; **no Microduck code or asset was copied** (their sandbox fork has
no LICENSE, their 3D assets are CC BY-SA-NC). `ARCHITECTURE.md` is the attribution
record.

### Runs anywhere — no Isaac Sim, no GPU

MuJoCo runs on CPU. The whole branch was built and verified without a GPU.
The **only** thing needing Isaac is the trace comparison in §4, which is blocked on a
policy anyway.

---

## 2. Current state

### Done and measured

| | evidence |
|---|---|
| MJCF physics twin, generated from the v4 URDF | **11/11** checks, `docs/VALIDATION_REPORT.txt` |
| Render rig driven purely from MuJoCo | **0.02 mm** worst link disagreement, `npm run check:rig` |
| Geometry pipeline | 45.4 MB → 5.8 MB (hull collision + decimated render) |
| Browser app running in Chromium | 24.6 Hz control / 123 Hz physics, 4/4 paw contacts, 0/21 saturated |
| Expression system | all 9 head/tail/ear joints, 7 personality presets |
| Skill manager | deterministic state machine, 3 Stage-4-validated gestures |
| Controls + HUD | keyboard, gamepad, public + engineering HUD |

Selected numbers worth trusting: total mass matches the URDF to 9 decimals
(2.459195327 vs 2.459195326 kg); standing joint error **0.0043 rad** against Isaac's
own documented `stand_test` figure of **0.0041 rad**; FK agrees with `V4Kin` to
0.0001 µm over 500 random poses.

### Not done

- **No walking.** No locomotion policy has ever been trained — that work is on
  `claude/bingo-locomotion-rl` and has not run either. The app shows a `NO_POLICY`
  banner, the skill manager refuses to enter `WALK`, and pressing W does nothing.
  **This is deliberate. Do not "fix" it with an animation.**
- **Isaac ↔ MuJoCo trace comparison never ran** (§4).
- No recovery policy — a fall ends in a reported `FALLEN` state, not a get-up.
- No props (ball, box, ramp). The brief puts them after flat-ground validation.
- No touch/mobile controls.
- Head renders faceted — raise `export_render_model.py --render-budget 250000`.

---

## 3. Setup

```sh
git clone https://github.com/hassaan141/Blender.git   # or: git fetch origin
cd Blender
git checkout claude/bingo-web-simulator

cd bingo-simulator/app
npm install                  # .npmrc sets legacy-peer-deps; see §6
../tools/vendor_runtime.sh   # copies MuJoCo + ONNX Runtime out of node_modules
npm run dev                  # http://localhost:5173
```

CPU-only verification, no browser needed:

```sh
python3 bingo-simulator/tools/validate_isaac_mujoco.py    # expect 11/11
cd bingo-simulator/app && npm run build && npm run check:rig
```

`app/public/vendor/` is gitignored (~40 MB of WASM). `package.json` pins the versions
and `tools/vendor_runtime.sh` reproduces it byte for byte.

### If you are on a headless box

The app still works; drive it with Playwright. Chromium needs
`--no-sandbox --use-gl=angle --use-angle=swiftshader`. `npm run check:rig` already does
this and is the template. **Never tell the operator to "open localhost" or "watch the
robot" — they may have no display.** Produce a screenshot or video, print its absolute
path, and attach it.

---

## 4. The one gap that needs a GPU

The MJCF is verified against the **v4 URDF** — the same model Isaac consumes, via the
independent `V4Kin` FK that the whole Stage-2/4 pipeline already trusts. That catches
conversion errors. It does **not** prove dynamic agreement with Isaac.

`validate_isaac_mujoco.py` prints `NOT RUN` for that section rather than implying
parity. To close it, on a machine with Isaac Lab and a trained policy:

```sh
./isaaclab.sh -p rl/tools/eval_velocity.py --checkpoint <ckpt>.pt --out isaac_trace.json
python3 bingo-simulator/tools/validate_isaac_mujoco.py --isaac-trace isaac_trace.json
```

The `--isaac-trace` branch is a stub. Implementing the comparison is legitimate work.

---

## 5. Wiring in a policy when one exists

1. Export the checkpoint to ONNX → `bingo-simulator/app/public/policies/`.
2. Add an entry under `policies` in `app/public/policy_manifest.json`, move `status`
   off `NO_POLICY`.
3. The loader **refuses** a policy whose manifest or ONNX graph disagrees with the
   runtime: obs dim **66**, action dim **12**, **24 Hz**, canonical joint order.

**Do not loosen those checks to make a policy load.** Fix whichever side is wrong. On
a quadruped a silently reordered joint vector does not throw — it walks wrong, and
looks like a bad policy rather than a loading bug. That refusal behaviour is copied
from Microduck's daemon, which exists to prevent exactly this.

The contract (`docs/locomotion/TASK1_DESIGN.md` on the other branch):

```
66 = base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + command 3
   + joint_pos_rel 21 + joint_vel 21 + last_action 12

q_target[j] = stance[j] + action_scale[j] * action[j]        (12 leg joints)
action_scale = SY 0.125 / SP 0.195 / knee 0.170              (per joint, not scalar)
```

**24 Hz, not 50.** Microduck runs 50 because that is *its* training rate; Bingo's
Stage-4/5 stack is 120 Hz physics with decimation 5. Copying the number rather than
the principle would silently mis-feed any policy trained here.

All 21 joints are observed while only 12 are actioned, so the expressive joints are
already in the policy's observation and Task 2 needs no obs-shape change.

---

## 6. Things already found — do not rediscover these

**Four collision meshes in the v4 URDF are degenerate** — exactly 12 triangles and
0.0000 mm³: `tail_pitch`, `head_yaw`, `l_ear_pitch`, `r_ear_pitch`. MuJoCo's URDF
importer refuses the model outright because of them. They are the same four
near-massless intermediate links `bingo_v4.py` documents as the articulation-
conditioning problem. The generator sidesteps it properly by using the URDF's explicit
`<inertial>` per link, so mesh volume is never needed.

**Self-collision must be OFF.** Isaac spawns this robot with
`enabled_self_collisions=False`, so the validated Stage-4 physics has never had
robot-vs-robot contacts — and MuJoCo's default is on. With it on, the head rested
against the torso and `head_yaw` sat pinned at its 6 N·m ceiling against a −8 N·m
contact force, reaching 0.014 rad of a commanded 0.200. Encoding Isaac's setting via
`contype`/`conaffinity` moved standing joint error 0.1538 → 0.0043 rad and contact
points 30 → 7.

**Paw collision geometry is not coarsened like the rest.** A uniform 256-face hull cap
looked fine and measurably changed the physics: the stance rose 2.3 mm and one paw
stopped touching (3/4 contacts). The four `*_knee` links get their own 4096-face cap.

**Dynamic `import()` needs an absolute URL.** A relative specifier resolves against the
importing *module*, not the page, so after a production build
`./vendor/mujoco/mujoco.js` became `/bundle/vendor/...` and 404'd — while every
`fetch()` kept working, because fetch resolves against the document.

**The MuJoCo bindings are the official DeepMind `@mujoco/mujoco`**, not the older
community `mujoco_wasm`: `MjModel.from_xml_path` (static, not `new Model(...)`), enums
via `mjtObj.mjOBJ_*.value`, and `data.contact` is an Embind vector with `size()`/`get()`
that goes stale every step and must be `delete()`d.

**STL is non-indexed**, so `computeVertexNormals()` gives per-face normals and the robot
renders flat-shaded. Weld, then `toCreasedNormals` at 36°.

**npm needs `legacy-peer-deps`** — R3F v9 declares `expo` as a peerOptional and with
React 19 present npm walks into expo's peer set and fails ERESOLVE. `.npmrc` handles it.

---

## 7. Rules

1. **Never fake locomotion.** No animation playback, no root teleportation, no
   hand-written gait equations, no independently animated Three.js skeleton. If there
   is no policy, the honest output is that the robot stands still and says so.
2. **The render rig is driven entirely from MuJoCo state.** If you touch
   `renderRig.js` or `GameCanvas.jsx`, re-run `npm run check:rig` and report the number.
3. **Do not change Bingo's morphology** — URDF geometry, pivots, limits, masses,
   collision geometry — to make anything easier.
4. **Do not modify the Stage 1–5 pipeline** (`stage2/`, `stage4/`, `stage5/`,
   `motions/`, `blend_sources/`, `bingo_v4.py`).
5. **Stage-4 status is authoritative.** Only Yes/No/What have a real Stage-4 pass; the
   other six are shown disabled *with the reason*, not hidden. Do not promote a skill
   without quantitative evidence.
6. **Personality changes head/tail/ear behaviour, not gait**, and the UI says so. The
   Task-2B audit found the six clips contain no separable personality gait — during
   walking the between-clip body-height difference (34.7 mm) is smaller than the
   within-clip standard deviation of either (±28.9, ±44.3 mm). The presets are built
   from measured per-clip RMS joint rates, which do differ 5× on the tail.
7. **Report honestly.** If something was not run, say NOT RUN.

---

## 8. Where things are

```
bingo-simulator/
├── ARCHITECTURE.md            design report + the Microduck study + attribution
├── README.md                  setup, architecture, validation, limitations
├── docs/VALIDATION_REPORT.txt the 11/11 output
├── tools/
│   ├── export_bingo_mujoco.py   URDF -> MJCF
│   ├── export_render_model.py   meshes -> collision hulls + render meshes
│   ├── export_motions.py        Stage-4 motions -> JSON
│   ├── validate_isaac_mujoco.py the 11 checks
│   └── vendor_runtime.sh        MuJoCo + ORT out of node_modules
└── app/
    ├── src/game/              framework-independent core (no React, no Three)
    │   ├── constants.js         the control contract
    │   ├── physics/mujoco.js    WASM boot, name-based index maps
    │   ├── policies/loader.js   ONNX + manifest validation
    │   ├── controllers/         expression layer
    │   ├── skills/              skill router + motion playback
    │   ├── controls/            keyboard, gamepad
    │   └── runtime/sim.js       the fixed-step loop
    ├── src/scene/             R3F canvas + render rig
    ├── src/ui/                HUD + skill bar
    ├── tools/check_rig_vs_physics.mjs   rig vs MuJoCo, in a real browser
    └── public/robot|motions|policies
```

Also read `MEMORY.md` at the repo root — the project's source of truth for Stages 1–5,
authoritative over the root `README.md`, which describes an older layout.

The commit messages on this branch carry the measurements and the reasoning, including
what failed and why. They are long on purpose.
