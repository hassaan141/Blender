# Handoff — picking this up on a machine with compute

Read this first if you are a fresh Claude session, or a human setting one up.

Two branches of work were built in an environment with **no GPU, no Isaac Sim, no
Isaac Lab and no Blender**. Everything that could be verified without them was
verified and the numbers are in the repo; everything that could not is labelled
`NOT RUN` rather than assumed. This file says exactly which is which, and what to do
first.

---

## 1. Getting it onto the box

Nothing needs copying by hand. It is all in git.

```sh
# if the repo is not there yet
git clone https://github.com/hassaan141/Blender.git
cd Blender

# if it is already there
git fetch origin

git branch -a | grep claude/
```

| branch | what is on it |
|---|---|
| `claude/bingo-locomotion-rl` | Task 1 command-conditioned locomotion env + tools, Task 2B data audit |
| `claude/bingo-web-simulator` | browser simulator: MuJoCo physics twin, render rig, skills, HUD |

They are independent and both branch off `main`. Check out whichever you want to work
on. The clone is ~230 MB and includes every asset the tools need — the v4 URDF, the
collision hulls, the Stage-3/4 motions and the v4 USD are all tracked.

### The one thing that will bite you

**28 pre-existing files hardcode `/home/hassaan/Bingo/Blender`** — `run_clip.sh`, all
of `stage4/`, most of `rl/tools/`. If the checkout lives anywhere else they break with
confusing "file not found" errors.

```sh
# check
grep -rl "/home/hassaan" --include=*.py --include=*.sh . | wc -l    # 28

# cheapest fix: make the old path real
sudo mkdir -p /home/hassaan/Bingo
sudo ln -s /actual/path/to/Blender /home/hassaan/Bingo/Blender
```

Everything added on these two branches uses relative paths and does not care where the
repo lives.

### Not in the clone

`.gitignore` excludes `blend/`, `raw/`, `output/`, `bingo_urdf_rev_3/`, `*.stp` and the
NVIDIA residential assets. None of it is needed for either branch. You also need,
separately:

- **Isaac Lab** at `~/robotics/IsaacLab` (Isaac Sim 4.5.0) — for the locomotion branch
- **Blender 5.2** at `~/Bingo/local/blender-5.2.0-linux-x64/blender` — only for the
  Stage 1-5 animation pipeline, which neither branch touches
- **Node 20+** — only for the simulator branch

---

## 2. `claude/bingo-locomotion-rl` — run this first

The Task-1 environment exists and has **never been executed**. Not trained, not even
API-checked, because that needs Isaac Lab. There is no `.onnx` anywhere in the repo.

### Step 1 — verify the API before anything else

The env was written against Isaac Lab's API without being able to import it. Every
assumption is checked by one script:

```sh
cd ~/robotics/IsaacLab
./isaaclab.sh -p <repo>/rl/tools/verify_locomotion_api.py --headless
```

It exits non-zero and names the broken assumption. A static pre-check already narrowed
the risk: **56 of 72 Isaac Lab names used are corroborated** by Bingo code that has
already run on your machine; 16 are not. Three of those matter
(`docs/locomotion/TASK1_DESIGN.md`, "What is actually unverified"):

| assumption | fallback if wrong |
|---|---|
| `actions.joint_pos.scale` accepts a **dict** | one scale of 0.125 (SY's headroom) |
| `actions.joint_pos.use_default_offset` exists | set the offset in a custom action term |
| `mdp.UniformVelocityCommandCfg.class_type` is subclassable | stock command + an event term for zero/turn-in-place |

Five more (`OnPolicyRunner`, `RslRlVecEnvWrapper`, `parse_env_cfg`,
`load_cfg_from_registry`, `dump_yaml`) have **no corroboration at all**, because
`rl/README.md` points at `rl/bingo_rl/scripts/train.py` and that file is not on disk.
They fail loudly at import; copy the argument handling from Isaac Lab's own bundled
`scripts/reinforcement_learning/rsl_rl/train.py` if they differ.

### Step 2 — the stance gate

```sh
./isaaclab.sh -p <repo>/rl/tools/play_velocity.py \
    --task Bingo-Velocity-StandTest-v4-Play-v0 --checkpoint none --num_envs 4
```

Action scale is 0, so the robot can only hold `STAND_SOLVED`. If it cannot stand, that
is not a reward problem and training will not fix it.

### Step 3 — train the curriculum

Eleven cumulative stages, `Bingo-Velocity-Flat-v4-S1-v0` … `-S11-v0`. Obs and action
shapes are identical across all of them, so each warm-starts from the last.

```sh
./isaaclab.sh -p <repo>/rl/tools/train_velocity.py \
    --task Bingo-Velocity-Flat-v4-S1-v0 --num_envs 4096 --headless \
    --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"
```

Do not jump to S11. Aggressive randomisation before a gait exists is how these runs
fail, and `improved_walking_cfg.py` records that lesson from this project's own history.

### Step 4 — the completion gate

```sh
./isaaclab.sh -p <repo>/rl/tools/eval_velocity.py --checkpoint <ckpt>.pt --headless \
    --report docs/locomotion/EVAL_S<n>.txt --video
```

14 fixed command segments, seven pass/fail gates. **Task 1 is complete only when Bingo
is actually driven by the command** — a policy that survives while marching on the spot
scores a perfect velocity-tracking reward and fails these gates by design.

### Two things already found and worked around

- `BINGO_V4_CFG.init_state` still carries the **rev_3 pose**, which is not a valid v4
  stance: measured against the real hulls it puts one paw 207 mm in front of the base
  while another is at 24 mm, and the base origin ends up outside the support polygon.
  The locomotion env overrides it locally with `STAND_SOLVED`. `bingo_v4.py` is
  deliberately untouched — Stage 5 depends on it and RSI overwrites the init state
  there anyway.
- Action scale is **per joint** (SY 0.125 / SP 0.195 / knee 0.170), not Stage 5's
  single 0.3, which would ask SY for 0.90 rad against 0.378 of headroom.

### Task 2 is blocked on data, not code

`docs/locomotion/STYLE_DATA_AUDIT.txt`: of the six personality clips, only Deadpan and
Laidback are physically plausible walking clips, and during walking the between-clip
body-height difference (34.7 mm) is **smaller than the within-clip standard deviation**
of either (±28.9 and ±44.3 mm). Cheeky averages 1.05 feet down at Froude 6.2 — authored
animation, not locomotion.

Personality *gait* transfer is not supported by this data. What is separable is
expressive-channel activity (tail RMS rate spans 5×). The acceptance criteria for the
extra capture that would unblock real Task 2B are in the design report §4.

---

## 3. `claude/bingo-web-simulator`

```sh
cd bingo-simulator/app
npm install                  # .npmrc sets legacy-peer-deps for R3F v9 + React 19
../tools/vendor_runtime.sh   # copies MuJoCo + ONNX Runtime out of node_modules
npm run dev
```

`app/public/vendor/` is gitignored (~40 MB of WASM); `vendor_runtime.sh` reproduces it
from the versions `package.json` pins.

Already verified in a real browser (Chromium + Playwright): 24.6 Hz control / 123 Hz
physics, 4/4 paw contacts, 0/21 actuators saturated, gestures executing from their
Stage-4 references.

```sh
python3 bingo-simulator/tools/validate_isaac_mujoco.py   # 11/11, CPU only
cd bingo-simulator/app && npm run check:rig              # rig vs MuJoCo, 0.02 mm
```

### The one gap that needs your GPU

The **Isaac ↔ MuJoCo command-trace comparison never ran.** The MJCF is verified against
the v4 URDF — the same model Isaac consumes — which catches conversion errors but does
not prove dynamic agreement. To close it:

```sh
# on this box, once a policy exists
./isaaclab.sh -p rl/tools/eval_velocity.py --checkpoint <ckpt>.pt --out isaac_trace.json
python3 bingo-simulator/tools/validate_isaac_mujoco.py --isaac-trace isaac_trace.json
```

The tool currently prints `NOT RUN` for that section rather than implying parity.

One encouraging signal in the meantime: MuJoCo's standing joint error is **0.0043 rad**
against Isaac's own documented `stand_test` figure of **0.0041 rad**.

### Wiring a trained policy in

1. Export the checkpoint to ONNX, put it in `bingo-simulator/app/public/policies/`.
2. Add an entry to `app/public/policy_manifest.json` under `policies` and set
   `status` away from `NO_POLICY`.
3. The loader **refuses** a policy whose manifest or ONNX graph disagrees with the
   runtime — obs dim 66, action dim 12, 24 Hz, canonical joint order. Do not loosen
   those checks to make a policy load; fix whichever side is wrong. On a quadruped a
   silently reordered joint vector does not throw, it just walks wrong.

---

## 4. What to tell a fresh Claude session

Point it at this file, then:

- `MEMORY.md` — the project's own source of truth for Stages 1-5. Authoritative over
  `README.md`, which still describes the older Path A / Path B layout.
- `docs/locomotion/TASK1_DESIGN.md` — obs/action/reward spec, and the evidence tags
  (`[MEASURED]` / `[FROM REPO]` / `[INFERRED]`) saying what is actually known.
- `bingo-simulator/ARCHITECTURE.md` — the simulator design and the Microduck study.
- The commit messages on both branches carry the measurements and the reasoning,
  including the failures. They are long on purpose.

Two standing rules from this work worth repeating to it:

1. **Stage-4 status is authoritative.** A Stage-3 animation existing does not make a
   clip usable. Only Yes/No/What have a real Stage-4 pass.
2. **Do not fake locomotion.** No animation playback, no root teleportation. If there
   is no policy, the honest output is that the robot stands still and says so.
