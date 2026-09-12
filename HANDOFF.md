# Handoff — Bingo free locomotion (Task 1), then expressive locomotion (Task 2)

**Branch:** `claude/bingo-locomotion-rl`
**Read this first.** If you are a fresh Claude session, this file plus
`docs/locomotion/TASK1_DESIGN.md` is your context. Do not re-derive it.

---

## 1. What we are trying to do

Bingo is an expressive 21-DOF quadruped. The existing Stage 1–5 pipeline turns
Ashley's authored Blender animation into physically feasible robot motion. That
pipeline **works and is not being changed**.

This branch is a **separate line of work**: teach Bingo to actually *locomote* under
command, rather than replay authored performances.

```
Task 1   command [vx, vy, yaw_rate] -> RL policy -> 12 leg joint targets -> physics
Task 2A  + gait/posture parameters (gait frequency, swing height, body height/pitch)
Task 2B  + personality/style conditioning
```

This is **not imitation learning**. No personality clip is used as a training
reference for Task 1.

### Success for Task 1

Bingo is **actually driven by the velocity command** — forward, backward, lateral,
turn-in-place, start/stop — while staying upright and not saturating its actuators.

Surviving is not success. Oscillating around one pose is not success. A policy that
marches on the spot scores a *perfect* velocity-tracking reward, because its base
velocity is zero; that is precisely why `rl/tools/eval_velocity.py` exists and applies
seven explicit pass/fail gates instead of reporting episode return.

---

## 2. Current state — read this before assuming anything works

Everything on this branch was written in an environment with **no GPU, no Isaac Sim,
no Isaac Lab**. So:

| | |
|---|---|
| environment code | written, never executed |
| API compatibility | **unverified** — checked statically only |
| training | **never run** |
| trained policy | **does not exist** (no `.onnx`, no checkpoint anywhere) |
| Task 2 | not started; blocked on data, see §6 |

What *was* verified, on CPU, and can be trusted:

- The v4 joint limits, the validated standing pose, and the per-joint action scales —
  all measured from the URDF and the real collision hulls.
- The Task-2B motion audit (`rl/tools/analyze_style_motions.py`), run on all nine
  Stage-3 clips.
- The eval battery's metric maths (`rl/tools/test_eval_velocity_metrics.py`,
  4 tests, `python3` only).

**Nothing else has ever run.** Treat the env as a well-argued draft, not as working
code.

---

## 3. Setup

```sh
git clone https://github.com/hassaan141/Blender.git   # or: git fetch origin
cd Blender
git checkout claude/bingo-locomotion-rl
```

Needs: Isaac Lab at `~/robotics/IsaacLab` (Isaac Sim 4.5.0). Everything else the
locomotion work touches — the v4 URDF, collision hulls, the v4 USD, the Stage-3
motions — is tracked in git.

### The path trap — does NOT affect Task 1

28 pre-existing files hardcode `/home/hassaan/Bingo/Blender`. **None of them is on the
Task-1 path**, so the repo can live anywhere and you do not need root:

```sh
grep -rn "/home/hassaan" rl/bingo_rl/bingo_rl/   # 0 hits - the RL package is clean
grep -n  "/home/hassaan" rl/tools/verify_locomotion_api.py \
                         rl/tools/train_velocity.py \
                         rl/tools/play_velocity.py \
                         rl/tools/eval_velocity.py   # 0 hits
```

Every asset the package needs is resolved relative to the source file
(`Path(__file__).resolve().parents[N]`), and `bingo_rl` imports nothing from
`stage2/` or `stage4/`. The hardcoded files are all Stage-2/Stage-4 animation-pipeline
tools (`stage4/*`, `stage2/run_all.sh`, `run_clip.sh`, `rl/tools/replay_v4.py`,
`track_v4_physics.py`, …) which Task 1 never invokes.

**If you later need one of those tools** and cannot `sudo`, rewrite the paths in place
rather than symlinking:

```sh
grep -rl "/home/hassaan/Bingo/Blender" --include=*.py --include=*.sh . \
  | xargs sed -i "s#/home/hassaan/Bingo/Blender#$(pwd)#g"
```

Keep that out of commits unless the whole team moves — `git update-index
--skip-worktree <file>` per file, or just `git checkout -- .` when you are done.

One caveat: this was established by reading the code, not by importing it (the branch
was authored without Isaac Lab). If an import does fail on a missing path, that is new
information — record it rather than assuming the analysis above was right.

Two USD paths *are* referenced by strings the package builds at import time —
`URDF/bingo_urdf_rev_1/...` and `URDF/bingo_urdf_rev_3/...`, the latter gitignored and
absent from a fresh clone. They are only strings in a dataclass and are never opened
unless you spawn a rev_1/rev_3 task. The locomotion tasks spawn the v4 USD
(`rl/v4_usd/bingo_v4.usd`), which is tracked.

---

## 4. The work, in order

### Step 1 — verify the API. Do this before anything else.

The env was written against Isaac Lab's API without being able to import it once.

```sh
cd ~/robotics/IsaacLab
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/verify_locomotion_api.py --headless
```

It exits non-zero and names the broken assumption. **If it fails, fix the config, not
the check** — the checks encode what `docs/locomotion/TASK1_DESIGN.md` assumes.

A static pre-pass already narrowed the risk (`docs/locomotion/API_CORROBORATION.txt`):
**56 of 72** Isaac Lab names used are corroborated by Bingo code that already runs on
this machine. Of the 16 that are not, three matter:

| assumption | if wrong, fall back to |
|---|---|
| `actions.joint_pos.scale` accepts a **dict** | a single scale of **0.125** (SY's headroom); costs SP/knee authority |
| `actions.joint_pos.use_default_offset` exists | set the stance offset in a custom action term |
| `mdp.UniformVelocityCommandCfg.class_type` is subclassable | stock uniform command + an event term that injects zero / turn-in-place |

Five more have **no corroboration at all** — `OnPolicyRunner`, `RslRlVecEnvWrapper`,
`parse_env_cfg`, `load_cfg_from_registry`, `dump_yaml` — because `rl/README.md` points
at `rl/bingo_rl/scripts/train.py` and that file is not on disk. They fail loudly at
import. If they differ, copy the argument handling from Isaac Lab's own bundled
`scripts/reinforcement_learning/rsl_rl/train.py`, which is guaranteed to match.

### Step 2 — the stance gate

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/play_velocity.py \
    --task Bingo-Velocity-StandTest-v4-Play-v0 --checkpoint none --num_envs 4
```

Action scale is 0, so Bingo can only hold `STAND_SOLVED`. If it cannot stand here,
that is not a reward problem and no amount of PPO will fix it.

### Step 3 — train the curriculum

Eleven cumulative stages, `Bingo-Velocity-Flat-v4-S1-v0` … `-S11-v0`. Observation and
action shapes are identical across all of them by construction, so each warm-starts
from the previous checkpoint.

```
S1  forward only, nominal physics      S7   observation noise
S2  + stand (exact zero command)       S8   + friction variation
S3  + wider velocity range             S9   + mass variation
S4  + turning                          S10  + reset-state variation
S5  + reverse / lateral                S11  + pushes
S6  + start/stop transitions
```

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/train_velocity.py \
    --task Bingo-Velocity-Flat-v4-S1-v0 --num_envs 4096 --headless \
    --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"

./isaaclab.sh -p ~/Bingo/Blender/rl/tools/train_velocity.py \
    --task Bingo-Velocity-Flat-v4-S2-v0 --resume_from <S1 ckpt>.pt --headless
```

**Do not jump to S11.** Aggressive randomisation before a gait exists is how these
runs fail, and `improved_walking_cfg.py` records that lesson from this project's own
history (v3 dragged on three legs; v6 parked one foot as a static prop).

### Step 4 — evaluate against the gates

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/eval_velocity.py \
    --checkpoint <ckpt>.pt --headless --video \
    --report docs/locomotion/EVAL_S<n>.txt --out docs/locomotion/eval_s<n>.json
```

14 fixed command segments; seven gates: drives forward, drives backward, turns in
place, moves laterally, stands still on zero command, no falls, torque not saturated.

---

## 5. Things already found — do not rediscover these

**The default stance in `bingo_v4.py` is wrong for v4.** `init_state.joint_pos` still
carries the rev_3 pose (`SP ±0.3, knee ±0.6`). Measured against the real collision
hulls it puts one paw 207 mm in front of the base while another is at 24 mm, paws
8.14 mm out of level, and the base origin ends up **outside** the support polygon —
a pose that must topple the instant gravity is applied. `stage4/stand_test.py`
independently records it jamming `fr_SP_J` into its +1.56 limit.

The locomotion env overrides it locally with `STAND_SOLVED` (all four paws at exactly
−0.180 m, spread 0.00 mm, base height 0.182 m). **`bingo_v4.py` is deliberately NOT
edited** — it is validated Stage-4 physics and Stage 5 depends on it, and RSI
overwrites the init state there anyway. Keep it that way.

**Action scale is per joint, not a scalar.** SY 0.125 / SP 0.195 / knee 0.170, chosen
so `|action| = 3` lands exactly on each joint's soft limit given its headroom from the
stance. Stage 5's single 0.3 would ask SY for 0.90 rad against 0.378 of headroom —
the SY overrun `MEMORY.md` records as warned-but-unsolved. SY's range is 3.7× smaller
than SP's; one number cannot serve both.

**Observation is 66, and all 21 joints are in it** even though only 12 are actioned —
so Task 2 can move the expressive joints without an observation-shape change. Layout
verified by arithmetic against a known-good figure: `improved_walking_cfg.py` records
58 for the 17-joint rev_3 robot, and 3+3+3+3+**17**+**17**+12 = 58.

**Control rate is 24 Hz** (120 Hz physics, decimation 5), matching Stage 4 and
Stage 5 — *not* the 50 Hz typical of Isaac Lab locomotion tasks. Any weight inherited
from a 50 Hz config that scales with control rate (`action_rate_l2` above all) is a
candidate for re-measurement, not a validated value.

---

## 6. Task 2 is blocked on data, not on code

Before writing any style-conditioned policy, read
`docs/locomotion/STYLE_DATA_AUDIT.txt`. Measured on all nine Stage-3 clips:

| clip | verdict | why |
|---|---|---|
| deadpan | USABLE | 2.48 feet down, Froude 0.02, 7 cycles |
| laidback | USABLE | 2.56 feet down, Froude 0.03, 6 cycles |
| timid | MARGINAL | only 2 cycles on the least-active foot |
| cheeky | NOT LOCOMOTION | **1.05 feet down** (airborne), peak 3.31 m/s, **Froude 6.20** |
| enthusiastic | NOT LOCOMOTION | travels 0.045 m net — in place |
| eccentric | NOT LOCOMOTION | hind paws never touch down (authored sit) |

And during the *walking* segments specifically, the between-clip body-height
difference (Deadpan 173.2 vs Laidback 138.5 mm = **34.7 mm**) is **smaller than the
within-clip standard deviation of either** (±28.9 and ±44.3 mm). Walking speed is
effectively constant across all six (0.061–0.140 m/s).

**Personality *gait* transfer is not supported by this data.** Do not synthesise a
"Cheeky gait" and attribute it to Ashley's animation.

What *is* separable is expressive-channel activity — RMS joint rate spans 5× on the
tail (cheeky 1.80 → laidback 0.35 rad/s) and 3× on the head. That is the defensible
version of Task 2B today: style conditioning on head/tail/ear behaviour layered on
the Task-1 policy.

To unblock real personality-gait work, the capture needs, per personality: a
straight-line walk ≥5 s at roughly constant speed, all four paws cycling, ≥3 clean
touchdown-to-touchdown cycles per foot, mean feet-down ≥1.5, peak root speed under
2.3 m/s (Froude 3). That is spec §B.2 with acceptance criteria attached.

---

## 7. Rules

1. **Do not modify the Stage 1–5 pipeline** — `stage2/`, `stage4/`, `stage5/`,
   `motions/`, `blend_sources/`, or `bingo_v4.py`. Unless a reproducible physics
   defect demands it, and then say so explicitly.
2. **Do not fake locomotion.** No animation playback, no root teleportation, no
   hand-written gait equations. Walking comes from policy inference or it does not
   happen yet.
3. **Stage-4 status is authoritative.** A Stage-3 animation existing does not make a
   clip usable.
4. **Report honestly.** If something was not run, say NOT RUN. The value of this
   branch so far is that its claims are tagged `[MEASURED]` / `[FROM REPO]` /
   `[INFERRED]`; keep that discipline.
5. Measure before tuning. Every weight in the reward that differs from stock carries
   its justification inline — add yours the same way.

## 8. Where things are

```
rl/bingo_rl/bingo_rl/locomotion/     the environment
  bingo_velocity_env_cfg.py          cfg + the 11 curriculum stages
  bingo_velocity_mdp.py              categorical command + Bingo reward terms
  agents/rsl_rl_ppo_cfg.py           PPO config
rl/tools/
  verify_locomotion_api.py           RUN FIRST
  train_velocity.py  play_velocity.py  eval_velocity.py
  analyze_style_motions.py           Task 2B audit (CPU)
  test_eval_velocity_metrics.py      offline tests (CPU)
docs/locomotion/
  TASK1_DESIGN.md                    the spec, with evidence tags
  STYLE_DATA_AUDIT.txt               what the personality clips contain
  API_CORROBORATION.txt              which API names are corroborated
  README.md                          run order
```

Also read `MEMORY.md` — the project's own source of truth for Stages 1–5, and
authoritative over `README.md`, which still describes an older layout.

The commit messages on this branch carry the measurements and the reasoning,
including what failed. They are long on purpose.
