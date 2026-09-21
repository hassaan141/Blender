# Bingo locomotion — run order

Free (command-conditioned) locomotion and, later, expressive locomotion. This is a
**separate branch of work** from the Stage 1–5 animation pipeline: it reads no
motion reference, and modifies nothing under `stage2/`, `stage4/`, `stage5/`,
`motions/` or `blend_sources/`.

Read [TASK1_DESIGN.md](TASK1_DESIGN.md) first — it is the design report, and it
records which claims are measured, which are read from working repo code, and which
are unverified assumptions about the local Isaac Lab API.

## 0. Verify the API — before anything else

The locomotion module was written without access to an Isaac Lab install, so every
API assumption is checked by one script. **Run it first.** It exits non-zero on any
failure and tells you exactly which assumption broke.

```sh
cd ~/robotics/IsaacLab
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/verify_locomotion_api.py --headless
```

It checks: module paths, `mdp` function names, config attribute names, that the
actuators resolve to explicit `IdealPD` (implicit drives do **not** enforce effort
limits on this robot), the 21 joint names, the resolved 66-dim observation, the
12-dim action, that the stance pose is `STAND_SOLVED` and inside the joint limits,
that the action scale leaves headroom at |action| = 3, that every command category
is reachable including exact zero and turn-in-place, and a 48-step zero-action
rollout for NaNs.

## 1. Stance gate

If Bingo cannot hold its own stance under its own drives, no reward will fix it.
Action scale is 0 in this task, so the robot can only hold `STAND_SOLVED`.

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/play_velocity.py \
    --task Bingo-Velocity-StandTest-v4-Play-v0 --checkpoint <any> --num_envs 4
```

Watch for: does it settle, do all four paws stay down, does the base stay near
0.182 m? This mirrors `stage4/stand_test.py`, but inside the RL env so the env's own
spawn height, physics and contact setup are what gets tested.

## 2. Train the curriculum

Eleven stages, cumulative. Randomisation stays off until a gait exists — this
project has already learned that lesson once (see `improved_walking_cfg.py`).

| stage | adds | | stage | adds |
|---|---|---|---|---|
| S1 | forward only, nominal physics | | S7 | observation noise |
| S2 | + stand (exact zero command) | | S8 | + friction variation |
| S3 | + wider velocity range | | S9 | + mass variation |
| S4 | + turning | | S10 | + reset-state variation |
| S5 | + reverse / lateral | | S11 | + pushes |
| S6 | + start/stop transitions | | | |

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/train_velocity.py \
    --task Bingo-Velocity-Flat-v4-S1-v0 --num_envs 4096 --headless \
    --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"

# then warm-start each later stage from the previous checkpoint
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/train_velocity.py \
    --task Bingo-Velocity-Flat-v4-S2-v0 --resume_from <S1 checkpoint>.pt --headless
```

Observation and action shapes are identical across all 11 stages by construction, so
a checkpoint always loads.

## 3. Evaluate — this is what defines "done"

```sh
./isaaclab.sh -p ~/Bingo/Blender/rl/tools/eval_velocity.py \
    --checkpoint <run>/model_1499.pt --headless \
    --report ~/Bingo/Blender/docs/locomotion/EVAL_S<n>.txt \
    --out ~/Bingo/Blender/docs/locomotion/eval_s<n>.json --video
```

Fourteen fixed command segments, then seven pass/fail gates: drives forward, drives
backward, turns in place, moves laterally, stands still on zero command, no falls,
torque not saturated.

**Task 1 is complete only when Bingo is actually driven by the command.** Surviving,
or oscillating around one pose, fails these gates by design — a policy that marches
on the spot scores a perfect velocity-tracking reward, because its *base* velocity
is zero, which is exactly why the `stand_still` term and these gates exist.

## Offline checks (no Isaac needed)

```sh
python3 rl/tools/test_eval_velocity_metrics.py       # eval metric maths
python3 rl/tools/analyze_style_motions.py            # Task 2B data audit
python3 rl/tools/plot_stance_and_action_scale.py     # stance + action-scale figure
```

## Task 2 status

Not started. The data audit that Task 2B depends on **is** done, and its conclusion
constrains what Task 2 can honestly claim — see §4 of the design report and
[STYLE_DATA_AUDIT.txt](STYLE_DATA_AUDIT.txt). Short version: only Deadpan and
Laidback are physically plausible walking clips; the between-clip posture difference
during walking is smaller than the within-clip spread; what *is* separable is
expressive-channel (head/tail/ear) activity, which varies 3–5× across clips.
Personality **gait** transfer is not supported by today's data, and the additional
authored locomotion needed to support it is specified with acceptance criteria in
the report.

## Files

```
rl/bingo_rl/bingo_rl/locomotion/
├── __init__.py                  gym registration (flat, play, stand-test, S1..S11)
├── bingo_velocity_env_cfg.py    the environment
├── bingo_velocity_mdp.py        categorical command + Bingo-specific reward terms
└── agents/rsl_rl_ppo_cfg.py     PPO runner config

rl/tools/
├── verify_locomotion_api.py     run first
├── train_velocity.py            staged training
├── play_velocity.py             watch a checkpoint
├── eval_velocity.py             the battery that defines "done"
├── test_eval_velocity_metrics.py
├── analyze_style_motions.py     Task 2B data audit
└── plot_stance_and_action_scale.py
```
