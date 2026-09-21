# Bingo Locomotion (Task 1) — Status & Plan
**Branch:** `claude/bingo-locomotion-rl` · **Machine:** Kiwi (`/pub0/muhammadf/Blender`) · **Date:** 2026-09-15

## 0. Why this doc exists

Picking this branch back up this session, my own memory of it (written 2026-09-13)
was stale — training had progressed well past what that memory recorded. This is a
fresh read of `git status`, the training logs, and every `EVAL_S*.txt` on disk, plus
the plan agreed with you this session: **investigate the torque-saturation gate
before pushing further into the curriculum.**

Nothing in this document has been executed yet except the read-only investigation in
§4 and the recording in §6. §5 is a plan, not a diff.

---

## 1. Project structure — where everything lives

Two things matter here: (a) the validated Stage 1–5 animation pipeline, which
**must not be touched**, and (b) the Task 1 locomotion controller, which is what
this branch actually builds. Everything else in the repo (there's a lot) is
either infrastructure both depend on, or unrelated side-work.

### 1a. The Stage 1–5 animation pipeline — DO NOT TOUCH

Separate project, separate goal: take Ashley's (the animator's) authored Blender
performances and get them running **as-authored** on the physical Bingo v4 robot in
Isaac Sim — replay, not from-scratch locomotion. This is the pipeline `HANDOFF.md`
rule 1 protects: don't modify `stage2/`, `stage4/`, `stage5/`, `motions/`,
`blend_sources/`, or `bingo_v4.py` unless a reproducible physics defect demands it,
and say so explicitly if it ever does. Source of truth for this section: root
`MEMORY.md` (authoritative over `README.md`, which describes an older layout).

| Stage | What it does | Status | Key files |
|---|---|---|---|
| **1 — Exact Blender v4 rig** | The animator-facing rig, built directly from the URDF so bones sit at the real joint positions/axes/limits. | Validated | `blend_sources/Bingo_V4_AnimatorRig.blend` (armature `Bingo_Robot`); kinematic URDF `URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf`; physics URDF `..._physics.urdf`; physics config `rl/bingo_rl/bingo_rl/bingo_v4.py` (actuator Kp/Kd/effort limits — **this is the file Task 1 also reuses read-only, see 1b**). 21 actuated joints: 12 leg + 3 head + 2 tail + 4 ear. |
| **2 — Spatial retargeting** | Solves Ashley's legacy art-rig performance onto Bingo's actual leg kinematics (`SY→SP→knee`, no ankle/paw joint). | Working; validated per-clip | `stage2/extract_source_motion.py`, `detect_contacts.py`, `solve_spatial_retarget.py`, `v4_kinematics.py`, `bake_v4_motion.py`, `evaluate_retarget.py`. Input: `blend_sources/Bingo_<Personality>.blend`. Output: `stage2/out/<clip>_retarget.npz` + `blend_sources/Bingo_<Personality>_V4_Retargeted.blend`. |
| **3 — Exact kinematic Isaac replay** | Gravity-off replay of the retargeted motion, root pose + all 21 joints, frame-by-frame — proves the retarget reproduces Blender's FK before spending any physics/GPU time. | Validated (zero joint-write error) | Export `scripts/bake_conform.py`; viewer `rl/tools/replay_v4.py`; output `motions/<clip>_v4.npz` (`cheeky_v4.npz`, `deadpan_v4.npz`, `timid_v4.npz`, `laidback_v4.npz`, `enthusiastic_v4.npz`, `eccentric_v4.npz`, `reaction_{yes,no,what}_v4.npz`). |
| **4 — Full-physics feasibility** | Gravity + contacts + the real `IdealPDActuator` model, 24 Hz reference targets interpolated at 120 Hz physics. This is where the leg PD gains and 3.0 N·m effort limit in `bingo_v4.py` were derived and validated (`stage4/actuator_analysis.py`). | Infrastructure validated; clip-specific (Cheeky/Timid not yet a full pass) | `rl/tools/track_v4_physics.py`, `stage4/contact_model.py`, `build_contact_model.py`, `dynamic_retarget.py`, `dynamic_audit.py`, `stand_test.py`, `ground_fix.py`, `actuator_analysis.py`, `balance_adjust.py`. Output: `stage4/out/<clip>_v4_stage4.npz` + `.csv` audit. |
| **5 — Residual RL polish** | Small learned correction (`q_target = q_ref + 0.3 × action`) on top of the Stage 4 reference, legs only — head/tail/ear stay feed-forward. | Implemented/trained for Timid only; experimental, not general | `rl/bingo_rl/bingo_rl/stage5/bingo_stage5_env.py` (+ `_cfg.py`); PPO config reused from `rl/bingo_rl/bingo_rl/track/agents/skrl_ppo_cfg.yaml`; eval `rl/tools/eval_stage5.py`. Tasks `Bingo-Stage5-Timid-{Direct,Play}-v0`. Checkpoint: `logs/skrl/bingo_stage5_timid/2026-08-26_13-59-06_ppo_torch/`. |

Note the directory names don't line up 1:1 with the numbers — there's no `stage1/`
or `stage3/` top-level folder; Stage 1 lives in `blend_sources/` + the URDF, and
Stage 3 is just `scripts/bake_conform.py` + `rl/tools/replay_v4.py` output. `stage2/`,
`stage4/`, `stage5/` are the only stages with dedicated code directories.

### 1b. The Task 1 locomotion controller — what this branch actually builds

**Entirely separate objective from the pipeline above.** Stage 1–5 replays authored
performances; Task 1 is **from-scratch command-conditioned locomotion** —
`[vx, vy, yaw_rate] → RL policy → 12 leg joint targets`, trained by PPO with no
motion clip used as a reference at all (see §2 below for the full spec).

| Piece | File |
|---|---|
| Env cfg, reward weights, the 11-stage curriculum | `rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_env_cfg.py` |
| Categorical velocity-command term + Bingo-specific reward terms | `rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_mdp.py` |
| PPO hyperparameters | `rl/bingo_rl/bingo_rl/locomotion/agents/rsl_rl_ppo_cfg.py` |
| Gym task registration (`Bingo-Velocity-Flat-v4-{S1..S11}-v0`, stance-test, play) | `rl/bingo_rl/bingo_rl/locomotion/__init__.py` |
| CLI entry points | `rl/tools/train_velocity.py`, `play_velocity.py`, `eval_velocity.py` |
| ONNX export for external (non-Isaac-Lab) use | `rl/tools/export_onnx.py` (exists, deprioritized this session) |
| Checkpoints | `logs/rsl_rl/bingo_velocity_v4/Bingo-Velocity-Flat-v4-S{n}-v0/` |
| Eval reports / videos | `docs/locomotion/EVAL_S{n}.txt`, `eval_s{n}.json`, `eval_video/` |
| Design spec, with evidence tags | `docs/locomotion/TASK1_DESIGN.md` |

It **reuses, read-only**, the validated Stage 4 physics — `bingo_v4.py`'s actuator
gains/effort-limits and the v4 USD (`rl/v4_usd/bingo_v4.usd`) — because it's the
same physical robot. Everything else (reward, training loop, task registration,
checkpoints, eval) is new and lives entirely under `locomotion/` and the
`-v4-`-suffixed task namespace, so it can never collide with the pipeline's own
tasks.

Also present in `rl/bingo_rl/bingo_rl/` but **not** Task 1 and not this session's
concern: `track/`, `track_expr/`, `track_v4/` (DeepMimic-style animation trackers —
Stage 5's ancestors), `amp/` (dog-mocap AMP gait imitation), `improved_walking_cfg.py`
(an earlier, separate RL walking success that Task 1's env cfg structurally borrows
from but does not train on).

---

## 2. What Task 1 is

Command-conditioned locomotion: `[vx, vy, yaw_rate] → RL policy → 12 leg joint
targets → physics`. Not imitation learning. Success is measured by
`rl/tools/eval_velocity.py`'s 7-gate battery (14 held-command segments — forward,
backward, turning, lateral, diagonal, arc, stand, start/stop), not episode return,
because a policy that stands still and does nothing scores a *perfect*
velocity-tracking reward (zero commanded, zero measured).

## 3. Where training actually stands

An 11-stage curriculum is defined in `bingo_velocity_env_cfg.py`
(`TRAINING_STAGES` / `apply_training_stage`): stages 1–6 shape the command
distribution and rewards, stages 7–11 add domain randomisation one source at a time
(obs noise, friction, mass, actuator variation, pushes). Each stage warm-starts from
the previous stage's checkpoint.

**Stages 1–6 are trained and evaluated.** Gate pass-rate by stage:

| stage | what it adds | checkpoint | gates passed |
|---|---|---|---|
| S1 | forward-only, nominal physics | `model_950.pt`(ish, final) | 2/7 |
| S2 | + stand (exact zero command) | `model_2998.pt` | 2/7 |
| S3 | + wider fwd/bwd velocity range | `model_4497.pt` | 4/7 |
| S4 | + turning | `model_5996.pt` | 5/7 |
| S5 | + reverse / lateral | `model_7495.pt` | 6/7 |
| S6 | + start/stop transitions | `model_8994.pt` | 6/7 |

All checkpoints live under `logs/rsl_rl/bingo_velocity_v4/Bingo-Velocity-Flat-v4-S{n}-v0/`
(untracked, ~121 MB total — not yet decided whether these belong in git).

**From S3 onward, the *only* gate that fails is "torque not saturated."** Every
other gate (drives forward/backward, turns in place, moves laterally, stands still,
no falls) has passed since S5. The torque gate's max saturation has been flat at
16–17% (worst case, the `forward_fast` segment) across S3 → S6 — four stages of
additional training on command distribution did not move it. That plateau, not a
lack of training, is why S7–S11 (randomisation) is not the right next step yet:
piling domain randomisation on top of a policy that already can't clear the torque
gate under nominal physics is very unlikely to fix that gate, and
`HANDOFF.md` explicitly warns against jumping ahead in the curriculum before the
gait is solid.

Also sitting in the working tree, untouched this session (deprioritised, per your
call, in favour of the torque investigation):
- `stage4/ground_fix_conform.py` + `blend_sources/Bingo_Walk_V4_Upright.blend` +
  `motions/bingo_walk_v4_upright*.npz` — a new ground-locking script for a
  conform-rig-authored walk clip. Animation-pipeline-adjacent, not part of the RL
  curriculum.
- `rl/tools/export_onnx.py` — exports a trained checkpoint to ONNX + a DOF-order
  manifest for an external (browser MuJoCo) runtime.
- `scripts/{inspect_blend,playback_physics,replay_motion}.py` — Blender 5.x API
  compat fixes (layered Action API), unrelated to RL.
- The three previously-known S1/S2-era fixes (ear-joint stance, contact-sensor
  `prim_path`, rsl-rl-lib 5.0.1 compat shim) are still uncommitted in
  `bingo_velocity_env_cfg.py` / `{train,play,eval}_velocity.py`.

## 4. Torque-saturation gate — what "saturated" actually measures

From `eval_velocity.py`: `torque_sat_pct` is the fraction of (step × env × leg-joint)
samples where `|applied_torque| ≥ 0.99 × effort_limit`. `effort_limit = 3.0 N·m` is
the leg actuator's hard ceiling, set in `bingo_v4.py` (validated Stage-4 physics,
off-limits to edit per `HANDOFF.md` rule 1 without a proven physics defect). The gate
requires `<5%` on **every one of the 14 battery segments**; `forward_fast`
(0.40 m/s) is worst at 16–17%, `arc`/`move_b`/`forward` next (~9–14%), turning
(~8%), stand/stop near 0%.

Reading `bingo_v4.py`'s own (extensively commented) derivation of the leg PD gains:
`Kp = {SY: 40, SP: 120, knee: 120}` was sized from **static** worst-case droop
(steady-state gravity + ground-reaction load, "≤0.05 rad droop"), explicitly *not*
from a dynamic swing/impact torque budget. At `Kp = 120` (SP/knee), a transient
tracking error of just **0.025 rad** already produces the full 3.0 N·m ceiling — a
very tight margin, given the RL policy issues a brand-new position target every
control step (24 Hz) with nothing in the env forcing that target to move smoothly
step to step.

**Working hypothesis:** the saturation is a transient spike from abrupt per-step
target jumps hitting a very stiff (statically-tuned) PD gain, not a sustained
overload — consistent with saturation tracking locomotion speed/aggressiveness
(`forward_fast` > `forward` > `stand`) and appearing exactly at S3, the stage that
first widens the commanded velocity range to 0.40 m/s (S1/S2's low gate scores were
for a different reason — the policy wasn't obeying commands at all yet, so torque
data from those stages isn't informative about this).

Two reward-side levers could address this **without touching `bingo_v4.py`'s
validated physics**:
1. `action_rate_l2` (currently `-0.005`) penalizes step-to-step action delta.
   `HANDOFF.md` already flags this specific weight as inherited from a 50 Hz
   reference config and never re-measured at this env's actual 24 Hz control rate —
   under-penalizing large per-step jumps at the slower rate is a plausible direct
   cause of target jumps large enough to spike torque.
2. `dof_torques_l2` (currently `-5.0e-5`) is a very light direct torque penalty;
   strengthening it would more directly teach PPO to avoid actions that demand
   near-ceiling torque.

**Not yet done, and the planned next step:** a per-joint breakdown (which of the 12
leg joints actually saturate — SP/knee are the stiff-gain suspects, not SY) and a
step-level correlation between `|Δaction|` and saturation events. `eval_velocity.py`
currently only reports a *pooled* saturation % across all 12 joints, so this needs a
small new diagnostic script before picking between the two levers above — otherwise
any reward-weight change is a guess that costs a training run to find out if it was
wrong.

## 5. Plan (not yet executed)

1. **Diagnose**: read-only script against the S6 checkpoint (`model_8994.pt`),
   `forward_fast` segment (worst offender) — per-joint torque-saturation %, and
   correlation between `|Δaction|` and saturation events, per joint.
2. **Fix**: based on the diagnostic, adjust `action_rate_l2` and/or `dof_torques_l2`
   in `bingo_velocity_env_cfg.py` (in-scope file, not on the Stage 1–5 do-not-touch
   list).
3. **Verify**: continue training from the S6 checkpoint with the new weights for a
   modest number of iterations (a reward-shaping correction, not a new curriculum
   stage — not a full S7 run), then re-run the eval battery.
4. **Decide**: if the torque gate passes (<5% everywhere) with the other 6 gates
   still holding, that becomes the new baseline before S7 (observation noise)
   onward. If the diagnostic instead implicates the PD gains themselves (i.e. this
   turns out to be a physics limitation, not a policy-smoothness one), stop and
   flag that explicitly rather than editing `bingo_v4.py` unilaterally — that file
   is shared with the validated Stage 4/5 animation pipeline.

## 6. Recordings

`docs/locomotion/eval_video/` — the full 14-segment eval battery run against the S6
checkpoint (`model_8994.pt`), rendered headless (no display on this machine, so this
is the only way to actually see what the policy does). Covers stand, forward,
forward-fast, reverse, both turn directions, turn-in-place, both lateral directions,
diagonal, arc, and the start/stop transitions, in that order matching
`docs/locomotion/EVAL_S6.txt`.
