# Bingo AMP — Naturalness Improvement Plan

**Status:** v9 produces a clean diagonal dog trot (fl·br +0.98, fr·bl +1.00, fl·fr −0.98, tilt 0.008, base 0.192, vx 0.568). It is a solid, shippable result. This document is about closing the remaining "clean but slightly mechanical" gap — **and it is entirely a reference-motion problem, not a reward problem.**

---

## Core diagnosis

**The AMP reward (`_get_rewards` in `bingo_amp_env.py`) is good engineering and is NOT the bottleneck.** The v9 trot-phase + height shaping works as intended.

**The bottleneck is the retargeted reference motion (`bingo_trot.npz`).** AMP is a distribution-matching method: the discriminator can only make Bingo as natural as the reference it is shown. The current reference is a kinematic stick-figure that has discarded most of the dog's real motion content before AMP ever sees it.

### Evidence (measured directly from `bingo_trot.npz`)

The AMP obs is 49-dim. The channels that carry "dog-ness" are the 4 foot positions relative to root (last 12 dims) + root height + root orientation + root/body velocities. In the current reference, those channels contain:

| Channel | Reference value | What a real dog has | Verdict |
|---|---|---|---|
| Foot vertical travel (all 4) | **6–10 mm** peak-to-peak | several cm of paw lift | dead — discriminator learns "natural = no lift" |
| Root height | **0.000 mm** variation (constant 0.19 m) | stride-synced vertical bob | dead constant |
| Root orientation | **frozen identity quaternion**, all 198 frames | subtle pitch/roll each stride | dead constant |
| Body angular velocities | **exactly zero** everywhere | nonzero, stride-synced | dead constant |
| Foot fore-aft (x) travel | ~0.05 m | present | OK (this is why the trot *timing* looks right) |

Three of the channels that define "natural" are dead constants; the one that should show paw lift shows 8 mm. The clean metrics + mechanical look are both explained by this: the policy is faithfully matching a reference that is itself flat and low-lift.

---

## Root cause — two choices in `retarget_dog_to_bingo.py`

**1. The root trajectory is synthesized, not retargeted.**
In the FK section:
```python
root_pos[:, 2] = BINGO_HEIGHT      # constant — dog's vertical bob discarded
root_quat = identity (wxyz 1,0,0,0) # frozen — dog's pitch/roll discarded
body_ang = zeros                    # dog's angular motion discarded
```
The dog clip's real root vertical bob, pitch, and roll are thrown away and replaced with a flat conveyor-belt glide.

**2. The foot "position" excludes knee flexion — the joint that actually lifts the foot.**
In `foot_pos_base()`, the knee angle is hardcoded to `0.0`:
```python
for jt, ang in [(f"{leg}_SY_J", sy), (f"{leg}_SP_J", sp), (f"{leg}_knee", 0.0)]:
```
So the "foot" is really the knee-hinge location (computed from SY + SP only), which barely moves vertically by construction. This is the 8 mm. The knee flexion that would lift the paw is not in the foot signal at all.

Combined: the discriminator sees a flat glide with barely-moving feet, matches it perfectly, and the result is a clean but mechanical trot.

---

## The plan (priority order) — all retargeter-side, env + reward unchanged

### Fix 1 — Restore real foot height to the reference (biggest visual win, ~1 line)
In `foot_pos_base`, stop hardcoding the knee angle to 0. Pass the actual per-frame knee angle so FK computes the true shank position as the knee flexes. Better: compute the **shank tip** (knee-link origin + the −0.071 m offset down the shank, per the URDF), not the knee hinge — that's the true contact point.

**Verification gate:** foot-z peak-to-peak should move from ~0.008 m to roughly **0.03–0.06 m**. That number is the proof the reference improved. Do not retrain until this is confirmed.

### Fix 2 — Carry the dog's root bob + body oscillation into the reference
Replace the flat `root_pos[:,2]=BINGO_HEIGHT` / identity-quaternion / zero-angular-velocity block with the dog clip's actual root height oscillation and root pitch/roll, scaled to Bingo (same `0.16/0.35` factor for vertical amplitude). Write them into `root_pos[:,2]`, `root_quat`, and `body_angular_velocities`. The dog clip already contains this — the retargeter is just discarding it.

### Fix 3 — Feed a distribution of gaits, not one loop
Currently imitates `dog_trot.txt` alone, tiled 6×. Retarget `dog_pace.txt` (and a walk clip if one can be sourced) too, and point `motion_file` at a dataset of all of them. AMP matches a *distribution* — giving it one exact periodic loop is what produces the metronomic quality. Start with trot + pace (pace loaded fine previously; `dog_walk` 404'd from the Peng mirror).

### Fix 4 — Foot-height obs comes for free after Fix 1
The AMP obs already gives the discriminator foot (x,y,z) relative to root — but z is currently flat. Once Fix 1 makes z vary, this channel starts carrying lift information automatically. Optional extra: add explicit per-foot vertical *velocity* to the obs so lift dynamics are visible too.

### Fix 5 — Only then, revisit discriminator capacity
Discriminator is currently [1024, 512], same as the policy — plenty for a richer reference. Do NOT touch it until the reference carries real signal; a bigger discriminator on a flat reference just learns the flat reference faster.

---

## The honest ceiling

Bingo has **no ankle / no foot joint** — the knee is the last actuated DOF per leg (3 DOF/leg: SY, SP, knee). The shank (knee child link) is the terminal segment; its tip is the contact point. No amount of AMP or reward tuning can produce a dog's **paw roll through contact** — that DOF does not exist in hardware.

What these fixes recover is everything *above* the ankle: real knee-driven foot lift, stride-synced body bob and pitch, and gait variety. That is the majority of the remaining naturalness gap, and it is all currently being discarded before AMP sees it.

**Estimate:** Fixes 1–3 get most of the way from "clean but mechanical trot" to "reads as a real small dog." A few hours of retargeter work + retraining, no hardware change. Keep v9 as the baseline and compare honestly; if a change doesn't visibly beat v9, don't keep it.

---

## Concrete tasks to hand off

1. **Fix 1:** In `retarget_dog_to_bingo.py`, stop hardcoding the knee angle to 0 in `foot_pos_base` — pass the real per-frame knee angle and compute the shank-tip position including the −0.071 m shank offset, so the reference feet show true vertical lift. Print the new foot-z peak-to-peak (expect 0.03–0.06 m, up from ~0.008 m).

2. **Fix 2:** Carry the dog clip's root height oscillation and root pitch/roll into the reference instead of the flat constant-height identity-quaternion root; scale vertical amplitude by 0.16/0.35. Also write nonzero `body_angular_velocities`.

3. **Fix 3:** Retarget `dog_pace.txt` as well and point `motion_file` at a trot+pace dataset.

**Do NOT change** `bingo_amp_env.py` reward logic or `skrl_amp_cfg.yaml` scales for the first pass — the v9 reward is good; the reference is what needs fixing. Verify Fix 1's foot-z number before spending GPU time on a retrain.