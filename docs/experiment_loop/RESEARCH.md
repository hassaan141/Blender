# Research and adaptation — 2026-09-18

Research completed before implementation. No external implementation was copied.

## Akira Sasaki / @gclue_akira

[Profile](https://x.com/gclue_akira) and the [post indexed as the quadruped RL case](https://x.com/gclue_akira/status/2098300921658868185) could not be fetched directly from X in this session. The exact viral post previously sent in another conversation is not identifiable with certainty from the supplied text. The secondary index links this status for both trajectory-control and RL descriptions, so those descriptions should not be conflated.

The requested previous-result → diagnosis → change → Isaac Lab training → readable video → report → human-feedback workflow is implemented as the user's requirement. Direct X verification of the loop-boundary wording and the later A/B announcement remains unavailable; it is not presented here as independently verified primary evidence. Repository evidence below independently establishes finite comparisons and explicit rejection.

## FaBoAI repository

`FaBoAI/Singula` returns 404. The FaBoAI organization's public repository is [SingularityDog](https://github.com/FaBoAI/SingularityDog), consistent with the truncated `Singula…` link. Inspected revision: **4bad4011f6c75f2528bd6e51e849b0331b1db0df**.

Read its README, toolkit, current/next-loop descriptions, L25/L31/L32 evaluations, operations/training contracts, `bounded_job.py`, and `artifact_manifest.py`. Useful patterns:

- Unique experiment folders and explicit finite argv requests.
- Shared advisory locks and timeouts; dispatch is not completion.
- Exit status plus expected output checks and hashes.
- Independent alternatives from the same parent, with equal seed/budget.
- Separate historical results, current status, videos and failed decisions.
- Recovery records that preserve partial work rather than silently retrying.

[L25](https://github.com/FaBoAI/SingularityDog/blob/4bad4011f6c75f2528bd6e51e849b0331b1db0df/docs/l25-evaluation.md) documents independent A/B/C conditions; [L32](https://github.com/FaBoAI/SingularityDog/blob/4bad4011f6c75f2528bd6e51e849b0331b1db0df/docs/l32-evaluation.md) distinguishes fixed corrections from learning and rejects unsuccessful outcomes. The [toolkit](https://github.com/FaBoAI/SingularityDog/blob/4bad4011f6c75f2528bd6e51e849b0331b1db0df/docs/toolkit.md) explicitly says the public package cannot reproduce its private historical training stack. We reuse workflow ideas, not its robot physics or success claims.

## Secondary case index

[Awesome Astra Embodied AI, quadruped RL case](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI#-astra-builds-rl-training-environments-and-training) records the reported five-day, 25-loop, nine-motion history. This is context, not reproducibility evidence or proof that the workflow guarantees natural walking.

## Isaac Lab primary references

Read the [technical paper](https://arxiv.org/html/2511.04831v1) and [official RL scripts documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html). Isaac Lab supplies simulation/task interfaces and RL-library wrappers; skrl supports task selection, explicit checkpoints and headless video evaluation. Therefore this runner orchestrates Bingo's existing skrl training/evaluation entry points. It does not implement PPO, replace the environment, switch simulator backends or modify actuators. Existing local scripts, rather than potentially newer documentation CLI syntax, determine the executable commands.

## Bingo command audit

Read `docs/natural_walk/run_train.py`, `run_eval.py`, `make_comparison.py`, the actual `natural_walk/train.py`, `evaluate.py`, registration/config classes and existing command JSONs.

The old campaign wrapper enforces its completed three-attempt budget and hardcodes output locations. It remains unchanged. New hooks call the underlying scripts through a small runtime configuration wrapper, writing everything to a fresh experiment directory. The wrapper applies identical selected settings at `gym.make` in training and evaluation, preserving the original script's `__file__`-based asset discovery. Original training YAML is saved before this runtime override; `train.effective_settings.json` is the authoritative selected-setting overlay. Full native params, overlay, commands, reference, input hashes and source snapshots are retained together.

The default seed is attempt 03, with reference 02b and BL contact correction enabled. It remains a rejected experimental seed; Locomotion 1 stays champion until a new candidate is approved. Default budgets match the previous local stack: 512 environments, seed 42, 100 iterations × 24 rollout steps, 60-second evaluation, one-second settling, held command 0.25 m/s, GPU 1.
