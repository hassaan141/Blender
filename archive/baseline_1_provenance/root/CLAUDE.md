# Bingo Project Rules

## Current objective
Build a smooth, slow, elegant reference-guided RL walking controller for Bingo and export it to ONNX.

Target walking behavior:
- forward speed around 0.25 m/s
- consistent speed, no bursts
- smooth legs, minimal jitter
- stable body
- sustained walking without falling
- preserve the recognizable Stage 4 walking style

## Current best approach
Reference-guided residual RL is the main approach.

q_target = q_ref(phase) + residual_scale * action

The current reference-guided RL result is visually better than:
- PD-only
- old S6 from-scratch locomotion

Do not optimize toward PD-only or S6.

## Protected files
Do not change without explicit approval:
- bingo_v4.py
- actuator Kp/Kd
- effort limits
- canonical Stage 1-5 assets
- canonical Stage 4 motion files

## Working style
- Measure before changing things.
- Change 1-2 related variables per experiment.
- Evaluate every candidate with the same deterministic test.
- Keep a candidate only if it improves the current best.
- Do not repeatedly train a configuration that has plateaued.
- Save large logs/results to files instead of printing them in chat.
- Do not stop after every experiment to ask permission.
- Never call a run successful only because reward increased. Visual gait quality and stability matter.

## Experiment outputs
Store optimization runs under:

docs/walk_ref/loop_runs/run_XX/

Each run should contain:
- hypothesis
- config
- checkpoint
- metrics.json
- evaluation video
- KEEP or REJECT result

Maintain:
docs/walk_ref/loop_runs/leaderboard.csv