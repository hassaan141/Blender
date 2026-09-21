# Walking Optimization Goal

Baseline:
docs/walk_ref/loop_runs/baseline/

Target:
- survive 60 s
- mean vx: 0.20–0.30 m/s
- vx std < 0.06 m/s
- max vx < 0.40 m/s
- reduce joint acceleration RMS
- reduce residual action-rate RMS
- preserve the current reference-guided gait visually

Priority:
1. survival
2. consistent slow speed
3. remove speed bursts
4. reduce shaking
5. reduce torque saturation

Allowed changes:
- command/speed curriculum
- phase-rate mapping
- reference timing
- residual scale
- action smoothing
- action-rate penalty
- joint-acceleration penalty
- torque penalty
- reference tracking weights

Protected:
- bingo_v4.py
- Kp/Kd
- effort limits
- Stage 1–5 canonical assets

Loop:
1. Measure current best.
2. Identify biggest failing metric.
3. Make ONE focused hypothesis.
4. Change at most 1–2 related parameters.
5. Train a short warm-started candidate.
6. Run eval_walk_loop.py.
7. KEEP only if objectively better.
8. Otherwise REJECT and revert.
9. Repeat.

Run maximum 8 experiments.

Save each experiment to:
docs/walk_ref/loop_runs/run_XX/

Maintain:
docs/walk_ref/loop_runs/leaderboard.csv

Do not print training logs into chat.
Only print one summary line per experiment.

At the end, render the best candidate and stop.