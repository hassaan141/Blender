# Bingo MuJoCo sim-to-sim adaptation

This work adapts the frozen `BASELINE_1/policy.pt` actor to the browser simulator's
existing MuJoCo model. The Python wrapper loads the browser MJCF directly from
`bingo-simulator/app/public/robot/bingo_scene.xml` and resolves joints and actuators
by name. It does not modify the browser runtime, MJCF, checkpoint, or robot assets.

## Runtime contract

- Observation: 95 raw fields: 67 heading-relative proprioception, phase sin/cos,
  normalized `[vx,yaw]`, next feedforward leg target, previous filtered residual.
- Policy: BASELINE_1 actor and its saved observation mean/variance.
- Action: 12 leg residuals, clipped to `[-1,1]`, scaled by 0.3 rad.
- Runtime: 0.3 residual EMA, 0.1 command slew, 24 Hz control, five 120 Hz MuJoCo
  physics steps, frozen Stage 4/5 reference banks, and `STAND_SOLVED` at zero command.
- MuJoCo asset: the browser's `bingo_scene.xml` and its adjacent collision meshes.

The baseline checkpoint SHA-256 recorded before MuJoCo evaluation was
`35955f919f41e128fe40a1148c03b5a0d3e64a7dca4e99132feda544c0859cb9`.

## Commands

Python environment: `/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python` with the
official `mujoco` Python bindings. All long jobs use the repository's one-shot
completion runner; do not poll their output files.

Baseline evaluation:

```bash
MUJOCO_GL=egl tools/experiment_loop/run_and_wake.sh \
  --thread "$CODEX_THREAD_ID" --timeout 1200 \
  --result-dir tools/experiment_loop/results/sim2sim_baseline_mujoco_20260923 \
  -- /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python \
  tools/experiment_loop/sim2sim/evaluate.py \
  --checkpoint BASELINE_1/policy.pt \
  --out tools/experiment_loop/results/sim2sim_baseline_mujoco_20260923/evaluation \
  --seconds 8 --settle 1 --video --device cuda:0
```

Training and candidate evaluation are separate bounded jobs. `train_ppo.py` starts
from BASELINE_1 actor weights, creates a fresh value network, and regularizes policy
updates toward the frozen actor. Candidate metrics and nine command videos are
written by `evaluate.py`; the original baseline remains read-only.

## Files

- `tools/experiment_loop/sim2sim/mujoco_env.py`: model-backed simulator adapter.
- `tools/experiment_loop/sim2sim/train_ppo.py`: bounded PPO adaptation.
- `tools/experiment_loop/sim2sim/evaluate.py`: nine-command metrics and videos.
- `tools/experiment_loop/results/`: immutable job outputs and completion receipts.

At this stage, no adapted checkpoint is accepted until every required command passes
the survival gate and its videos are reviewed against the baseline.

Final three-attempt results: [leaderboard and review](summary.md). The
[browser probe](browser_probe.md) rejected attempt 03 for deployment. The installed
browser policy directory remains unchanged.
