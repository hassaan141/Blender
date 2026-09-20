# Autonomous turning campaign

**Finished: eight loops, best retained candidate loop_0005/A. Full natural-gait gates remain unmet.** See [results](TURNING_RESULTS.md), [leaderboard](leaderboard.csv), [runtime bundle](BEST.json), and [forward/left/right video](final_comparison.mp4). No loop 9 was started.

Task: learn left/right heading changes with the existing reference + residual PPO controller, vx .15–.20 m/s and yaw command −.6 to +.6 rad/s. Not sidestepping. At most eight total loops; stop early only for a convincing qualifying gait or repeated evidence of a fundamental block.

Backward baseline is frozen at `../loops/loop_0007/A/`; its pointer and hashes are in `../FROZEN_BACKWARD_BASELINE.json`. Forward baseline remains `../../walk_ref/quality_runs/run_05/`. Historical forward, backward and lateral artifacts and native controller/physics source are protected by before/after hashes. No AMP, physics, URDF, gains or effort-limit changes.

The isolated turning extension adds one yaw observation (70→71 dimensions). Checkpoint migration pads the first policy/value weight matrices with a zero column and preserves existing weights, optimizer moments and normalization channels. A task-space reference bank uses local stance velocity `(vx - yaw*y, yaw*x)` to vary inside/outside stride and foot-path direction, preserving native timing and height. New yaw tracking is the one added task reward. Later loops change only one targeted setting per hypothesis.

Each candidate gets short left/right physics previews before training, then 300 PPO iterations / 512 environments / seed 42, warm-started from the retained candidate (first from the forward canonical checkpoint). Main evaluations: 60s each at yaw ±.4. Speed input is recorded per candidate; loop3 calibrated it from .175 to .215 to obtain actual travel near .15m/s. Additional probes: 30s each at yaw ±.6 and yaw zero. Straight and turning metrics are distinct. This does not claim a combined forward/backward/turning policy; the existing backward policy remains available unchanged.

Success requires full observed survival, no joint-limit violations, joint acceleration <=13.20023 rad/s², mean slip <=.24 m/s, no joint saturation >10%, measured vx .12–.23, vx std <=.075, yaw mean error <=.15 rad/s (<=.08 at zero), yaw std <=.3, each foot contact duty <=.85 and p95 clearance >=5mm, plus visual approval by the active agent. The explicit foot gates were added after loop3 exposed inherited rear-left dragging; earlier provisional KEEP decisions do not imply these passed. The core also imposes vx std <.06 on both main turns for full goal qualification. Provisional KEEP means overall improvement, not final success. Contact/slip diagnostics use collision geometry, not force sensors.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/core.py \
  --run-continuous --max-loops 8 --resume \
  --config docs/experiment_loop/turning/config.json
```

The active agent reads reports, metrics and video montages, writes one hypothesis through `agent_inbox`, reviews previews and post-training videos, then selects KEEP/REJECT. The file bridge requires that active agent; it does not generate random sweeps. No user approval is requested between loops. Each completed loop is sealed with a manifest; failed loops cannot replace the champion.

## Removed sidestepping experiment

The unused Claude sidestepping campaign was removed at the user's request. Turning retained the useful lazy Isaac Lab registration pattern in its own worker; it has no runtime dependency on the removed code. The diagnostic lesson remains: setting native `cmd_vx=0` freezes phase advance and blends the reference toward standing. Historical sealed snapshots were preserved for exact provenance. See [cleanup record](../CLEANUP.md).

Reproduction: use [the recorded command](TURNING_RESULTS.md) with video enabled to match the evaluation initialization. The first10s were verified exactly. `--no-video` skips a render warm-up reset and yields different individual trajectories.
