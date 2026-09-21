# Unified command policy campaign

**STOPPED: eight attempts completed.** See [RESULTS.md](RESULTS.md) for the retained experimental loop6 bundle, videos, metrics and limitations. The full natural command gait goal was not met. Do not start another loop without a new instruction.

Goal: one residual policy for `[vx (m/s), yaw (rad/s)]`, ranges
`[-0.20, 0.30]` and `[-0.60, 0.60]`. No ONNX export. All work is experimental
until the full command battery and videos pass. Teachers remain read-only.

## Interface audit

| Teacher | Observation | Reference runtime |
|---|---|---|
| Natural attempt 03 | 70, world-frame | reference_02b, signed phase, EMA .3 |
| Backward loop 7/A | 70, world-frame | retained backward geometry, negative phase |
| Turning loop 5/A | 71, heading-local, yaw appended | BEST.json reference + yaw bank |

All teachers output twelve residual leg actions, with .3 rad residual scale
and EMA alpha .3. Student input is 95 values: heading-relative proprioception
67, phase sine/cosine 2, normalized physical vx/yaw commands 2, next feedforward
leg target 12, previous filtered residual 12. A fresh network is distilled;
incompatible teacher checkpoints are never loaded as the student.

Runtime blends teacher references continuously by command, reverses phase for
backward motion, and reverses yaw-bank geometry when phase reverses. Near-zero
turns maintain a stepping clock; this is an unvalidated extrapolation, not a
claimed teacher capability. Command slew uses alpha .1 at 24 Hz. Exact teacher
cadence is calibrated to the physical command endpoints (+.20 forward maps to
+.25 reference drive, +.15 moving turn to +.215, backward -.20 stays -.20).

Collection runs each original teacher controller in actual Isaac Lab simulation.
The shared observation includes its actual next reference target and filter
state. No dog angles, AMP training, altered gains, effort limits, or physics.
The existing native Isaac Lab/skrl PPO trainer fine-tunes the distilled network.
Curriculum expands every 1,920 control steps from straight/stand to near-zero
turns, forward turns, backward turns, then mixed random commands.

## Evaluation and bounded execution

`rl/bingo_rl/experiment_loop/command/run.py` enforces eight maximum loop IDs,
per-loop source/config/commands/logs/checkpoints, protected hashing before and
after subprocesses, and a 90-minute subprocess timeout. Re-running an unfinished
loop resumes from completed stages. Completed decisions cannot be overwritten.
The active agent reviews each result and chooses one targeted next hypothesis.
A failed candidate never replaces the champion; no promotion from reward alone.

Each battery simulates all nine held commands plus transitions for 60 seconds.
Each case currently has one deterministic trial; survival is not a broad
reliability estimate. Close-up videos cover 12 seconds at 12 fps. Metrics include
world-contact paw slip, torque saturation, acceleration, body motion, joint
limits and duty factors. Raw per-step arrays are retained for diagnoses.

Protected inventory: [PROTECTED.json](PROTECTED.json).

Rendering note: this machine's CUDA order differs from `nvidia-smi` order.
Training uses CUDA_VISIBLE_DEVICES=0 (RTX 4090), while validated rendering uses
CUDA_VISIBLE_DEVICES=1 (RTX 2080 Ti), as the historical evaluator did. The 4090
capture path produced black images here. Video probes are not gait candidates.
Only one rendering process runs at a time. The runner checks required output
artifacts because Isaac can return exit code zero after an internal failure.

PPO includes one 2,048-transition teacher replay update after each PPO update,
using a separate Adam optimizer at 1e-4. This is supervised distillation replay,
not AMP. Normalization uses the current student observation preprocessor.
