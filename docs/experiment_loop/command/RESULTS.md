# Unified command policy: bounded campaign result

**Stopped after eight training attempts. Full natural command locomotion is NOT achieved.**
Loop 6 is retained as the best balanced experimental student. Loops 7 and 8 turn
more strongly, but introduce an unacceptable leaning stand; neither is promoted.

[One policy](best/policy.pt) · [Runtime](best/runtime.py) · [Reference bundle](best/references/) · [Reproduce](best/reproduce.sh)

[All keyboard commands](loops/loop_0006/commands.mp4) · [Teacher comparison](loops/loop_0006/teacher_comparison.mp4) · [Previous champion comparison](loops/loop_0006/comparison.mp4) · [Transitions](loops/loop_0006/evaluation/transitions.mp4)

## Retained metrics

Each case is one deterministic 60-second trial. All ten survived; this is not a
population reliability estimate. Five cases pass every numerical gate: stand,
forward, backward, forward-left and forward-right. Pivots remain anchored and
undertrack yaw; backward turns and transitions undertrack yaw. Video does not
pass the full natural-gait goal. No physical joint-limit violations occurred.

| Keys | Command vx, yaw | Measured vx | Measured yaw | Accel rad/s² | Max saturation % | Slip m/s | All gates | Video |
|---|---|---:|---:|---:|---:|---:|---|---|
| none | [0.0, 0.0] | -0.001 | -0.006 | 0.14 | 0.00 | 0.000 | pass | [view](loops/loop_0006/evaluation/stand.mp4) |
| W | [0.2, 0.0] | 0.176 | -0.029 | 11.15 | 17.87 | 0.146 | pass | [view](loops/loop_0006/evaluation/forward.mp4) |
| S | [-0.2, 0.0] | -0.182 | 0.020 | 8.09 | 6.71 | 0.104 | pass | [view](loops/loop_0006/evaluation/backward.mp4) |
| A | [0.0, 0.4] | -0.009 | 0.233 | 7.94 | 2.68 | 0.059 | FAIL | [view](loops/loop_0006/evaluation/left.mp4) |
| D | [0.0, -0.4] | -0.006 | -0.128 | 7.62 | 2.75 | 0.069 | FAIL | [view](loops/loop_0006/evaluation/right.mp4) |
| W+A | [0.15, 0.4] | 0.158 | 0.374 | 9.81 | 13.21 | 0.140 | pass | [view](loops/loop_0006/evaluation/forward_left.mp4) |
| W+D | [0.15, -0.4] | 0.133 | -0.383 | 9.84 | 10.95 | 0.120 | pass | [view](loops/loop_0006/evaluation/forward_right.mp4) |
| S+A | [-0.15, 0.4] | -0.155 | 0.146 | 9.51 | 5.58 | 0.120 | FAIL | [view](loops/loop_0006/evaluation/backward_left.mp4) |
| S+D | [-0.15, -0.4] | -0.181 | -0.166 | 8.70 | 6.85 | 0.114 | FAIL | [view](loops/loop_0006/evaluation/backward_right.mp4) |
| sequence | 5-second command transitions | 0.002 | 0.004 | 10.06 | 5.72 | 0.090 | FAIL | [view](loops/loop_0006/evaluation/transitions.mp4) |

Full body motion, cadence/contact onsets, duty factors, per-joint saturation and
tracking errors: [metrics.json](best/metrics.json). [CSV table](metrics.csv).
Raw joint/root trajectories: [retained evaluation](loops/loop_0006/evaluation/).
Slip uses world paw-tip horizontal speed while the collision hull is in contact;
it should not be compared directly with older reports using other slip formulas.

## Eight-loop history

| Loop | Targeted change | Decision | Result |
|---|---|---|---|
| 1 | Teacher distillation + mixed PPO | REJECT | Survives, but pivots travel forward and backward turns stall. |
| 2 | Consistent pivot/backward task-space banks | KEEP experimental | Removes pivot translation and restores backward turning. |
| 3 | Yaw tracking weight .3 → .6 | KEEP experimental | Small yaw/rocking improvements; pivots still deficient. |
| 4 | Rear-left pivot swing clearance +8 mm | REJECT | Foot remains planted; stand lean and 82% saturation. |
| 5 | Explicit balanced core-command sampling | REJECT | Pivots improve, but stand and transitions fall. |
| 6 | Replace failed stand demonstrations with surviving champion rollouts | **KEEP experimental** | Restores stand/transitions; retains moving gaits and modest pivot gains. |
| 7 | Pivot-only yaw geometry gain ×1.5 | REJECT | Better yaw, but stand lean18.79° and forward saturation18.57%. |
| 8 | Match stand tracking rewards to applied stand blend | REJECT | Pivot yaw+.294/−.272, but stand still leans17.40°. |

[Machine-readable history](LOOP_INDEX.csv). Every loop contains hypothesis,
source/config, logs, checkpoint, 60-second metrics, video, comparisons, decision,
and a sealed hash manifest. Failed candidates never replaced champion. Rejected
loops 5 and 7 were used only as explicitly unpromoted recovery training parents.

## What worked and what remains

Actual rollouts from the three incompatible teachers were converted into a
shared observation interface, then distilled into a fresh network. Reference
geometry changes, signed backward phase and clean stand replay preserved most
teacher behavior. Runtime loads one student checkpoint, not three teacher policies.

The native stand feedforward and inherited walking tracking rewards disagree by
0.571 rad RMS. Correcting that mismatch in the last attempt helped command
tracking but did not repair the stand within this budget. The rear-left paw is
still in contact about97% during pivots. Backward contact labels also differ from
the forward labels used by inherited rewards. These are unresolved, documented
issues—not evidence that simply running more PPO or raising velocity reward will
solve the merge. The retained loop 6 intentionally excludes rejected changes.

The stand dataset audit found only75/128 randomized phase starts survived with
loop3. Replay repair uses only complete surviving trajectories; broad randomized
reliability is not established for the retained policy.

## Interface and reproduction

`[vx_cmd, yaw_cmd]` in m/s and rad/s. Interface bounds: vx[-.20,.30], yaw[-.60,.60].
The whole rectangle is not certified. W/S use±.20, A/D use±.40yaw; diagonal commands
use±.15vx. Opposite keys cancel; release commands zero. `keyboard_command()` supplies
this mapping; a live keyboard event frontend is not included.

One heading-relative95-value observation → one512/256/128 ELU network → twelve leg
residual actions. The matched runtime applies EMA.3, residual scale.3, phase and
reference blending, and command smoothing.1 at24Hz. The nine expressive joints
remain reference-driven. The repository's existing Isaac Lab/validated robot
runtime is required; all inference reference NPZs are copied into the bundle.

From repository root:

```bash
docs/experiment_loop/command/best/reproduce.sh --vx 0.20 --yaw 0 --seconds 20
```

Use `CommandController.act(env, [vx, yaw])` with the bundled CommandEnv for changing
commands. Teacher weights are not required. Training commands, configuration and
source snapshots remain in each sealed loop; teacher replay datasets and their
provenance remain under teachers/. No ONNX export or physics changes were made.

Loop1 metric erratum: the original hard-limit field measured90% soft training
margins. Corrected metrics live in corrections/; sealed original results remain
unchanged. There was no physical hard-limit violation.

## Command-range validation

The packaged policy was additionally evaluated for60s without training. The full
command rectangle is **not supported reliably**: maximum forward-left and the
extreme transition sequence fall. Surviving cases can still substantially
undertrack commands. These tests do not promote another candidate.

| Case | Survival | vx | yaw | Accel rad/s² | Max saturation % |
|---|---|---:|---:|---:|---:|
| max_forward | pass | 0.290 | -0.057 | 25.88 | 51.41 |
| max_backward | pass | -0.182 | 0.019 | 8.12 | 6.00 |
| max_left | pass | -0.007 | 0.313 | 7.99 | 2.75 |
| max_right | pass | -0.005 | -0.205 | 8.15 | 4.87 |
| max_forward_left | FAIL | 0.180 | 0.467 | 44.58 | 58.54 |
| max_forward_right | pass | 0.349 | -0.879 | 52.26 | 66.24 |
| max_backward_left | pass | -0.159 | 0.285 | 9.73 | 6.71 |
| max_backward_right | pass | -0.173 | -0.236 | 9.41 | 8.12 |
| creep | pass | 0.002 | 0.011 | 2.83 | 0.42 |
| transitions | FAIL | 0.053 | 0.120 | 23.04 | 21.75 |

[Raw edge metrics](edge_validation/metrics.json). The command means exclude frames
after the first fall; consult survival before interpreting those means.

[Final sanity check](SANITY.json): protected inventory and eight sealed loops unchanged;
packaged checkpoint matches loop6. Bundle replay passed a five-second forward smoke.
