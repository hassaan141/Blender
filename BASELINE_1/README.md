# Bingo Locomotion — BASELINE 1

**Status:** frozen complete Baseline 1 bundle, consolidated 2026-09-21. This is the only active locomotion handoff. Historical experiments are preserved under `../archive/baseline_1_provenance/`.

Baseline 1 is one heading-relative residual policy accepting physical `[vx, yaw_rate]` commands and driving twelve leg joints. It combines the retained forward, backward, and turning reference families, then uses the validated `STAND_SOLVED` pose when commands settle to zero. The neural checkpoint came from unified-command loop 6; the upright stand correction is part of the matched runtime in this directory. **Do not use `policy.pt` with an older environment.**

## Use it

```bash
# No keys: validated upright stand
BASELINE_1/reproduce.sh --seconds 20

# W / forward
BASELINE_1/reproduce.sh --vx 0.20 --yaw 0 --seconds 20

# S / backward
BASELINE_1/reproduce.sh --vx -0.20 --yaw 0 --seconds 20

# A/D pivots; diagonal examples use vx +/-0.15 and yaw +/-0.40
BASELINE_1/reproduce.sh --vx 0 --yaw 0.40 --seconds 20
```

Keyboard mapping implemented by `runtime.keyboard_command`: W/S = ±0.20 m/s, A/D = ±0.40 rad/s, diagonals = ±0.15 m/s with ±0.40 rad/s; opposite keys cancel and release returns `[0,0]`. A live keyboard event frontend is not included.

Primary artifacts:

- `policy.pt` — one 95-input, 12-output policy; SHA-256 `35955f919f41e128fe40a1148c03b5a0d3e64a7dca4e99132feda544c0859cb9`.
- `env.py`, `runtime.py`, `common.py`, `play.py` — required matched runtime.
- `references/` — local forward, backward, turning, pivot, and backward-turn reference banks. Teacher checkpoints are not required at inference.
- `videos/all_commands.mp4` — nine panels: stand, W, S, A, D, W+A, W+D, S+A, S+D.
- `videos/locomotion_to_stand.mp4` and `videos/transitions.mp4` — command transitions.
- `metrics/core_commands.json`, `metrics/edge_commands.json` — complete final and edge-case measurements.
- `history/RUN_INDEX.csv` — normalized 69-row index; the adjacent original ledgers retain every recorded run row and reason.
- `MANIFEST.json` and `VALIDATION.json` — integrity and replay evidence.

## Runtime contract

- Observation: 95 values = heading-relative proprioception 67, phase sine/cosine 2, normalized physical command 2, next feedforward leg target 12, previous filtered residual 12.
- Network: 512/256/128 ELU → 12 raw leg residual actions.
- Control: 24 Hz (120 Hz physics, decimation 5), residual EMA 0.3, residual scale 0.3 rad, command slew alpha 0.1.
- Reference drive reverses for backward motion and uses adjacent yaw geometry for turns. Near-zero turns retain a stepping clock.
- At `[0,0]`, phase is frozen, feedforward becomes the existing validated `STAND_SOLVED`, and incompatible learned play-bow residual authority fades smoothly to zero.
- Nine expressive head/tail/ear joints remain reference-driven.
- External dependencies retained in the repository: Isaac Lab environment, `rl/bingo_rl`, `URDF/bingo_urdf v4_w_ear_joints`, `rl/v4_usd`, and `stage4/out/collision_hulls.npz`. Physics, URDF, Kp/Kd, effort limits, and canonical animation assets were not changed.

## Certified core behavior

Each row is one deterministic 60-second case after one second settling. Survival is observed evidence, not a multi-seed guarantee.

| Keys | Command `[vx,yaw]` | Measured vx | Measured yaw | Accel rad/s² | Max saturation | Slip m/s | Existing numerical gate |
|---|---:|---:|---:|---:|---:|---:|---|
| none | `[0,0]` | 0.0003 | 0.0019 | 0.064 | 0.00% | 0.00003 | PASS; visual upright stand |
| W | `[0.20,0]` | 0.176 | -0.029 | 11.15 | 17.87% | 0.146 | PASS |
| S | `[-0.20,0]` | -0.182 | 0.020 | 8.09 | 6.71% | 0.104 | PASS |
| A | `[0,+0.40]` | -0.009 | +0.233 | 7.94 | 2.68% | 0.059 | FAIL yaw tracking |
| D | `[0,-0.40]` | -0.006 | -0.128 | 7.62 | 2.75% | 0.069 | FAIL yaw tracking |
| W+A | `[0.15,+0.40]` | 0.158 | +0.374 | 9.81 | 13.21% | 0.140 | PASS |
| W+D | `[0.15,-0.40]` | 0.133 | -0.383 | 9.84 | 10.95% | 0.120 | PASS |
| S+A | `[-0.15,+0.40]` | -0.155 | +0.146 | 9.51 | 5.58% | 0.120 | FAIL yaw/vx variation |
| S+D | `[-0.15,-0.40]` | -0.181 | -0.166 | 8.70 | 6.85% | 0.114 | FAIL yaw/vx variation |
| sequence | 5-second changes | — | — | 8.78 | 6.36% | 0.089 | survives; yaw tracking incomplete |

The upright stand has all four feet in contact, body height 0.17947 m, roll RMS 0.078°, pitch RMS 0.435°, zero applied action-rate RMS, zero torque saturation, and 0.00321 rad RMS error from `STAND_SOLVED`. All eight held moving-command metric dictionaries are exactly unchanged from unified loop 6.

## Important limits

“Complete Baseline 1” means the frozen first integrated baseline, not that every command in the accepted interface rectangle is safe. Interface bounds are vx `[-0.20,0.30]` m/s and yaw `[-0.60,0.60]` rad/s, but only the core commands above are the primary tested set.

- Pivots remain anchored and undertrack yaw; rear-left contact is near-continuous.
- Backward turns undertrack yaw and have higher speed variation.
- Maximum forward-left `[0.30,0.60]` falls in edge testing.
- Maximum forward-right survives but overshoots to yaw -0.879 rad/s, acceleration 52.26 rad/s², and 66.24% maximum saturation.
- Maximum straight forward reaches 0.290 m/s but acceleration is 25.88 rad/s² and maximum saturation 51.41%.
- Extreme changing-command testing falls. Consult `metrics/edge_commands.json` before using non-core combinations.
- Slip/contact values are collision-geometry proxies, not force-sensor measurements.
- There is no unified-policy ONNX export.

## Complete experiment history

The full original ledgers are in `history/`; raw checkpoints, videos, logs, source snapshots, decisions, and manifests are in the provenance archive. No failed run was silently promoted.

| Campaign | Runs | Result carried forward |
|---|---:|---|
| Free velocity curriculum | S1–S6 | Reached 6/7 gates; nominal torque saturation remained. Historical, not this checkpoint. |
| Reference walk original loop | 8 + baseline | Selection bug audited; run 6 was the valid refinement start. |
| Reference walk refinement | 8 + start recheck | Run 5 retained; established timing/reference dependency. |
| Reference walk quality | 6 + baseline | Run 5 canonical forward gait: vx 0.193, acceleration 11.00, bob 1.51 mm. |
| Natural-walk edits | 3 | All rejected visually; attempt 3 supplied the later forward teacher/reference. |
| Locomotion 2 exact retarget | 1 principal candidate | Kinematically valid, failed Stage 4 physics. |
| Locomotion 2 AMP | 6 | Run 6 found a genuine larger gait through exploration; separate policy, not the unified checkpoint. |
| July AMP v9 | historical | rev_3-era visual benchmark only. |
| Backward autonomous campaign | 8 | Loop 7 retained teacher; vx -0.183, but vx std/yaw-rate goals remained unmet. |
| Turning autonomous campaign | 8 | Loop 5 retained teacher; heading-local yaw worked, but left drag/saturation gates remained. |
| Unified command campaign | 8 | Loop 6 retained network; loop 7/8 improved yaw but caused unacceptable stand lean. |
| Stand-only repair | 2 of 6 | Loop 1 exposed residual conflict; loop 2 fixed runtime stand and passed visual review. |

### Key progression

1. Forward gait was refined through fixed-baseline acceleration gates, cadence preservation, torque/saturation diagnosis, and visual quality review.
2. Small natural-walk paw edits improved local clearance but did not visually replace the canonical gait.
3. Backward used signed phase plus corrected contact and stance geometry; loop 7 became the frozen backward teacher without claiming full naturalness.
4. Turning introduced heading-relative observations and signed yaw reference banks; loop 5 became the turning teacher.
5. Real simulations from the three incompatible teachers were converted into the shared 95-value interface, distilled into a fresh policy, and PPO fine-tuned with teacher replay.
6. Unified loop 6 best balanced locomotion and survival. Stand repair then replaced the obsolete inherited play-bow target with `STAND_SOLVED` and removed only zero-command residual authority. No retraining was needed, avoiding gait forgetting.

## Provenance and maintenance

Historical records were moved—not rewritten—to `../archive/baseline_1_provenance/`. `PROVENANCE_MAP.json` records every former active path. Historical manifests remain evidence of their original sealed contents; protection records containing old absolute paths are historical and are superseded for current integrity checks by this bundle’s manifest.

Treat this directory as immutable Baseline 1. New work must use a new directory and compare against `metrics/core_commands.json`, the videos, and the limitations above. Do not overwrite this checkpoint, references, videos, metrics, validation, or manifest.
