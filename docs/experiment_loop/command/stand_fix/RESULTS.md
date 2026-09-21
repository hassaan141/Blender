# Unified command stand repair — result

**Solved in loop 2; stopped early after two of six allowed attempts.** The retained neural checkpoint is unchanged unified loop 6. The fix is in the matched runtime: it uses the existing validated `STAND_SOLVED` instead of the obsolete v4 default play-bow, and smoothly fades the learned residual to zero only as both smoothed commands settle to zero.

## Deliverables

- Runtime/checkpoint/reference bundle: [`best/`](best/)
- Upright stand: [`best/stand.mp4`](best/stand.mp4)
- All nine stand/W/S/A/D/diagonal panels: [`best/commands.mp4`](best/commands.mp4)
- Moving-to-stand window: [`best/locomotion_to_stand.mp4`](best/locomotion_to_stand.mp4)
- Full metrics: [`best/metrics.json`](best/metrics.json)
- Individual videos: [`best/videos/`](best/videos/)

## Stand result

One deterministic 60-second case, with one second excluded from metrics: survival 100%; height 0.17947 m; roll RMS 0.078°; pitch RMS 0.435°; all four contact duty 100%; acceleration 0.0638 rad/s²; applied action-rate RMS 0; torque saturation 0%; slip 0.000028 m/s; no soft or physical limit violation. Settled legs match `STAND_SOLVED` at 0.00321 rad RMS and 0.00618 rad maximum error, with maximum temporal joint standard deviation 0.000028 rad. Visual review confirms an upright, non-crouched, non-leaning stand throughout.

## Regression result

Forward, backward, left, right, forward-left, forward-right, backward-left and backward-right held-command metric dictionaries exactly match unified loop 6. The 60-second changing-command case survives. Compared with loop 6, transition acceleration improves 10.06→8.78 rad/s² and slip improves 0.0903→0.0886 m/s; maximum transition saturation increases slightly 5.72→6.36% without a visible or survival regression. Existing unified-policy limitations remain unchanged: pivots and backward turns still undertrack yaw.

## Loop history

| Loop | One diagnosed cause / targeted change | Result |
|---|---|---|
| 1 | Wrong inherited stand target → exact `STAND_SOLVED` | REJECT: policy residual conflicts with corrected target and causes falls. Four-arm probe proves zero residual holds the target. |
| 2 | Learned obsolete-stand residual → command-activity fade to zero | KEEP: visual stand passes, transitions survive, all held moving cases exactly preserved. |

No fine-tuning was necessary; retaining the loop-6 network avoids gait forgetting. The checkpoint must be used with the bundled stand-fixed environment. No physics, URDF, actuator, Kp/Kd, effort-limit, canonical baseline, teacher, or historical loop file was modified.
