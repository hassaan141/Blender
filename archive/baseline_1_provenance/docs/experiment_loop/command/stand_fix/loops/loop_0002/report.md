# Stand-fix loop 2 — KEEP / STAND SOLVED

Inherited loop 1's required correction to source the stand target from the existing `STAND_SOLVED`. Changed one additional cause: smoothly fade the applied filtered residual to zero only as both smoothed physical commands settle to zero. No training or checkpoint change.

Mandatory visual review passes: `stand_frames.jpg` shows a level upright pose throughout; `locomotion_to_stand_frames.jpg` shows locomotion settling smoothly into the same stand without collapse, crouch, or lean. Numerical stand result over 59 measured seconds: height 0.17947 m, roll 0.078°, pitch 0.435°, four contacts at 100%, acceleration 0.0638 rad/s², applied action-rate RMS 0, torque saturation 0%, and no limit violation. Settled leg joints match `STAND_SOLVED` at 0.00321 rad RMS / 0.00618 rad maximum error.

All eight held nonzero keyboard cases are exact metric matches to retained unified loop 6 because their activity scale is one. The 60-second transition sequence survives; acceleration improves 10.06→8.78 rad/s² and slip 0.0903→0.0886 m/s. Transition-only maximum saturation rises slightly 5.72→6.36%, below held-command behavior limits and without visible or survival regression.

Stop condition met after two of six allowed attempts. Remaining four attempts are intentionally unused.
