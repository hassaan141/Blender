# Stand-fix loop 1 — REJECT

Changed only `_stand_pose`: obsolete inherited v4 defaults → imported `STAND_SOLVED`. No training; loop-6 checkpoint reused.

Stand fell before the one-second measurement window (`survival=false`, acceleration 53.58 rad/s², pitch RMS 27.78°, max torque saturation 33.26%). Video shows repeated tumble/reset rather than a stand. All eight held moving-command metrics are unchanged from loop 6, so gait behavior was not affected.

Causal probe: zero residual holds `STAND_SOLVED` from either walking reset or exact stand reset for five seconds (final height 0.1794–0.1795 m, tilt 0.38–0.43°, joint RMSE 0.0024–0.0034 rad). Policy-residual arms fail from both resets with up to 58.5° tilt. Next isolated cause: learned zero-command residual conflicts with the corrected feedforward target.
