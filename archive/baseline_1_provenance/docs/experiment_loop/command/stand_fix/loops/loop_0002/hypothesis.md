# Loop 2 hypothesis

Cause: the retained policy learned nonzero zero-command residuals against the obsolete play-bow target. The corrected `STAND_SOLVED` target is physically stable only when those incompatible residuals are absent.

Targeted change: continuously scale the applied filtered residual by command activity `max(|vx|/0.15, |yaw|/0.4)`, clamped to `[0,1]`. It reaches exactly zero only as both smoothed commands settle to zero; all canonical W/S/A/D and diagonal commands retain full residual authority. No training, reward, reference, or physics change.
