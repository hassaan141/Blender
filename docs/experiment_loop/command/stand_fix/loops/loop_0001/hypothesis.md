# Loop 1 hypothesis

Cause: the unified command environment calls the obsolete `BINGO_V4_CFG` default leg angles its stand pose. At zero command it therefore correctly freezes phase and settles, but settles into a play-bow rather than `STAND_SOLVED`.

Targeted change: source `_stand_pose` directly from the existing `STAND_SOLVED` constant. No residual gating, reward change, training, reference change, or physics change. Reuse the exact retained loop-6 checkpoint.
