# Sim-to-sim attempt decision

Python MuJoCo decision: **KEEP as a research candidate**.

Browser deployment decision: **REJECT**. The actual browser MuJoCo/WASM runtime
survived only stand with its normal Neutral expressive-joint animation and a
stand-to-command transition. The candidate therefore does not meet the browser
survival gate and must not replace the installed policy.

All-nine survival: True. Stand stable: True. Mean normalized command-tracking error (baseline/candidate): 0.3472 / 0.3272. Candidate survival is mandatory; a tracking gain cannot offset a fall.

See `comparison.csv` for per-command survival, speed, yaw, slip, acceleration, action-rate, saturation and joint-limit metrics.

See [browser probe](../browser_probe.md) for the controlled browser results and
the Python/browser runtime differences that explain much of the gap.
