# Experiment handoff

Three completed campaigns contain 22 experiments: original loop (8), refinement (8), and gait quality (6). [leaderboard.csv](leaderboard.csv) is the single active table; decisions retain each campaign's own protocol and are not a cross-campaign ranking.

The original loop incorrectly kept Run 05 despite a 37.8% acceleration increase. The subsequent audit fixed rejection at >120% of the original 12.236963 rad/s² baseline, rather than using a moving parent. Original Run 06 was the audited starting point for refinement. See the archived [audit](../refinement_runs/audit_existing.json) and [selection rules](../loop_runs/selection_rules.py).

Refinement Run 05 established the slow reference timing. The later quality campaign diagnosed front-left reference/contact/tracking asymmetry before reward changes; its accepted chain was baseline → Run 02 → Run 04 → **Run 05**. Quality Run 06 was rejected. [Diagnosis](../quality_runs/DIAGNOSIS.md) and [protocol](../quality_runs/PROTOCOL.md) preserve the measured reasoning and gates.

The current winner reduced acceleration from 12.5365 to 11.0002 rad/s² and bobbing from 2.3599 to 1.5098 mm, with 100% survival and unchanged cadence. Action-rate RMS increased from 0.017374 to 0.018130; it is not claimed as an improvement. The canonical checkpoint/video remain at their original paths; see [CURRENT_BEST](../CURRENT_BEST.md).

Historical rejected-run records are retained only in the archive as exact configuration/diagnostic evidence for the selection audit and gait tradeoffs. Obsolete intermediate training checkpoints and generated Python caches were deleted. Final/best historical checkpoints and all explicitly referenced checkpoint paths remain available; original commands resolve through compatibility links. No new experiment or training was performed during cleanup.
