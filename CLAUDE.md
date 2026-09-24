# Working rules (low token usage)

Full rationale: `LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md`. Project state: `SIM2SIM_BROWSER_HANDOFF.md`.

- Read the handoff plus only the files the next hypothesis needs. Trust verified contracts; do not redo the Isaac/ONNX/95-obs audits or scan `archive/`.
- One targeted change per attempt. Before launching, state hypothesis, exact command, timeout, result dir.
- Jobs > ~2 min: launch once, detached and bounded, then end the turn. No sleep/poll/"still running" loops, no Monitor/ScheduleWakeup polling:
  ```bash
  tools/experiment_loop/run_detached.sh --timeout 1800 \
    --result-dir tools/experiment_loop/results/NAME -- CMD ARGS...
  ```
  Then say: "I'm stopping polling. Please check back with me to inspect the result and continue; I won't automatically resume."
- On return: read `status.env` and logs once; inspect only the metrics/videos needed. Failed jobs never become champions. KEEP/REJECT against the real browser gate, not just Python MuJoCo reward.
- Keep stdout short; detailed telemetry goes to files. No subagents unless they have separate substantial work.
- Never overwrite protected baselines; don't exceed the agreed experiment bound.
