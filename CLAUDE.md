# Working rules (low token usage)

Full rationale: `LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md`. Project state: `SIM2SIM_BROWSER_HANDOFF.md`.

- Read the handoff plus only the files the next hypothesis needs. Trust verified contracts; do not redo the Isaac/ONNX/95-obs audits or scan `archive/`.
- One targeted change per attempt. Before launching, state hypothesis, exact command, timeout, result dir.
- Jobs > ~2 min: launch once through the runner with `run_in_background: true`, then end the turn. The harness wakes you when it exits - no sleep/poll loops, no Monitor/ScheduleWakeup polling, no "still running" turns.
  ```bash
  tools/experiment_loop/run_bg.sh --timeout 1800 \
    --result-dir tools/experiment_loop/results/NAME \
    --grep 'mean_reward|survival|KEEP|REJECT' --tail 15 -- CMD ARGS...
  ```
  It writes stdout/stderr/status to files and prints ~20 lines, so the wake-up costs tens of tokens, not thousands. Read that summary first; open `stdout.log` only if it is ambiguous. If the session may be closed before the job ends, use `run_detached.sh` instead (no wake-up) and say: "I'm stopping polling. Please check back with me to inspect the result and continue; I won't automatically resume."
- On return: read `status.env` and logs once; inspect only the metrics/videos needed. Failed jobs never become champions. KEEP/REJECT against the real browser gate, not just Python MuJoCo reward.
- Keep stdout short; detailed telemetry goes to files. No subagents unless they have separate substantial work.
- Never overwrite protected baselines; don't exceed the agreed experiment bound.
