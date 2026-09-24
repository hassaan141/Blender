# Low-usage workflow for Bingo experiments

Use this with [SIM2SIM_BROWSER_HANDOFF.md](SIM2SIM_BROWSER_HANDOFF.md). The main saving is to let long training/evaluation jobs run without an AI repeatedly checking unchanged logs. This does **not** reduce the compute cost of training; it reduces model turns spent waiting and rereading context.

## Shared rules

1. Read the handoff and the specific files for the next hypothesis. Trust already verified contracts unless new evidence contradicts them. Do not repeat the Isaac/ONNX/95-observation audits or scan the entire archive.
2. Make one targeted change per attempt. Before launching, record the hypothesis, exact command, timeout, and result directory. Bound the job by both its own iteration/step limit and an external `timeout`.
3. Keep short inspections and focused checks in the foreground. For a long job, launch it once, confirm only that launch succeeded, then **end the AI turn**. Do not use a cycle of sleep → log read → “still running.”
4. On completion, read the exit status and saved logs **once**; inspect only the metrics/videos needed for the decision. Failed jobs never become champions. Record KEEP/REJECT against the real browser gate, not only Python MuJoCo reward.
5. Keep stdout concise: write detailed telemetry to files and print only the final status, paths, and important metrics. Avoid agents/subagents unless they have separate substantial work; two models reviewing the same unfinished job waste usage.

## Codex: automatic one-time wake-up

The installed `codex` CLI supports `codex queue --thread ... --message ...`. The repository runner [run_and_wake.sh](tools/experiment_loop/run_and_wake.sh) runs a command once with a finite timeout, saves `command.txt`, `stdout.log`, `stderr.log`, `status.env`, and `queue.log`, then sends exactly one completion message. Use it only when `CODEX_THREAD_ID` is available and the job can safely continue after the turn ends.

```bash
cd /pub0/muhammadf/Blender
setsid nohup tools/experiment_loop/run_and_wake.sh \
  --thread "$CODEX_THREAD_ID" --timeout 1800 \
  --result-dir tools/experiment_loop/results/NAME \
  -- EXACT_COMMAND ARGUMENTS \
  >tools/experiment_loop/results/NAME.launch.log 2>&1 </dev/null &
```

End the turn after launch. When the queued message arrives, inspect the result directory once and continue. If `queue_status=failed`, the logs still exist, but automatic resumption did not happen. Do not add a polling fallback. See [the existing runner notes](docs/experiment_loop/LONG_COMMAND_WAKEUP.md).

## Claude Code: stop and manually resume

The locally installed `claude` CLI supports background sessions and `--resume`, but its help does **not** show a `codex queue` equivalent. Do not assume Codex's queue command can wake a Claude conversation. For a long experiment, ask Claude to launch a bounded detached command that writes status and logs, then end its turn. When you return, resume the Claude session and have it inspect those artifacts once. `claude --bg` can keep a Claude session running, but by itself does not prevent the model from repeatedly checking the job.

A shell pattern for Claude's terminal, with the actual command passed as arguments after the result directory:

```bash
result_dir=tools/experiment_loop/results/NAME
mkdir -p "$result_dir"
setsid nohup bash -c '
  result_dir=$1; shift
  timeout --foreground --signal=TERM --kill-after=30s 1800 "$@" \
    >"$result_dir/stdout.log" 2>"$result_dir/stderr.log"
  code=$?
  printf "exit_code=%s\n" "$code" >"$result_dir/status.env"
' _ "$result_dir" EXACT_COMMAND ARGUMENTS \
  >"$result_dir/launch.log" 2>&1 </dev/null &
```

Claude should report the result directory and say: **“I’m stopping polling. Please check back with me to inspect the result and continue; I won’t automatically resume.”** When you return, `claude --resume <session-id>` is available if you are using Claude Code's CLI; otherwise return to the same conversation.

### Prompt to give Claude

> Read `SIM2SIM_BROWSER_HANDOFF.md` and `LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md` first. Continue from the recorded browser-probe decision. Use targeted file reads. For any training or evaluation expected to take more than two minutes, launch one bounded detached job with a named result directory, saved logs, and a finite timeout. Confirm only that it started, then end your turn without polling. Tell me the result path. When I return, inspect status and logs once, decide what they mean, and continue. Do not run more experiments than the agreed bound, overwrite protected baselines, or promote a candidate that fails the normal browser survival probe.

For the next Bingo attempt, the highest-value work is matching Python training/evaluation to the browser's expressive-joint targets and real stand-to-command transition. [The probe evidence](docs/experiment_loop/sim2sim/browser_probe.md) already shows that adding timing jitter is not the first fix.
