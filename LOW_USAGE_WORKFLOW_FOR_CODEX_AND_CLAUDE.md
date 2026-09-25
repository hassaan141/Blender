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

## Claude Code: one automatic wake-up (preferred)

Correction to what this file said before: Claude **can** resume itself after a long job, as long as the conversation is still open. Claude Code's `Bash` tool takes `run_in_background: true`; the harness owns the process, keeps it alive across turns, and re-invokes Claude exactly once when it exits. That is a wake-up, not polling - Claude spends zero turns waiting. It is how the BASELINE_1, Timid and Laidback runs here were actually driven, and it needs no `CODEX_THREAD_ID`, `codex queue`, `claude --bg` or `--resume`.

The cost to control is therefore not the waiting, it is the **completion message**: a training run that prints 4000 lines drops all 4000 into Claude's context at wake-up. [run_bg.sh](tools/experiment_loop/run_bg.sh) fixes that. It stays in the foreground of its own shell (so the harness still owns it), bounds the command with `timeout`, writes `command.txt`, `stdout.log`, `stderr.log` and `status.env`, and prints only a summary.

```bash
cd /pub0/muhammadf/Blender
tools/experiment_loop/run_bg.sh --timeout 1800 \
  --result-dir tools/experiment_loop/results/NAME \
  --grep 'mean_reward|survival|KEEP|REJECT' --tail 15 \
  -- EXACT_COMMAND ARGUMENTS
```

Launch that with `run_in_background: true`, confirm only that it started, and end the turn. The wake-up then costs about twenty lines instead of the whole log: verdict (`OK`, `TIMEOUT after Ns`, or `FAILED exit=N`), stdout/stderr line counts, the lines matching `--grep`, the stdout tail, and stderr's tail on failure. Choose `--grep` so the KEEP/REJECT call is visible without opening a file; open `stdout.log` only when the summary is genuinely ambiguous.

The other rules still hold: one bounded job per attempt, a named result directory, no `sleep`/poll loops, and no `Monitor` or `ScheduleWakeup` polling a job the harness already tracks. A long `ScheduleWakeup` fallback is justified only for work that can hang without ever exiting, which a `timeout`-bounded job cannot.

### Fallback: fully detached, no wake-up

If the conversation may be closed before the job finishes, use [run_detached.sh](tools/experiment_loop/run_detached.sh) instead: same artifacts, but `setsid`-detached and silent. Claude reports the result directory and says: **"I'm stopping polling. Please check back with me to inspect the result and continue; I won't automatically resume."** When you return, resume the session and have it read `status.env` and the logs once.

### Prompt to give Claude

> Read `SIM2SIM_BROWSER_HANDOFF.md` and `LOW_USAGE_WORKFLOW_FOR_CODEX_AND_CLAUDE.md` first. Continue from the recorded browser-probe decision. Use targeted file reads. For any training or evaluation expected to take more than two minutes, launch one bounded job through `tools/experiment_loop/run_bg.sh` with `run_in_background: true`, a named result directory, a finite timeout, and a `--grep` pattern covering the metrics the decision turns on. Confirm only that it started, then end your turn without polling; the harness will wake you when it exits. Read the summary first and open the logs only if it is ambiguous. Do not run more experiments than the agreed bound, overwrite protected baselines, or promote a candidate that fails the normal browser survival probe.

For the next Bingo attempt, the highest-value work is matching Python training/evaluation to the browser's expressive-joint targets and real stand-to-command transition. [The probe evidence](docs/experiment_loop/sim2sim/browser_probe.md) already shows that adding timing jitter is not the first fix.
