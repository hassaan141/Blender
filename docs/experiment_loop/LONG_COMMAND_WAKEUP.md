# Low-usage experiment jobs

Use `tools/experiment_loop/run_and_wake.sh` for bounded Isaac, MuJoCo, training, or evaluation commands that may run for more than a few minutes. It runs the command once, writes complete artifacts, and sends one `codex queue` completion message. It never polls the process or rereads live logs.

```bash
cd /pub0/muhammadf/Blender
setsid nohup tools/experiment_loop/run_and_wake.sh \
  --thread "$CODEX_THREAD_ID" \
  --timeout 1800 \
  -- rl/bingo_rl/run_experiment.sh --sim2sim \
  >tools/experiment_loop/launch.log 2>&1 </dev/null &
```

Replace the command after `--` with the exact bounded job. The runner creates a result directory under `tools/experiment_loop/results/` containing:

- `command.txt`
- `status.env`
- `stdout.log`
- `stderr.log`
- `queue.log`

When the queued message arrives, inspect those files once and continue from the recorded status. Do not add a polling loop. A nonzero exit, timeout, or failed queue call is recorded in `status.env`; the job artifacts remain available for recovery.

Do not use this wrapper for interactive commands, approval-dependent actions, or destructive operations that need supervision. Keep training bounded with an explicit timeout and an experiment-specific output directory.

## Installation note

This checkout exposes `.codex` as read-only, so the maintained copy lives in `tools/experiment_loop/`. In a checkout where `.codex` is writable, it can be installed as the reusable runner with:

```bash
mkdir -p .codex/queue
cp tools/experiment_loop/run_and_wake.sh .codex/queue/run-and-wake.sh
```

No locomotion experiment is launched by installing this workflow.
