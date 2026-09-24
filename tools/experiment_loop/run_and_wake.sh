#!/usr/bin/env bash
set -u

usage() {
  cat >&2 <<'EOF'
Usage:
  run_and_wake.sh --thread THREAD --timeout SECONDS [--result-dir DIR] -- COMMAND [ARGS...]
EOF
  exit 2
}

thread=""
timeout_s=""
result_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --thread) [[ $# -ge 2 ]] || usage; thread="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || usage; timeout_s="$2"; shift 2 ;;
    --result-dir) [[ $# -ge 2 ]] || usage; result_dir="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

[[ -n "$thread" && -n "$timeout_s" && $# -gt 0 ]] || usage

if [[ -z "$result_dir" ]]; then
  result_dir="tools/experiment_loop/results/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
mkdir -p "$result_dir"

printf 'command=' >"$result_dir/command.txt"
printf '%q ' "$@" >>"$result_dir/command.txt"
printf '\n' >>"$result_dir/command.txt"
start_epoch="$(date +%s)"
{
  printf 'status=running\n'
  printf 'started_epoch=%s\n' "$start_epoch"
  printf 'timeout_seconds=%s\n' "$timeout_s"
  printf 'result_dir=%s\n' "$result_dir"
} >"$result_dir/status.env"

set +e
timeout --foreground --signal=TERM --kill-after=30s "$timeout_s" "$@" \
  >"$result_dir/stdout.log" 2>"$result_dir/stderr.log"
exit_code=$?
set -e

end_epoch="$(date +%s)"
queue_message="experiment job finished: status=$exit_code; result_dir=$result_dir; inspect status.env, stdout.log, stderr.log once"
if command -v codex >/dev/null 2>&1 && codex queue --thread "$thread" --message "$queue_message" \
  >"$result_dir/queue.log" 2>&1; then
  queue_status=sent
else
  queue_status=failed
fi

{
  printf 'status=%s\n' "$([[ $exit_code -eq 0 ]] && echo complete || echo failed)"
  printf 'exit_code=%s\n' "$exit_code"
  printf 'started_epoch=%s\n' "$start_epoch"
  printf 'finished_epoch=%s\n' "$end_epoch"
  printf 'timeout_seconds=%s\n' "$timeout_s"
  printf 'result_dir=%s\n' "$result_dir"
  printf 'queue_status=%s\n' "$queue_status"
} >"$result_dir/status.env"

printf '%s\n' "$result_dir"
exit "$exit_code"
