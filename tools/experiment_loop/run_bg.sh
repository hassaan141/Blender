#!/usr/bin/env bash
# Token-saving runner for Claude Code's background Bash (run_in_background: true).
#
# Difference from run_detached.sh: this stays in the FOREGROUND of its own
# shell, so the harness owns the process and re-invokes Claude exactly once,
# when it exits. No polling, no wake-up service, no "still running" turns.
#
# Bulky output goes to files; only a compact summary reaches the transcript, so
# the auto-resume notification costs tens of tokens instead of thousands.
#
# Usage: tools/experiment_loop/run_bg.sh --timeout SECS --result-dir DIR \
#          [--tail N] [--grep REGEX] -- CMD ARGS...
#   --tail N     lines of stdout to echo at the end (default 15, 0 to suppress)
#   --grep RE    also echo stdout lines matching RE (the metrics you decide on)
set -euo pipefail

timeout_s="" result_dir="" tail_n=15 grep_re=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)    timeout_s=$2; shift 2 ;;
    --result-dir) result_dir=$2; shift 2 ;;
    --tail)       tail_n=$2; shift 2 ;;
    --grep)       grep_re=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$timeout_s" && -n "$result_dir" && $# -gt 0 ]] || {
  echo "usage: $0 --timeout SECS --result-dir DIR [--tail N] [--grep RE] -- CMD ARGS..." >&2
  exit 2; }
[[ -e "$result_dir/status.env" ]] && {
  echo "refusing: $result_dir already has a result" >&2; exit 2; }

mkdir -p "$result_dir"
printf '%q ' "$@" >"$result_dir/command.txt"; echo >>"$result_dir/command.txt"
started=$(date -Is); echo "$started" >"$result_dir/started_at"

set +e
timeout --signal=TERM --kill-after=30s "$timeout_s" "$@" \
  >"$result_dir/stdout.log" 2>"$result_dir/stderr.log"
code=$?
set -e

{ printf "exit_code=%s\n" "$code"
  printf "started_at=%s\n" "$started"
  printf "finished_at=%s\n" "$(date -Is)"
} >"$result_dir/status.env"

# --- compact summary: this, and only this, is what Claude pays tokens for ---
case "$code" in
  0)   verdict=OK ;;
  124) verdict="TIMEOUT after ${timeout_s}s" ;;
  *)   verdict="FAILED exit=$code" ;;
esac
echo "== $verdict  dir=$result_dir"
echo "== stdout=$(wc -l <"$result_dir/stdout.log") lines  stderr=$(wc -l <"$result_dir/stderr.log") lines"
if [[ -n "$grep_re" ]]; then
  echo "== matches /$grep_re/:"
  grep -E "$grep_re" "$result_dir/stdout.log" | tail -20 || echo "  (none)"
fi
if [[ "$tail_n" -gt 0 ]]; then
  echo "== last $tail_n stdout:"; tail -n "$tail_n" "$result_dir/stdout.log"
fi
if [[ "$code" -ne 0 ]]; then
  echo "== last 10 stderr:"; tail -n 10 "$result_dir/stderr.log"
fi
exit "$code"
