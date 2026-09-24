#!/usr/bin/env bash
# Claude Code counterpart of run_and_wake.sh: run one bounded command detached,
# save logs + exit status, and never wake anything. The user resumes the session.
#
# Usage: tools/experiment_loop/run_detached.sh --timeout SECS --result-dir DIR -- CMD ARGS...
set -euo pipefail

timeout_s="" result_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) timeout_s=$2; shift 2 ;;
    --result-dir) result_dir=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$timeout_s" && -n "$result_dir" && $# -gt 0 ]] || {
  echo "usage: $0 --timeout SECS --result-dir DIR -- CMD ARGS..." >&2; exit 2; }
[[ -e "$result_dir/status.env" ]] && { echo "refusing: $result_dir already has a result" >&2; exit 2; }

mkdir -p "$result_dir"
printf '%q ' "$@" >"$result_dir/command.txt"; echo >>"$result_dir/command.txt"
date -Is >"$result_dir/started_at"

setsid nohup bash -c '
  result_dir=$1; timeout_s=$2; shift 2
  timeout --signal=TERM --kill-after=30s "$timeout_s" "$@" \
    >"$result_dir/stdout.log" 2>"$result_dir/stderr.log"
  code=$?
  { printf "exit_code=%s\n" "$code"; printf "finished_at=%s\n" "$(date -Is)"; } >"$result_dir/status.env"
' _ "$result_dir" "$timeout_s" "$@" >"$result_dir/launch.log" 2>&1 </dev/null &

echo "launched pid=$! result_dir=$result_dir timeout=${timeout_s}s"
