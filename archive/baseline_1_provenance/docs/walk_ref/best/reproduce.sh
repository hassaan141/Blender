#!/usr/bin/env bash
# Evaluate the immutable retained baseline into a new directory; never train.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
if (( $# > 1 )); then echo "Usage: $0 [new-output-directory]" >&2; exit 2; fi
out_dir="${1:-$repo_root/outputs/walk_ref_reproduction/$(date -u +%Y%m%dT%H%M%S)-$$}"
out_dir="$(realpath -m -- "$out_dir")"
case "$out_dir/" in
  "$repo_root/docs/"*|"$repo_root/logs/"*|"$repo_root/archive/"*|"$repo_root/rl/"*)
    echo "Choose a fresh output directory outside docs, logs, archive and source." >&2; exit 2 ;;
esac
if [[ -e "$out_dir" ]]; then echo "Refusing to overwrite existing output: $out_dir" >&2; exit 2; fi
mkdir -p -- "$(dirname -- "$out_dir")"
mkdir -- "$out_dir"
cd -- "$repo_root"
exec env CUDA_VISIBLE_DEVICES=1 /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python   rl/tools/eval_walk_loop.py --headless --diagnostics   --checkpoint "$repo_root/docs/walk_ref/quality_runs/run_05/policy.pt"   --out_dir "$out_dir"   --kit_args '--/rtx/verifyDriverVersion/enabled=false --no-window'
