#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
exec /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python \
  BASELINE_1/play.py \
  --checkpoint BASELINE_1/policy.pt \
  --references BASELINE_1/references \
  --vx 0 --yaw 0 --headless --device cuda:0 \
  --kit_args '--/rtx/verifyDriverVersion/enabled=false --no-window' "$@"
