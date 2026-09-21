#!/usr/bin/env bash
set -euo pipefail
BUNDLE="$(cd "$(dirname "$0")" && pwd)"
cd "$BUNDLE/../../../../.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
exec /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python "$BUNDLE/play.py" --checkpoint "$BUNDLE/policy.pt" --references "$BUNDLE/references" --vx 0 --yaw 0 --headless --device cuda:0 --kit_args '--/rtx/verifyDriverVersion/enabled=false --no-window' "$@"
