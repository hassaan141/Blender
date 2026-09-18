#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd -- "$repo_root"
# REJECTED experimental candidate: reproduce for inspection, not as canonical best.
exec python3 docs/natural_walk/run_eval.py "review_attempt03_$(date -u +%Y%m%dT%H%M%S)_$$"   --reference docs/natural_walk/reference_02b/reference.npz   --checkpoint docs/natural_walk/attempt_03/policy.pt --contact-fix
