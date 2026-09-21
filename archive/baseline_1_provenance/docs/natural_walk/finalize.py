from pathlib import Path
import json,hashlib,os,csv
R=Path(__file__).resolve().parents[2];W=R/'docs/natural_walk'
decision={'decision':'REJECT','visual':'Improved but still brief BL clearance; substantial rear-foot skimming remains. Matched stride and overview do not establish a convincingly natural four-leg walk. Not promoted.','metrics':'Survival100%; BL geometric duty85.42% vs95.49%; BL clearance p95 7.12 vs2.91mm; BL slip .284 vs.373m/s; vx.1863 vs.1929; acceleration11.53 vs11.00. These local gains do not satisfy the visual gate.','stop':'Three completed training attempts without a qualifying visual improvement. No further training authorized in this loop.'}
(W/'attempt_03/decision.json').write_text(json.dumps(decision,indent=2))
best={'status':'bounded_loop_finished_without_visual_success','training_attempts':3,'natural_walk_milestone_achieved':False,'retained_best_reference':str(R/'docs/walk_ref/refinement_runs/run_05/reference.npz'),'retained_best_checkpoint':str(R/'docs/walk_ref/quality_runs/run_05/policy.pt'),'retained_best_video':str(R/'docs/walk_ref/quality_runs/best_render/eval.mp4'),'strongest_experimental_candidate':{'run':'attempt_03','reference':str(W/'reference_02b/reference.npz'),'checkpoint':str(W/'attempt_03/policy.pt'),'comparison_video':str(W/'attempt_03/comparison.mp4'),'decision':'REJECT; diagnostic improvement only, not a new canonical baseline'}}
(W/'best.json').write_text(json.dumps(best,indent=2))
protected=json.loads((W/'protected_sha256.json').read_text());bad=[p for p,h in protected.items() if not (R/p).exists() or hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
checks={'protected_files_checked':len(protected),'mismatches':bad,'training_attempts_completed':len(list(W.glob('attempt_*/policy.pt'))),'all_full_evaluations_survived':all(json.loads((W/f'attempt_{i:02d}/evaluation/metrics.json').read_text())['survival']==1 for i in range(1,4)),'original_baseline_metrics_reproduced':json.loads((W/'baseline_reproduction_check.json').read_text())['numeric_and_artifact_metadata_match_except_task_name'],'new_work_locations':['docs/natural_walk/','rl/bingo_rl/bingo_rl/natural_walk/'],'protected_integrity':'PASS' if not bad else 'FAIL','outcome':'No visual pass; canonical baseline retained'}
(W/'completion_audit.json').write_text(json.dumps(checks,indent=2));assert not bad and checks['training_attempts_completed']==3
(W/'reproduce_experimental.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd -- "$repo_root"
# REJECTED experimental candidate: reproduce for inspection, not as canonical best.
exec python3 docs/natural_walk/run_eval.py "review_attempt03_$(date -u +%Y%m%dT%H%M%S)_$$" \
  --reference docs/natural_walk/reference_02b/reference.npz \
  --checkpoint docs/natural_walk/attempt_03/policy.pt --contact-fix
''');(W/'reproduce_experimental.sh').chmod(0o755)
print(json.dumps(checks,indent=2))
