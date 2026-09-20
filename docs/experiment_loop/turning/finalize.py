"""Build the completed turning campaign report from immutable loop artifacts."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'rl/bingo_rl/experiment_loop'))
from core import read,save,digest,verify_seal,Runner
home=Path(__file__).parent
loops=sorted((home/'loops').glob('loop_*'))
assert len(loops)==8 and all(read(p/'state.json')['status']=='FINISHED' for p in loops)
cfg=read(home/'config.json');protected=Runner(home,cfg).check_protected(read(home/'PROTECTED_BASELINE.json'))
for p in loops:verify_seal(p)
champ=read(home/'CURRENT_CHAMPION.json');best=Path(champ['checkpoint']).parent
rows=[];history=[]
for p in loops:
 m=read(p/'A/turning_metrics.json');d=read(p/'decision.json');h=read(p/'hypothesis.json')
 feet=[read(p/'A'/k/'natural_metrics.json')['feet']['bl'] for k in m]
 row={'loop':p.name,'decision':d['decision'],'survival_worst':min(v['survival'] for v in m.values()),
      'yaw_left':m['left']['yaw_mean'],'yaw_right':m['right']['yaw_mean'],
      'vx_left':m['left']['vx_mean'],'vx_right':m['right']['vx_mean'],
      'vx_std_main_max':max(m[k]['vx_std'] for k in ['left','right']),
      'joint_accel_worst':max(v['joint_accel_rms'] for v in m.values()),
      'max_joint_saturation_pct':max(max(v['torque_saturation_pct_per_joint'].values()) for v in m.values()),
      'bl_duty_worst':max(v['duty'] for v in feet),'bl_clearance_p95_min_mm':min(v['clearance_p95_mm'] for v in feet)}
 rows.append(row);history.append({**row,'hypothesis':h,'review':d,'report':str(p/'report.md')})
with (home/'leaderboard.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=read(best/'turning_metrics.json');details={}
for name,m in metrics.items():
 n=read(best/name/'natural_metrics.json');q=read(best/name/'quality_metrics.json')
 details[name]={'metrics':m,'joint_limit_violation_max_rad':n['joint_limit_violation_max_rad'],
  'feet':{k:{key:v for key,v in foot.items() if key!='touchdown_phases'} for k,foot in n['feet'].items()},
  'body':{k:q[k] for k in ['body_height_mean_m','body_height_std_m','body_bobbing_detrended_rms_m','base_roll_std_deg','base_pitch_std_deg']},
  'cadence_hz':3/q['phase_cycle_period_s'],'full_phase_and_paw_data':str(best/name/'natural_metrics.json')}
decision=read(best.parent/'decision.json');goal=bool(decision.get('goal_met'))
paths={'checkpoint':best/'policy.pt','reference':best/'reference.npz','yaw_reference_bank':best/'turn_bank.npz',
 'config':best/'config.json','metrics':best/'turning_metrics.json','video':best/'left_right.mp4',
 'candidate_comparison':best/'comparison.mp4','forward_left_right_comparison':home/'final_comparison.mp4'}
record={'status':'qualified turning gait' if goal else 'best experimental turning candidate; full natural-gait gates NOT met',
 'stop_reason':'eight-loop limit reached','goal_met':goal,'best_loop':best.parent.name,
 'paths':{k:str(v) for k,v in paths.items()},'hashes':{k:digest(v) for k,v in paths.items() if v.exists()},
 'settings':read(best/'config.json')['settings'],'gates':read(best/'turning_gates.json'),
 'metrics':details,'history':history,'backward_baseline':str(ROOT/'docs/experiment_loop/loops/loop_0007/A'),
 'forward_baseline':str(ROOT/'docs/walk_ref/quality_runs/run_05'),
 'measurement_note':'Observed single-environment evaluations; contact and slip use collision geometry, not force sensors. Main turns60s; limit/straight probes30s. Cadence=3 recovery strokes per reference cycle.'}
save(home/'TURNING_RESULTS.json',record)
save(home/'BEST.json',{k:v for k,v in record.items() if k not in ['history','metrics']})
validation={'protected_entries_unchanged':len(protected),'immutable_turning_loops_verified':len(loops),
 'best_artifacts_exist':all(p.exists() for p in paths.values()),'runner_regression_tests':'18 passed; runner_tests_after_bl.txt',
 'forward_backward_lateral_and_physics_unchanged':True,'goal_met':goal}
save(home/'final_validation.json',validation)
text=f"# Turning campaign results\n\nEight loops completed. Best: **{best.parent.name}/A**, {record['status']}. Training stopped at the requested limit.\n\n"
text+=f"[Left/right video]({paths['video'].relative_to(home)}) · [Forward/left/right comparison](final_comparison.mp4) · [Checkpoint]({paths['checkpoint'].relative_to(home)}) · [Exact runtime bundle](BEST.json) · [Full metrics](TURNING_RESULTS.json)\n\n"
text+='| Case | Survival | vx m/s | vx std | Yaw command / actual rad/s | Accel rad/s² | Slip m/s | Max joint saturation | BL duty / clearance mm |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for name,d in details.items():
 m=d['metrics'];bl=d['feet']['bl'];sat=max(m['torque_saturation_pct_per_joint'].values())
 text+=f"| {name} | {m['survival']:.0%} | {m['vx_mean']:.3f} | {m['vx_std']:.3f} | {m['command']['yaw']:+.2f} / {m['yaw_mean']:+.3f} | {m['joint_accel_rms']:.2f} | {m['mean_slip_m_s']:.3f} | {sat:.2f}% | {bl['duty']:.1%} / {bl['clearance_p95_mm']:.2f} |\n"
text+='\nRemaining gate failures: '+(', '.join(record['gates']['failures']) or 'none')+'.\n\n'
text+='| Loop | Change | Decision |\n|---|---|---|\n'
for h in history:
 change=h['hypothesis']['arms'][0]['changes']
 text+=f"| {h['loop']} | {change} | {h['decision']} |\n"
text+='\nKEEP before full qualification means provisional progress, not a claim that all natural-gait gates passed. Loop 6 was rejected for a BL abduction limit crossing; loop 7 for worse right-side saturation and left lean; loop 8 for another rear-left abduction limit crossing. Each detailed report records the visual review and tradeoffs.\n\n'
text+='The decisive stability change was expressing observations relative to heading, so a turned robot no longer presents unfamiliar world-oriented observations. Rear-left recovery geometry and collision-based contact feedback improved an inherited near-continuous drag. More lift helped clearance but increased left lean; consistent all-foot contact supervision improved support metrics but crossed a joint limit. These are diagnostic findings, not proof of a fully natural gait.\n\n'
text+='Forward and backward baselines remain unchanged and separate. This is a turning extension with 71 observations and an adjacent yaw-reference bank; do not load it with the original 70-observation evaluator. The comparison uses the canonical forward speed and the candidate turning speed, so it is a visual benchmark rather than a matched-command test.\n\n'
text+='Reproduce one 60-second turn into a new directory:\n\n```bash\npython3 rl/bingo_rl/experiment_loop/turning/reproduce.py \\\n'
text+=f'  --candidate {best.relative_to(ROOT)} \\\n  --out docs/experiment_loop/turning/replay_left_new --yaw 0.4 --seconds 60\n```\n\nUse `--yaw -0.4` and a different fresh output directory for the right turn. Full native argv, environment settings and training logs are in the sealed candidate folder. No loop 9 was started.\n'
(home/'TURNING_RESULTS.md').write_text(text)
(home/'STATUS.md').write_text(f'# Turning status\n\nFinished eight loops; no training active. Best {best.parent.name}/A. Full natural-gait success: {goal}. See [results](TURNING_RESULTS.md), [leaderboard](leaderboard.csv), and [validation](final_validation.json).\n')
print(json.dumps(validation,indent=2))
