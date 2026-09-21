import csv,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'loop_runs'
sys.path.insert(0,str(OLD))
from selection_rules import refinement_selection
FIELDS=['run','decision','parent','survival','vx_mean','vx_std','vx_max','joint_accel_rms','accel_vs_original_pct','residual_action_rate_rms','fl_SP_saturation_pct','fl_knee_saturation_pct','mean_saturation_pct','reference_rms','all_speed_gates','reason']
original=json.loads((OLD/'baseline/metrics.json').read_text())
def row(name,m,decision,parent,reason):
 sat=m['torque_saturation_pct_per_joint']
 return dict(run=name,decision=decision,parent=parent,**{k:m[k] for k in ['survival','vx_mean','vx_std','vx_max','joint_accel_rms','residual_action_rate_rms']},accel_vs_original_pct=100*(m['joint_accel_rms']/original['joint_accel_rms']-1),fl_SP_saturation_pct=sat['fl_SP_J'],fl_knee_saturation_pct=sat['fl_knee'],mean_saturation_pct=sum(sat.values())/len(sat),reference_rms=m['reference_tracking_error_rad']['rms'],all_speed_gates=.2<=m['vx_mean']<=.3 and m['vx_std']<.06 and m['vx_max']<.4,reason=reason)
if not (ROOT/'leaderboard.csv').exists():
 m=json.loads((ROOT/'start_recheck/metrics.json').read_text())
 with (ROOT/'leaderboard.csv').open('w') as f:
  w=csv.DictWriter(f,FIELDS,lineterminator="\n");w.writeheader();w.writerow(row('start_recheck',m,'START','original/run_06','audited best valid checkpoint; fixed original-baseline cap'))
if len(sys.argv)>1:
 name=sys.argv[1];p=ROOT/name;e=json.loads((p/'experiment.json').read_text());m=json.loads((p/'metrics.json').read_text());parent=json.loads((ROOT/e['parent_run']/'metrics.json').read_text());reasons=refinement_selection(m,parent,original);decision='REJECT' if reasons else 'KEEP';reason='; '.join(reasons) or 'refinement priorities improved; all mandatory guards passed'
 e.update(status='complete',decision=decision,reason=reason,checkpoint=m['checkpoint']);(p/'experiment.json').write_text(json.dumps(e,indent=2))
 existing=list(csv.DictReader((ROOT/'leaderboard.csv').open()));assert name not in [r['run'] for r in existing]
 with (ROOT/'leaderboard.csv').open('a') as f:csv.DictWriter(f,FIELDS,lineterminator="\n").writerow(row(name,m,decision,e['parent_run'],reason))
 best=name if decision=='KEEP' else e['parent_run'];bm=json.loads((ROOT/best/'metrics.json').read_text());(ROOT/'best.json').write_text(json.dumps({'run':best,'checkpoint':bm['checkpoint'],'original_acceleration_baseline':original['joint_accel_rms'],'absolute_acceleration_ceiling':1.2*original['joint_accel_rms']},indent=2))
 print(f"{name}: {decision}; vx {m['vx_mean']:.3f} ± {m['vx_std']:.3f}, peak {m['vx_max']:.3f}, accel {m['joint_accel_rms']:.2f}, rate {m['residual_action_rate_rms']:.4f}; {reason}")
