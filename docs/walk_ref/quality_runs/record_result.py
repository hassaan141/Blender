import csv,json,pathlib,sys
from selection import decide,score
R=pathlib.Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
b=load(R/'baseline/metrics.json');bq=load(R/'baseline/quality_metrics.json')
FIELDS=['run','decision','parent','survival','vx_mean','vx_std','vx_max','joint_accel_rms','fl_SP_sat_pct','fl_knee_sat_pct','bobbing_rms_mm','vertical_velocity_rms','action_rate_rms','cadence_s','quality_score','reason']
def row(name,m,q,decision,parent,reason):
 return dict(run=name,decision=decision,parent=parent,**{k:m[k] for k in ['survival','vx_mean','vx_std','vx_max','joint_accel_rms']},fl_SP_sat_pct=m['torque_saturation_pct_per_joint']['fl_SP_J'],fl_knee_sat_pct=m['torque_saturation_pct_per_joint']['fl_knee'],bobbing_rms_mm=1000*q['body_bobbing_detrended_rms_m'],vertical_velocity_rms=q['body_vertical_velocity_rms_m_s'],action_rate_rms=m['residual_action_rate_rms'],cadence_s=q['phase_cycle_period_s'],quality_score=score(m,q,b,bq),reason=reason)
if not (R/'leaderboard.csv').exists():
 with (R/'leaderboard.csv').open('w') as f:
  w=csv.DictWriter(f,FIELDS,lineterminator='\n');w.writeheader();w.writerow(row('baseline',b,bq,'BASELINE','refinement/run_05','retained gait remeasured; exact scalar metrics'))
if len(sys.argv)>1:
 name=sys.argv[1];p=R/name;e=load(p/'experiment.json');m=load(p/'metrics.json');q=load(p/'quality_metrics.json');pm=load(R/e['parent_run']/'metrics.json');pq=load(R/e['parent_run']/'quality_metrics.json');reasons=decide(m,q,pm,pq,b,bq);decision='REJECT' if reasons else 'KEEP';reason='; '.join(reasons) or 'physical gait quality improved; speed/cadence and baseline quality guards passed'
 e.update(status='complete',decision=decision,reason=reason,checkpoint=m['checkpoint']);(p/'experiment.json').write_text(json.dumps(e,indent=2))
 rows=list(csv.DictReader((R/'leaderboard.csv').open()));assert name not in [x['run'] for x in rows]
 with (R/'leaderboard.csv').open('a') as f:csv.DictWriter(f,FIELDS,lineterminator='\n').writerow(row(name,m,q,decision,e['parent_run'],reason))
 best=name if decision=='KEEP' else e['parent_run'];(R/'best.json').write_text(json.dumps({'run':best,'checkpoint':load(R/best/'metrics.json')['checkpoint']},indent=2))
 print(f"{name}: {decision}; FL SP/knee {m['torque_saturation_pct_per_joint']['fl_SP_J']:.2f}/{m['torque_saturation_pct_per_joint']['fl_knee']:.2f}%, accel {m['joint_accel_rms']:.2f}, bob {1000*q['body_bobbing_detrended_rms_m']:.3f} mm, vx {m['vx_mean']:.3f}±{m['vx_std']:.3f}; {reason}")
