"""Record an original-loop result using the fixed original-baseline guards.

The historical audit comparison is retained separately from training ancestry.
Updating an existing row is idempotent and preserves its audit columns.
"""
import csv,json,math,pathlib,sys
from selection_rules import old_selection
root=pathlib.Path(__file__).resolve().parent
fields=['run','decision','parent','survival','vx_mean','vx_std','vx_max','velocity_rmse','joint_accel_rms','residual_action_rate_rms','torque_saturation_mean_pct','reference_rms','reason','original_decision','audited_against','accel_regression_pct']
def row(name,m,decision,parent,reason,original,comparison,baseline):
 return dict(run=name,decision=decision,parent=parent,**{k:m[k] for k in fields if k in m},velocity_rmse=math.hypot(m['vx_mean']-.25,m['vx_std']),torque_saturation_mean_pct=sum(m['torque_saturation_pct_per_joint'].values())/12,reference_rms=m['reference_tracking_error_rad']['rms'],reason=reason,original_decision=original,audited_against=comparison,accel_regression_pct=100*(m['joint_accel_rms']/baseline['joint_accel_rms']-1))
if __name__=='__main__':
 baseline=json.loads((root/'baseline/metrics.json').read_text())
 rows=list(csv.DictReader((root/'leaderboard.csv').open())) if (root/'leaderboard.csv').exists() else [row('baseline',baseline,'BASELINE','','original reference','BASELINE','baseline',baseline)]
 if len(sys.argv)>1:
  name=sys.argv[1];p=root/name;e=json.loads((p/'experiment.json').read_text());m=json.loads((p/'metrics.json').read_text())
  comparison=e.get('audit_comparison_checkpoint',e['parent_run']);parent=json.loads((root/comparison/'metrics.json').read_text())
  reasons=old_selection(m,parent,baseline);decision='REJECT' if reasons else 'KEEP';reason='; '.join(reasons) or 'corrected fixed-baseline protocol passed'
  e.update(status='complete',decision=decision,reason=reason,checkpoint=m['checkpoint']);(p/'experiment.json').write_text(json.dumps(e,indent=2))
  new=row(name,m,decision,e['parent_run'],reason,e.get('original_decision',decision),comparison,baseline)
  indexes=[i for i,r in enumerate(rows) if r['run']==name]
  if indexes:rows[indexes[0]]=new
  else:rows.append(new)
  print(f'{name}: {decision}; {reason}')
 with (root/'leaderboard.csv').open('w') as f:
  w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(rows)
