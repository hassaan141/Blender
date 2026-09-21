import pathlib,json,csv,math,sys
root=pathlib.Path('/pub0/muhammadf/Blender/docs/walk_ref/loop_runs')
fields=['run','decision','parent','survival','vx_mean','vx_std','vx_max','velocity_rmse','joint_accel_rms','residual_action_rate_rms','torque_saturation_mean_pct','reference_rms','reason']
def row(name,m,decision,parent,reason):
 return dict(run=name,decision=decision,parent=parent,**{k:m[k] for k in fields if k in m},velocity_rmse=math.hypot(m['vx_mean']-.25,m['vx_std']),torque_saturation_mean_pct=sum(m['torque_saturation_pct_per_joint'].values())/12,reference_rms=m['reference_tracking_error_rad']['rms'],reason=reason)
if not (root/'leaderboard.csv').exists():
 with (root/'leaderboard.csv').open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerow(row('baseline',json.loads((root/'baseline/metrics.json').read_text()),'BASELINE','','Reproduced exactly by baseline_recheck'))
if len(sys.argv)>1:
 name=sys.argv[1]; p=root/name; e=json.loads((p/'experiment.json').read_text());m=json.loads((p/'metrics.json').read_text());b=json.loads((root/e['parent_run']/'metrics.json').read_text()); orig=json.loads((root/'baseline/metrics.json').read_text())
 reasons=[]
 if m['survival']<b['survival']:reasons.append('survival regression')
 if m['reference_tracking_error_rad']['rms']>orig['reference_tracking_error_rad']['rms']*1.25:reasons.append('reference guard failed')
 if m['joint_accel_rms']>b['joint_accel_rms']*1.2:reasons.append('acceleration guard failed')
 if m['vx_max']>max(.4,b['vx_max']+.02):reasons.append('peak-speed guard failed')
 def speed_pass(x):return .2<=x['vx_mean']<=.3 and x['vx_std']<.06
 if not speed_pass(b):
  if math.hypot(m['vx_mean']-.25,m['vx_std'])>=math.hypot(b['vx_mean']-.25,b['vx_std']):reasons.append('velocity RMSE did not improve')
 else:
  if not speed_pass(m):reasons.append('mean/std speed gates regressed')
  if b['vx_max']>=.4:
   if m['vx_max']>=b['vx_max']:reasons.append('peak did not improve')
  else:
   if m['vx_max']>=.4:reasons.append('peak gate regressed')
   if m['joint_accel_rms']>=b['joint_accel_rms'] or m['residual_action_rate_rms']>b['residual_action_rate_rms']:reasons.append('shaking metrics did not improve together')
 decision='REJECT' if reasons else 'KEEP'; reason='; '.join(reasons) if reasons else 'priority metric improved; survival, peak, acceleration and reference guards passed'
 e.update(status='complete',decision=decision,reason=reason,checkpoint=m['checkpoint']);(p/'experiment.json').write_text(json.dumps(e,indent=2))
 with (root/'leaderboard.csv').open('a') as f:csv.DictWriter(f,fields).writerow(row(name,m,decision,e['parent_run'],reason))
 print(f"{name}: {decision} — survival {m['survival']:.0%}, vx {m['vx_mean']:.3f} ± {m['vx_std']:.3f}, peak {m['vx_max']:.3f}, accel {m['joint_accel_rms']:.2f}, residual rate {m['residual_action_rate_rms']:.4f}; {reason}")
