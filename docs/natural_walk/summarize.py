from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[2];W=R/'docs/natural_walk';rows=[]
for name,path in [('locomotion_1',W/'baseline_eval')]+[(p.name,p/'evaluation') for p in sorted(W.glob('attempt_*'))]:
 if not (path/'metrics.json').exists():continue
 m=json.loads((path/'metrics.json').read_text());q=json.loads((path/'quality_metrics.json').read_text());n=json.loads((path/'natural_metrics.json').read_text());row={'run':name,'survival':m['survival'],'vx_mean':m['vx_mean'],'vx_std':m['vx_std'],'vx_max':m['vx_max'],'joint_acceleration_rms':m['joint_accel_rms'],'action_rate_rms':m['residual_action_rate_rms'],'fl_SP_sat_pct':m['torque_saturation_pct_per_joint']['fl_SP_J'],'fl_knee_sat_pct':m['torque_saturation_pct_per_joint']['fl_knee'],'mean_torque_sat_pct':float(np.mean(list(m['torque_saturation_pct_per_joint'].values()))),'cycle_period_s':q['phase_cycle_period_s'],'body_bobbing_mm':q['body_bobbing_detrended_rms_m']*1000,'roll_rms_deg':m['base_roll_rms_deg'],'pitch_rms_deg':m['base_pitch_rms_deg'],'height_mean_m':q['body_height_mean_m'],'joint_limit_violation_rad':n['joint_limit_violation_max_rad'],'slip_rms_mean':float(np.mean([v['contact_slip_rms_m_s'] for v in n['feet'].values() if v['contact_slip_rms_m_s'] is not None]))}
 for l,v in n['feet'].items():row[l+'_duty']=v['duty'];row[l+'_clearance_p95_mm']=v['clearance_p95_mm']
 row['decision']='BASELINE' if name=='locomotion_1' else (json.loads((path.parent/'decision.json').read_text())['decision'] if (path.parent/'decision.json').exists() else 'PENDING_VISUAL_REVIEW')
 rows.append(row)
if rows:
 with (W/'leaderboard.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
