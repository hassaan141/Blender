"""Speed/cadence constraints; only physical gait quality is ranked."""
import math

def score(m,q,b,bq):
 s=m['torque_saturation_pct_per_joint'];bs=b['torque_saturation_pct_per_joint']
 return .25*(s['fl_SP_J']/bs['fl_SP_J']+s['fl_knee']/bs['fl_knee'])+.25*m['joint_accel_rms']/b['joint_accel_rms']+.25*q['body_bobbing_detrended_rms_m']/bq['body_bobbing_detrended_rms_m']

def decide(m,q,parent,pq,b,bq):
 reasons=[]
 if any(not math.isfinite(m[k]) for k in ["survival","vx_mean","vx_std","vx_max","joint_accel_rms"]):return ["non-finite evaluation metrics"]
 if m['survival']!=1 or m['any_fell'] or q['warmup_falls']:reasons.append('survival failure')
 if not .19<=m['vx_mean']<=.23:reasons.append('mean speed outside constraint')
 if m['vx_std']>=.06:reasons.append('speed variation exceeds constraint')
 if m['vx_max']>=.4:reasons.append('peak speed exceeds constraint')
 if m['joint_accel_rms']>=b['joint_accel_rms']:reasons.append('joint acceleration not below retained baseline')
 for key in ['body_bobbing_detrended_rms_m','body_vertical_velocity_rms_m_s']:
  if not math.isfinite(q[key]) or q[key]>bq[key]:reasons.append(key+' exceeds baseline')
 for j in ['fl_SP_J','fl_knee']:
  if m['torque_saturation_pct_per_joint'][j]>b['torque_saturation_pct_per_joint'][j]:reasons.append(j+' saturation exceeds baseline')
 if sum(m['torque_saturation_pct_per_joint'].values())>sum(b['torque_saturation_pct_per_joint'].values()):reasons.append('total saturation exceeds baseline')
 if m['reference_tracking_error_rad']['rms']>1.25*b['reference_tracking_error_rad']['rms']:reasons.append('reference tracking guard failed')
 if q['phase_cycle_period_s'] is None or abs(q['phase_cycle_period_s']-bq['phase_cycle_period_s'])>m['step_dt']:reasons.append('cadence changed')
 if not math.isfinite(score(m,q,b,bq)) or score(m,q,b,bq)>=score(parent,pq,b,bq):reasons.append('quality score did not improve')
 return reasons
