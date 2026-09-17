"""Read-only gait diagnostics. Contact is a height proxy, not a force sensor."""
import json
from pathlib import Path
import numpy as np

class QualityRecorder:
    def __init__(self, base, names, out):
        self.names=names;self.out=Path(out);self.warmup_falls=0
        self.motion_file=base.cfg.motion_file
        self.feet=[str(x) for x in base.robot.data.body_names] if hasattr(base.robot.data,'body_names') else []
        self.log={k:[] for k in ['phase','q_ref','q_act','root_z','root_vz','tip_z','ref_contact','computed_torque','raw_action']}
    def capture(self,base,action):
        d=base.robot.data; tips,_=base._foot_tips_local()
        idx=np.clip(np.round(base._current_times()*base._ref_fps).astype(int),0,base._n_ref_frames-1)
        values={'phase':base._clip_time/base.motion_duration,'q_ref':base._ff_leg_q,'q_act':d.joint_pos[:,base.robot_ctrl_indexes],
                'root_z':d.body_pos_w[:,base.ref_body_index,2]-base.scene.env_origins[:,2],
                'root_vz':d.body_lin_vel_w[:,base.ref_body_index,2], 'tip_z':tips[:,:,2]-base.scene.env_origins[:,2,None],
                'ref_contact':base._ref_contacts[idx], 'computed_torque':d.computed_torque[:,base.robot_ctrl_indexes], 'raw_action':action}
        for k,v in values.items():self.log[k].append(v.detach().cpu().numpy().copy())
    def finish(self,D,metrics,dt):
        extra={k:np.asarray(v) for k,v in self.log.items()};np.savez_compressed(self.out/'diagnostic_trace.npz',**D,**extra,joint_names=np.array(self.names),step_dt=dt)
        # All reported extras are restricted to pre-first-fall samples, like the evaluator.
        alive=~(np.cumsum(D['done'].astype(int),axis=0)>0); valid=alive[:,0]
        A={k:v[:,0][valid] for k,v in {**D,**extra}.items()}
        n=len(A['phase']);t=np.arange(n)*dt; rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
        z=A['root_z'];detrend=z-np.polyval(np.polyfit(t,z,1),t) if n>1 else z-z.mean()
        wraps=np.flatnonzero(np.diff(A['phase'])<-.5)+1
        summary={'motion_file':self.motion_file,'warmup_falls':self.warmup_falls,'samples_before_fall':n,'contact_definition':'shank-tip height < threshold; no contact-force sensor in this environment',
            'body_height_mean_m':float(z.mean()),'body_height_std_m':float(np.std(z)), 'body_bobbing_detrended_rms_m':rms(detrend),'body_height_p95_p05_m':float(np.percentile(z,95)-np.percentile(z,5)),
            'body_vertical_velocity_rms_m_s':rms(A['root_vz']),'body_vertical_accel_rms_m_s2':rms(np.diff(A['root_vz'])/dt),
            'phase_cycle_period_s':float(np.mean(np.diff(wraps))*dt) if len(wraps)>1 else None,
            'base_roll_std_deg':float(np.std(A['roll_deg'])),'base_pitch_std_deg':float(np.std(A['pitch_deg'])), 'joints':{},'feet':{}}
        phase_bin=np.minimum((A['phase']*12).astype(int),11)
        for j,name in enumerate(self.names):
            leg=j//3;sat=np.abs(A['torque'][:,j])>=2.97;r=A['resid'][:,j];err=A['q_ref'][:,j]-A['q_act'][:,j];targeterr=err+r;c=A['tip_z'][:,leg]<.03
            def mean(x,mask):return float(np.mean(x[mask])) if np.any(mask) else None
            summary['joints'][name]={'saturation_pct':float(100*sat.mean()),'residual_mean_rad':float(r.mean()),'residual_rms_rad':rms(r),'residual_abs_max_rad':float(np.max(np.abs(r))),
                'residual_rate_rms_rad_step':rms(np.diff(r)),'tracking_error_rms_rad':rms(err),'target_error_rms_rad':rms(targeterr),
                'joint_accel_rms':rms(np.diff(A['qdot'][:,j])/dt),'reference_accel_rms':rms(np.diff(A['q_ref'][:,j],n=2)/dt**2),
                'computed_torque_rms_Nm':rms(A['computed_torque'][:,j]),'computed_torque_abs_max_Nm':float(np.max(np.abs(A['computed_torque'][:,j]))),
                'saturation_during_contact_pct':mean(sat*100,c),'saturation_during_swing_pct':mean(sat*100,~c),'residual_when_saturated_rad':mean(r,sat),
                'saturation_phase_bins_pct':[mean(sat*100,phase_bin==k) for k in range(12)]}
        for leg,name in enumerate(['fl','fr','bl','br']):
            c=A['tip_z'][:,leg]<.03;rc=A['ref_contact'][:,leg]>.5;touch=(~c[:-1])&c[1:]
            summary['feet'][name]={'reference_contact_duty_pct':float(rc.mean()*100),'height_proxy_duty_pct':{str(th):float(np.mean(A['tip_z'][:,leg]<th)*100) for th in [.01,.02,.03]},
                'reference_proxy_mismatch_pct':float(np.mean(c!=rc)*100),'touchdown_count':int(touch.sum()),'touchdown_phase':A['phase'][1:][touch].tolist(),
                'tip_height_min_m':float(A['tip_z'][:,leg].min()),'tip_height_p95_m':float(np.percentile(A['tip_z'][:,leg],95)),
                'contact_phase_bins_pct':[float(c[phase_bin==k].mean()*100) if np.any(phase_bin==k) else None for k in range(12)]}
        self.out.joinpath('quality_metrics.json').write_text(json.dumps(summary,indent=2))
