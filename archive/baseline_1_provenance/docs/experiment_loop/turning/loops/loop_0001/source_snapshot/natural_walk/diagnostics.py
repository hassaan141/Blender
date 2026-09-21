"""Extend existing read-only metrics with paw paths, collision clearance, slip and limits."""
from pathlib import Path
import sys,numpy as np,json
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'rl/tools'))
from walk_quality_diagnostics import QualityRecorder as Original
class QualityRecorder(Original):
    def __init__(self,base,names,out):
        super().__init__(base,names,out)
        self.extra={k:[] for k in ['paw_world','paw_local','knee_pos','knee_quat','root_position','root_quaternion']}
        self.limits=base.robot.data.joint_pos_limits[0,base.robot_ctrl_indexes].detach().cpu().numpy()
    def capture(self,base,action):
        super().capture(base,action);d=base.robot.data;pw,pl=base._foot_tips_local()
        vals=[pw,pl,d.body_pos_w[:,base.key_body_indexes],d.body_quat_w[:,base.key_body_indexes],d.body_pos_w[:,base.ref_body_index],d.body_quat_w[:,base.ref_body_index]]
        for k,v in zip(self.extra,vals):self.extra[k].append(v.detach().cpu().numpy().copy())
    def finish(self,D,metrics,dt):
        super().finish(D,metrics,dt)
        sys.path.insert(0,str(ROOT/'stage4'));from contact_model import ContactModel,quat_to_R
        cm=ContactModel();extra={k:np.asarray(v)[:,0] for k,v in self.extra.items()};N=len(extra['paw_world']);clear=np.zeros((N,4));speed=np.zeros((N,4));legs=['fl','fr','bl','br']
        for t in range(N):
            for j,l in enumerate(legs):
                pos=extra['knee_pos'][t,j];rot=quat_to_R(extra['knee_quat'][t,j]);hull=cm.hull[l+'_knee'];world=hull@rot.T+pos;i=world[:,2].argmin();clear[t,j]=world[i,2]
                if t:
                    prev=quat_to_R(extra['knee_quat'][t-1,j])@hull[i]+extra['knee_pos'][t-1,j]
                    speed[t,j]=np.linalg.norm((world[i]-prev)[:2])/dt
        valid=~(np.cumsum(D['done'][:,0])>0);q=np.array(self.log['q_act'])[:,0];phase=np.array(self.log['phase'])[:,0];out={'contact_definition':'lowest collision-hull clearance < 3 mm; geometry proxy, not force measurement','slip_definition':'horizontal velocity of current lowest hull material point, during geometric contact','joint_limit_violation_max_rad':float(np.maximum(np.maximum(self.limits[:,0]-q,q-self.limits[:,1]),0)[valid].max()),'feet':{}}
        for j,l in enumerate(legs):
            c=clear[:,j]<.003;mask=c&valid;touch=np.flatnonzero(c[1:]&~c[:-1])+1
            out['feet'][l]={'duty':float(c[valid].mean()),'touchdown_phases':phase[touch].tolist(),'clearance_p95_mm':float(np.percentile(clear[valid,j],95)*1000),'paw_local_range_mm':(np.ptp(extra['paw_local'][valid,j],axis=0)*1000).tolist(),'contact_slip_rms_m_s':float(np.sqrt(np.mean(speed[mask,j]**2))) if mask.any() else None}
        np.savez_compressed(self.out/'paw_trajectories.npz',**extra,collision_clearance=clear,contact_slip_speed=speed,phase=phase,valid=valid,dt=dt)
        (self.out/'natural_metrics.json').write_text(json.dumps(out,indent=2))
