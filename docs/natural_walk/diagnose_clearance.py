"""Read-only attribution: marker clearance versus collision geometry under actual body pose."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'stage2'),str(R/'stage4')]
from v4_kinematics import V4Kin
from contact_model import ContactModel
kin=V4Kin(R/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf');cm=ContactModel();out={}
for name,folder in [('baseline',R/'docs/natural_walk/baseline_eval'),('attempt_01',R/'docs/natural_walk/attempt_01/evaluation'),('ref02_preflight',R/'docs/natural_walk/reference_02b/preflight'),('attempt_02',R/'docs/natural_walk/attempt_02/evaluation')]:
 d=np.load(folder/'diagnostic_trace.npz');p=np.load(folder/'paw_trajectories.npz');rots=Rotation.from_quat(p['root_quaternion'][:,[1,2,3,0]]).as_matrix();vals={k:[] for k in ['reference','residual_target','actual_fk']}
 for t in range(len(p['phase'])):
  for k,q in [('reference',d['q_ref'][t,0,6:9]),('residual_target',d['q_ref'][t,0,6:9]+d['resid'][t,0,6:9]),('actual_fk',d['q_act'][t,0,6:9])]:
   *_,support,_=kin.leg_points('bl',q,support_hull=cm.hull['bl_knee'],world_R=rots[t]);vals[k].append(float((rots[t]@support)[2]+p['root_position'][t,2]))
 swing=d['ref_contact'][:,0,2]<.5
 out[name]={k:{'swing_mean_mm':float(np.array(v)[swing].mean()*1000),'swing_fraction_below_3mm':float(np.mean(np.array(v)[swing]<.003))} for k,v in vals.items()};out[name]['fk_vs_recorded_collision_height_max_error_mm']=float(np.max(np.abs(np.array(vals['actual_fk'])-p['collision_clearance'][:,2]))*1000)
(R/'docs/natural_walk/clearance_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
