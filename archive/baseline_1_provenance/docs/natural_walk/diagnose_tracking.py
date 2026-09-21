from pathlib import Path
import sys,numpy as np,json
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'stage2'))
from v4_kinematics import V4Kin
kin=V4Kin(R/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf');out={}
for name,folder in [('baseline',R/'docs/natural_walk/baseline_eval'),('attempt_01',R/'docs/natural_walk/attempt_01/evaluation'),('reference_02b_preflight',R/'docs/natural_walk/reference_02b/preflight')]:
 d=np.load(folder/'diagnostic_trace.npz');p=np.load(folder/'paw_trajectories.npz');rs=Rotation.from_quat(p['root_quaternion'][:,[1,2,3,0]]).as_matrix();qr=d['q_ref'][:,0,6:9];qa=d['q_act'][:,0,6:9];res=d['resid'][:,0,6:9];ref=[];target=[];actual=[]
 for t in range(len(qr)):
  for arr,q in [(ref,qr[t]),(target,qr[t]+res[t]),(actual,qa[t])]:arr.append((rs[t]@kin.leg_fk('bl',q)[0])[2]+p['root_position'][t,2])
 ref=np.array(ref);target=np.array(target);actual=np.array(actual);swing=d['ref_contact'][:,0,2]<.5
 out[name]={'bl_swing_samples':int(swing.sum()),'residual_foot_height_shift_mean_mm':float((target-ref)[swing].mean()*1000),'target_minus_actual_foot_height_mean_mm':float((target-actual)[swing].mean()*1000),'ref_foot_z_mean_mm':float(ref[swing].mean()*1000),'actual_foot_z_mean_mm':float(actual[swing].mean()*1000)}
(R/'docs/natural_walk/tracking_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
