from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'stage2'))
from v4_kinematics import V4Kin,LEGS
kin=V4Kin(R/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf')
report={}
for name,path in [('baseline','docs/walk_ref/refinement_runs/run_05/reference.npz'),('blender','motions/bingo_walk_v4_upright_grounded_loopsmooth.npz')]:
 d=np.load(R/path);q=d['dof_positions'];tips=np.array([[kin.leg_fk(l,row[3*j:3*j+3])[0] for j,l in enumerate(LEGS)] for row in q]);rot=Rotation.from_quat(d['body_rotations'][:,0][:,[1,2,3,0]])
 world=np.einsum('tij,tkj->tki',rot.as_matrix(),tips)+d['body_positions'][:,0,None]
 rate=1.4*.25/.66;dt=1/float(d['fps'])/rate
 report[name]={'source':path,'playback_period_s':(len(q)-1)*dt,'joint_accel_rms':float(np.sqrt(np.mean(np.gradient(np.gradient(q[:,:12],dt,axis=0),dt,axis=0)**2))),'root_height_ptp_mm':float(np.ptp(d['body_positions'][:,0,2])*1000),'feet':{l:{'local_xyz_range_mm':(np.ptp(tips[:,j],axis=0)*1000).tolist(),'world_z_range_mm':(np.array([world[:,j,2].min(),world[:,j,2].max()])*1000).tolist(),'contact_duty':float(d['contacts'][:,j].mean()),'swing_frames':np.flatnonzero(~d['contacts'][:,j]).tolist()} for j,l in enumerate(LEGS)}}
 np.savez(R/'docs/natural_walk'/f'{name}_trajectories.npz',local=tips,world=world,contacts=d['contacts'],phase=np.linspace(0,1,len(q)))
d=np.load(R/'docs/locomotion2/motions/source/dog02_walk_02.npz');local=np.einsum('tji,tkj->tki',d['root_orient_rotmat'],d['paw_pos']-d['root_pos'][:,None]);report['dog']={'gait_period_s':float(d['gait_period_s']),'leg_order':d['leg_order'].tolist(),'root_height_ptp_m':float(np.ptp(d['root_pos'][:,2])),'feet':{str(l):{'contact_duty':float(d['foot_contact'][:,j].mean()),'local_xyz_range_m':np.ptp(local[:,j],axis=0).tolist(),'lift_world_range_m':float(np.ptp(d['paw_pos'][:,j,2]))} for j,l in enumerate(d['leg_order'])}}
(R/'docs/natural_walk/source_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
