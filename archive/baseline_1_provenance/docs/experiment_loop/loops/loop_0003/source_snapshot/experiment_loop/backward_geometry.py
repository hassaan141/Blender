"""Bounded task-space edits of the existing Bingo-native reference. No physics changes."""
import ast,sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import ROOT,read,save
arm=Path(sys.argv[1]);cfg=read(arm/'config.json');settings=cfg['settings']
sys.path.insert(0,str(ROOT/'stage2'));from v4_kinematics import V4Kin,LEGS
fn=next(n for n in ast.parse((ROOT/'scripts/retarget.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve_leg')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'scripts/retarget.py','exec'))
src=np.load(ROOT/'docs/natural_walk/reference_02b/reference.npz');d={k:src[k].copy() for k in src.files};original=d['dof_positions'].astype(float);q=original.copy();N=len(q)
kin=V4Kin(ROOT/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf')
rot=Rotation.from_quat(d['body_rotations'][:,0][:,[1,2,3,0]]).as_matrix()
tips=np.array([[kin.leg_fk(l,row[j*3:j*3+3])[0] for j,l in enumerate(LEGS)] for row in q]);target=tips.copy()
for j,l in enumerate(LEGS):
 mean=tips[:-1,j].mean(axis=0)
 target[:,j,1]=mean[1]+settings.get('lateral_scale',1.)*(tips[:,j,1]-mean[1])
 scale=settings.get('front_stride_scale' if j<2 else 'rear_stride_scale',1.)
 target[:,j,0]=mean[0]+scale*(tips[:,j,0]-mean[0])
 lift=settings.get(l+'_lift_mm',0.)/1000
 swing=~d['contacts'][:-1,j].astype(bool);num=len(swing)
 for start in [i for i in range(num) if swing[i] and not swing[(i-1)%num]]:
  ids=[];i=start
  while swing[i] and len(ids)<num:ids.append(i);i=(i+1)%num
  if len(ids)<4:continue
  bump=np.sin(np.linspace(0,np.pi,len(ids)))**2*lift
  for ix,h in zip(ids,bump):target[ix,j]+=rot[ix].T@np.array([0.,0.,h])
target[-1]=target[0];errors=[]
for t in range(N):
 for j,l in enumerate(LEGS):
  sl=slice(3*j,3*j+3);q[t,sl],err=solve_leg(kin,l,target[t,j],q[t,sl],kin.leg_limits(l));errors.append(err)
q[-1,:12]=q[0,:12]
if max(errors)>.001 or abs(q[:,:12]-original[:,:12]).max()>.4:raise ValueError('Bounded IK edit failed; do not train')
for t in range(N):
 for j,l in enumerate(LEGS):
  tip,kr,kp=kin.leg_fk(l,q[t,j*3:j*3+3]);oldtip,_,oldkp=kin.leg_fk(l,original[t,j*3:j*3+3]);oldlocal=rot[t].T@(d['body_positions'][t,j+1]-d['body_positions'][t,0]);is_tip=np.linalg.norm(oldlocal-oldtip)<np.linalg.norm(oldlocal-oldkp)
  d['body_positions'][t,j+1]=d['body_positions'][t,0]+rot[t]@(tip if is_tip else kp)
  xyzw=Rotation.from_matrix(rot[t]@kr).as_quat();d['body_rotations'][t,j+1]=xyzw[[3,0,1,2]]
fps=float(src['fps'])*settings.get('phase_gain',1.);rate=-1.4*.2/.66;d['fps']=np.array(fps);d['dof_positions']=q.astype(np.float32)
d['dof_velocities']=(np.gradient(q,1/fps,axis=0)*rate).astype(np.float32)
d['body_linear_velocities']=(np.gradient(d['body_positions'],1/fps,axis=0)*rate).astype(np.float32)
for j in range(d['body_rotations'].shape[1]):
 rr=Rotation.from_quat(d['body_rotations'][:,j][:,[1,2,3,0]]);omega=(rr[1:]*rr[:-1].inv()).as_rotvec()*fps*rate;d['body_angular_velocities'][:,j]=np.vstack([omega,omega[-1]])
np.savez_compressed(arm/'reference.npz',**d);np.savez_compressed(arm/'task_space_edit.npz',source=tips,target=target,joint_delta=q-original)
save(arm/'reference_audit.json',{'source':'reference_02b','settings':settings,'ik_max_error_m':max(errors),'max_joint_delta_rad':float(abs(q-original).max()),
 'max_paw_delta_mm':float(np.linalg.norm(target-tips,axis=2).max()*1000),'contacts_unchanged':True,'signed_phase_rate':rate,'reference_period_s':(N-1)/fps/abs(rate),
 'task_space_geometry':'Lateral excursions scaled about each foot mean; optional fore-aft stride and compact swing lift changes. All parameters explicit and cumulative.',
 'feet':{l:{'source_range_mm':(np.ptp(tips[:,j],axis=0)*1000).tolist(),'target_range_mm':(np.ptp(target[:,j],axis=0)*1000).tolist()} for j,l in enumerate(LEGS)}})
