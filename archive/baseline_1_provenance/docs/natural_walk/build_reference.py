"""Conservative task-space naturalization; immutable source; v4 FK and seeded IK."""
from pathlib import Path
import sys,json,argparse,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'stage2'),str(R/'scripts')]
from v4_kinematics import V4Kin,LEGS
# Existing CLI calls main unconditionally; load only its pure IK function via AST.
import ast
_tree=ast.parse((R/'scripts/retarget.py').read_text())
_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve_leg')
exec(compile(ast.Module(body=[_fn],type_ignores=[]),'scripts/retarget.py','exec'))
p=argparse.ArgumentParser();p.add_argument('--name',default='reference_01');p.add_argument('--lift-mm',type=float,default=4);p.add_argument('--smooth',type=float,default=.15);p.add_argument('--legs',default='fl,fr,bl,br');p.add_argument('--recovery-swing',action='store_true');a=p.parse_args()
o=R/'docs/natural_walk'/a.name;o.mkdir(exist_ok=False)
src=R/'docs/walk_ref/refinement_runs/run_05/reference.npz';z=np.load(src);d={k:z[k].copy() for k in z.files};q=d['dof_positions'].astype(float);n=len(q)
kin=V4Kin(R/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf');rot=Rotation.from_quat(d['body_rotations'][:,0][:,[1,2,3,0]]).as_matrix()
tips=np.array([[kin.leg_fk(l,row[3*j:3*j+3])[0] for j,l in enumerate(LEGS)] for row in q]);target=tips.copy();dog=np.load(R/'docs/locomotion2/motions/source/dog02_walk_02.npz');profiles=[]
# Only complete mocap swing bouts: normalized time and clearance shape, no angles.
for j in range(4):
 swing=~dog['foot_contact'][:,j];edges=np.diff(np.r_[False,swing,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
 for s,e in zip(starts,ends):
  if s==0 or e==len(swing) or e-s<4:continue
  h=dog['paw_pos'][s:e,j,2].copy();h-=np.linspace(h[0],h[-1],len(h));h=np.maximum(h,0)
  if h.max()<.003:continue
  profiles.append(np.interp(np.linspace(0,1,101),np.linspace(0,1,len(h)),h/h.max()))
profile=gaussian_filter1d(np.mean(profiles,axis=0),4);profile*=np.sin(np.linspace(0,np.pi,101))**2;profile/=profile.max()
# Smooth only small local task-space excursions, retaining original posture/range.
target+=a.smooth*(gaussian_filter1d(tips,2.0,axis=0,mode='wrap')-tips)
for j in range(4):
 if LEGS[j] not in a.legs.split(','):continue
 swing=~d['contacts'][:-1,j];N=len(swing)
 if a.recovery_swing:
  x=gaussian_filter1d(tips[:-1,j,0],2,mode='wrap');dx=(np.roll(x,-1)-np.roll(x,1))/2
  swing=dx>0.20*dx.max()
  d['contacts'][:-1,j]=~swing;d['contacts'][-1,j]=d['contacts'][0,j]
 starts=[i for i in range(N) if swing[i] and not swing[(i-1)%N]]
 for s in starts:
  ids=[];i=s
  while swing[i] and len(ids)<N:ids.append(i);i=(i+1)%N
  if len(ids)<6:continue
  bump=np.interp(np.linspace(0,1,len(ids)),np.linspace(0,1,101),profile)*a.lift_mm/1000
  for ix,h in zip(ids,bump):target[ix,j]+=rot[ix].T@np.array([0,0,h])
target[-1]=target[0];errors=[]
for t in range(n):
 for j,l in enumerate(LEGS):
  sl=slice(3*j,3*j+3);q[t,sl],err=solve_leg(kin,l,target[t,j],q[t,sl],kin.leg_limits(l));errors.append(err)
# Keep duplicated seam pose identical.
q[-1,:12]=q[0,:12];delta=q[:,:12]-d['dof_positions'][:,:12]
assert np.max(np.abs(delta))<.12 and max(errors)<.001
old=d['dof_positions'].copy();d['dof_positions']=q.astype(np.float32);fps=float(d['fps']);d['dof_velocities']=np.gradient(q,1/fps,axis=0).astype(np.float32)
for t in range(n):
 for j,l in enumerate(LEGS):
  tip,kR,kp=kin.leg_fk(l,q[t,j*3:j*3+3]);oldtip,oldR,oldkp=kin.leg_fk(l,old[t,j*3:j*3+3]);localold=rot[t].T@(d['body_positions'][t,j+1]-d['body_positions'][t,0]);is_tip=np.linalg.norm(localold-oldtip)<np.linalg.norm(localold-oldkp)
  d['body_positions'][t,j+1]=d['body_positions'][t,0]+rot[t]@(tip if is_tip else kp)
  xyzw=Rotation.from_matrix(rot[t]@kR).as_quat();d['body_rotations'][t,j+1]=xyzw[[3,0,1,2]]
d['body_linear_velocities']=np.gradient(d['body_positions'],1/fps,axis=0).astype(np.float32)
for j in range(1,5):
 rr=Rotation.from_quat(d['body_rotations'][:,j][:,[1,2,3,0]]);omega=(rr[1:]*rr[:-1].inv()).as_rotvec()*fps;d['body_angular_velocities'][:,j]=np.vstack([omega,omega[-1]])
d['natural_walk_lift_mm']=np.array(a.lift_mm);d['natural_walk_task_smoothing']=np.array(a.smooth)
np.savez(o/'reference.npz',**d);info={'source':str(src),'mocap_profile_count':len(profiles),'lift_mm':a.lift_mm,'lift_legs':a.legs,'recovery_swing':a.recovery_swing,'task_space_smoothing':a.smooth,'joint_delta_max_rad':float(abs(delta).max()),'ik_error_max_m':max(errors),'cadence_and_body_motion':'unchanged','contact_timing':'selected-leg swing follows existing forward recovery strokes' if a.recovery_swing else 'unchanged','body_arrays':'FK recomputed; no copied dog angles'};(o/'reference.json').write_text(json.dumps(info,indent=2));np.savez(o/'task_space_edit.npz',source=tips,target=target,dog_clearance_profile=profile,joint_delta=delta);print(info)
