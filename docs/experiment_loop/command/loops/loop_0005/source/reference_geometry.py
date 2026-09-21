"""Remove common translation from the teacher turn bank for pivot commands.

Uses the retained task-space yaw deformation, contacts, and paw heights. All IK
uses the validated Bingo kinematics/limits read-only. No actuator changes.
"""
import ast,sys,json,argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
from common import ROOT,HOME,TEACHERS
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HOME/'references');parser.add_argument('--pivot-bl-lift-mm',type=float,default=0.);args=parser.parse_args()
sys.path.insert(0,str(ROOT/'stage2'))
from v4_kinematics import V4Kin,LEGS
fn=next(n for n in ast.parse((ROOT/'scripts/retarget.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve_leg')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'scripts/retarget.py','exec'))
r=np.load(TEACHERS['turning'][1]);b=np.load(TEACHERS['turning'][1].with_name('turn_bank.npz'))
kin=V4Kin(ROOT/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf');q=r['dof_positions'].astype(float);fps=float(r['fps']);tips=np.array([[kin.leg_fk(l,row[j*3:j*3+3])[0] for j,l in enumerate(LEGS)] for row in q]);means=tips[:-1].mean(0)
height=np.zeros(len(q));swing=~r['contacts'][:-1,2].astype(bool);count=len(swing)
for start in [i for i in range(count) if swing[i] and not swing[(i-1)%count]]:
 ids=[];i=start
 while swing[i] and len(ids)<count:ids.append(i);i=(i+1)%count
 if len(ids)>=4:height[ids]=args.pivot_bl_lift_mm/1000*np.sin(np.linspace(0,np.pi,len(ids)))**2
height[-1]=height[0];rotation=Rotation.from_quat(r['body_rotations'][:,0][:,[1,2,3,0]]).as_matrix();lift=np.einsum('nji,nj->ni',rotation,np.stack([np.zeros_like(height),np.zeros_like(height),height],axis=1))
qs=[];ts=[];errors=[]
for k,yaw in enumerate(b['yaw']):
 target=tips+b['tipdelta'][k];target[:,:,0]-=tips[:,:,0]-means[None,:,0];target[:,2]+=lift
 edited=q[:,:12].copy()
 for t in range(len(q)):
  for j,l in enumerate(LEGS):
   sl=slice(j*3,j*3+3);edited[t,sl],err=solve_leg(kin,l,target[t,j],q[t,sl],kin.leg_limits(l));errors.append(float(err))
 edited[-1]=edited[0];target[-1]=target[0];qs.append(edited-q[:,:12]);ts.append(target-tips)
qs=np.array(qs);ts=np.array(ts);assert max(errors)<.001;assert abs(qs).max()<.6
out=args.output;out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'pivot_bank.npz',qdelta=qs.astype('f4'),vdelta=np.gradient(qs,1/fps,axis=1).astype('f4'),tipdelta=ts.astype('f4'),yaw=b['yaw'],fps=fps)
(out/'pivot_audit.json').write_text(json.dumps({'additional_pivot_bl_lift_mm':args.pivot_bl_lift_mm,'ik_max_error_m':max(errors),'max_joint_delta_rad':float(abs(qs).max()),'method':'Remove common fore-aft displacement; preserve teacher rotational task-space deformation and paw height/contact timing'},indent=2)+'\n')
print((out/'pivot_audit.json').read_text())
# Backward geometry uses its own Jacobians/posture, not forward joint deltas.
r=np.load(TEACHERS['backward'][1]);q=r['dof_positions'].astype(float);fps=float(r['fps']);tips=np.array([[kin.leg_fk(l,row[j*3:j*3+3])[0] for j,l in enumerate(LEGS)] for row in q]);means=tips[:-1].mean(0);qs=[];ts=[];errors=[]
for yaw in b['yaw']:
 target=tips.copy();dx=tips[:,:,0]-means[None,:,0]
 target[:,:,0]=means[None,:,0]+dx*(1-.7*yaw*means[None,:,1]/.20)
 target[:,:,1]+=.7*yaw*means[None,:,0]/.20*dx
 edited=q[:,:12].copy()
 for t in range(len(q)):
  for j,l in enumerate(LEGS):
   sl=slice(j*3,j*3+3);edited[t,sl],err=solve_leg(kin,l,target[t,j],q[t,sl],kin.leg_limits(l));errors.append(float(err))
 edited[-1]=edited[0];target[-1]=target[0];qs.append(edited-q[:,:12]);ts.append(target-tips)
qs=np.array(qs);ts=np.array(ts);assert max(errors)<.001;assert abs(qs).max()<.6
np.savez_compressed(out/'backward_turn_bank.npz',qdelta=qs.astype('f4'),vdelta=np.gradient(qs,1/fps,axis=1).astype('f4'),tipdelta=ts.astype('f4'),yaw=b['yaw'],fps=fps)
(out/'backward_turn_audit.json').write_text(json.dumps({'ik_max_error_m':max(errors),'max_joint_delta_rad':float(abs(qs).max()),'method':'Body-twist deformation solved on retained backward teacher posture; signed phase handled by runtime'},indent=2)+'\n')
print((out/'backward_turn_audit.json').read_text())
