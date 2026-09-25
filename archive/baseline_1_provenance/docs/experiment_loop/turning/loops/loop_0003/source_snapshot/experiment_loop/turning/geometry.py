"""Yaw-conditioned task-space reference bank; original assets remain read-only."""
import ast,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read,save
sys.path.insert(0,str(ROOT/'stage2'))
from v4_kinematics import V4Kin,LEGS
fn=next(n for n in ast.parse((ROOT/'scripts/retarget.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve_leg')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'scripts/retarget.py','exec'))
def build(arm):
 c=read(arm/'config.json');s=c['settings'];src=np.load(ROOT/'docs/walk_ref/refinement_runs/run_05/reference.npz');d={k:src[k].copy() for k in src.files}
 kin=V4Kin(ROOT/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf');q=d['dof_positions'].astype(float);N=len(q);fps=float(d['fps'])
 tips=np.array([[kin.leg_fk(l,row[j*3:j*3+3])[0] for j,l in enumerate(LEGS)] for row in q]);means=tips[:-1].mean(axis=0)
 yawgrid=np.linspace(-.6,.6,7);qs=[];ts=[];errors=[]
 for yaw in yawgrid:
  target=tips.copy();edited=q.copy()
  for j,l in enumerate(LEGS):
   dx=tips[:,j,0]-means[j,0];gain=s.get('turn_geometry_gain',.7)
   target[:,j,0]=means[j,0]+dx*(1-gain*yaw*means[j,1]/.175)
   target[:,j,1]=means[j,1]+s.get('turn_lateral_scale',1.)*(tips[:,j,1]-means[j,1])+s.get('turn_lateral_gain',.7)*yaw*means[j,0]/.175*dx
  for t in range(N):
   for j,l in enumerate(LEGS):
    sl=slice(j*3,j*3+3);edited[t,sl],err=solve_leg(kin,l,target[t,j],q[t,sl],kin.leg_limits(l));errors.append(float(err))
  edited[-1]=edited[0];target[-1]=target[0];qs.append(edited[:,:12]);ts.append(target)
 qs=np.array(qs);ts=np.array(ts);delta=qs-qs[3:4];tdelta=ts-ts[3:4]
 maxdelta=float(abs(delta).max());maxerror=max(errors)
 if maxerror>.001 or maxdelta>.45:raise RuntimeError(f'Unsafe reference: IK {maxerror}, joint displacement {maxdelta}')
 np.savez_compressed(arm/'turn_bank.npz',yaw=yawgrid,qdelta=delta.astype('f4'),vdelta=np.gradient(delta,1/fps,axis=1).astype('f4'),tipdelta=tdelta.astype('f4'),fps=fps)
 np.savez_compressed(arm/'reference.npz',**d)
 # Static yaw previews use authored joints, retaining native root/expressive channels.
 for idx,name in [(0,'right'),(6,'left')]:
  x={k:v.copy() for k,v in d.items()};x['dof_positions'][:,:12]=qs[idx];np.savez_compressed(arm/(name+'_reference.npz'),**x)
 save(arm/'reference_audit.json',{'ik_max_error_m':maxerror,'max_joint_delta_rad':maxdelta,'yaw_grid':yawgrid.tolist(),'mean_foot_positions_m':means.tolist(),'settings':s,'zero_yaw_delta_max':float(abs(delta[3]).max()),'method':'stance body twist: vx-w*y, vy=w*x; preserve native timing and height'})
if __name__=='__main__':build(Path(sys.argv[1]))
