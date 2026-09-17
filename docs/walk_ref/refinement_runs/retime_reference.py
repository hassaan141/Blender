"""Redistribute reference timing, preserving cycle duration and ordered poses.

Writes a derived asset; never edits a canonical source. One tuning parameter:
strength. Slow authored phases with large front-left SP/knee acceleration.
"""
import argparse,hashlib,json,pathlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--strength',type=float,required=True);a=p.parse_args()
source=pathlib.Path(a.source);out=pathlib.Path(a.out);assert source.resolve()!=out.resolve();assert 0<a.strength<1
before=hashlib.sha256(source.read_bytes()).hexdigest();z=np.load(source,allow_pickle=True);d={k:z[k].copy() for k in z.files}
fps=float(d['fps']);q=d['dof_positions'].astype(float);n=len(q);idx=[list(d['dof_names']).index(x) for x in ('fl_SP_J','fl_knee')]
acc=np.gradient(np.gradient(q[:,idx],1/fps,axis=0),1/fps,axis=0)
demand=gaussian_filter1d(np.sqrt(np.mean(acc**2,axis=1)),sigma=8,mode='wrap')
density=1+a.strength*(demand/demand.mean()-1)
clock=np.r_[0,np.cumsum((density[:-1]+density[1:])/2)];clock*=float(n-1)/clock[-1]
source_frame=np.interp(np.arange(n),clock,np.arange(n));assert np.all(np.diff(source_frame)>0)
lo=np.floor(source_frame).astype(int);hi=np.minimum(lo+1,n-1);frac=source_frame-lo
for k,v in list(d.items()):
 if v.ndim==0 or len(v)!=n:continue
 if v.dtype==bool:d[k]=v[np.rint(source_frame).astype(int)];continue
 w=frac.reshape((n,)+(1,)*(v.ndim-1))
 if k=='body_rotations':
  upper=v[hi].copy();sign=np.sum(v[lo]*upper,axis=-1,keepdims=True)<0;upper=np.where(sign,-upper,upper);result=(1-w)*v[lo]+w*upper;result/=np.linalg.norm(result,axis=-1,keepdims=True)
 else:result=(1-w)*v[lo]+w*v[hi]
 d[k]=result.astype(v.dtype)
d['dof_velocities']=np.gradient(d['dof_positions'],1/fps,axis=0).astype(np.float32)
d['body_linear_velocities']=np.gradient(d['body_positions'],1/fps,axis=0).astype(np.float32)
d['body_angular_velocities']*=np.gradient(source_frame).reshape(n,1,1)
d['refinement_timing_strength']=np.array(a.strength);d['refinement_source_frame']=source_frame
assert d['dof_positions'].shape==q.shape and float(d['fps'])==fps
assert np.allclose(d['dof_positions'][[0,-1]],q[[0,-1]])
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,**d)
(out.with_suffix('.json')).write_text(json.dumps({'source':str(source.resolve()),'source_sha256':before,'strength':a.strength,'frames':n,'fps':fps,'duration_unchanged':True,'phase_order_preserved':True,'source_frame_advance_min':float(np.diff(source_frame).min()),'source_frame_advance_max':float(np.diff(source_frame).max()),'poses':'linear samples of original ordered trajectory; no pose offsets or amplitudes changed'},indent=2))
