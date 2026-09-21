"""Derive command tracking and contact-cycle diagnostics from saved trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);a=p.parse_args();d=np.load(a.evaluation/'trajectories.npz');m=json.loads((a.evaluation/'metrics.json').read_text());dt=1/24
for i,(name,row) in enumerate(m.items()):
 valid=d['alive'][:,i];vx=d['velocity'][:,i,0]-d['commands'][:,i,0];yaw=d['yaw'][:,i]-d['commands'][:,i,1]
 row['vx_tracking_rmse']=float(np.sqrt(np.mean(vx[valid]**2))) if valid.any() else None
 row['yaw_tracking_rmse']=float(np.sqrt(np.mean(yaw[valid]**2))) if valid.any() else None
 settled=valid.copy()
 if name=='transitions':settled&=((np.arange(len(valid))*dt+1)%5>=1)
 row['settled_vx_tracking_rmse']=float(np.sqrt(np.mean(vx[settled]**2))) if settled.any() else None
 row['settled_yaw_tracking_rmse']=float(np.sqrt(np.mean(yaw[settled]**2))) if settled.any() else None
 cadence=[]
 for j in range(4):
  c=d['contact'][:,i,j];events=sum(bool(valid[t] and c[t-2:t].all() and (~c[t:t+2]).all()) for t in range(2,len(c)-1));cadence.append(float(events/max(valid.sum()*dt,dt)))
 row['swing_onsets_per_second_per_foot']=cadence
(a.evaluation/'metrics_extended.json').write_text(json.dumps(m,indent=2)+'\n')
