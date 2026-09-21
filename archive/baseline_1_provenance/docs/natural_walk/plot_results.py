from pathlib import Path
import argparse,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];W=R/'docs/natural_walk';p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();fig,axes=plt.subplots(4,3,figsize=(13,10),sharex=True)
for label,folder in [('Locomotion 1',W/'baseline_eval'),(a.attempt,W/a.attempt/'evaluation')]:
 d=np.load(folder/'paw_trajectories.npz');phase=d['phase'];bins=np.minimum((phase*48).astype(int),47);x=(np.arange(48)+.5)/48
 for j,l in enumerate(['FL','FR','BL','BR']):
  for k,(arr,title) in enumerate([(d['paw_local'][:,j,0]*1000,'fore-aft mm'),(d['collision_clearance'][:,j]*1000,'collision clearance mm'),((d['collision_clearance'][:,j]<.003).astype(float),'contact duty')]):
   y=[float(arr[(bins==b)&d['valid']].mean()) if np.any((bins==b)&d['valid']) else np.nan for b in range(48)]
   axes[j,k].plot(x,y,label=label);axes[j,k].set_ylabel(l+' '+title);axes[j,k].grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Measured physical paw paths and geometric contacts over the full reference cycle');fig.tight_layout();fig.savefig(W/a.attempt/'paw_comparison.png')
