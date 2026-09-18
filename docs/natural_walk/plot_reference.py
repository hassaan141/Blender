from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];base=np.load(R/'docs/natural_walk/baseline_trajectories.npz');d=np.load(R/'docs/natural_walk/reference_01/task_space_edit.npz');phase=base['phase']
f,axs=plt.subplots(4,3,figsize=(12,9),sharex=True)
for j,l in enumerate(['FL','FR','BL','BR']):
 for k,c in enumerate('xyz'):
  ax=axs[j,k];ax.plot(phase,d['source'][:,j,k]*1000,label='Locomotion 1');ax.plot(phase,d['target'][:,j,k]*1000,label='Natural reference 01');ax.set_ylabel(f'{l} {c} mm');ax.grid(alpha=.2)
axs[0,0].legend();f.suptitle('Conservative task-space edit: existing cycle, small swing clearance, light path smoothing');f.tight_layout();f.savefig(R/'docs/natural_walk/reference_01/trajectory_comparison.png')
