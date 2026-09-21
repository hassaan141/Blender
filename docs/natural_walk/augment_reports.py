from pathlib import Path #firefirfr
import numpy as np,json
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent
for folder in [W/'baseline_eval']+[p/'evaluation' for p in sorted(W.glob('attempt_*'))]:
 if not (folder/'paw_trajectories.npz').exists():continue
 d=np.load(folder/'paw_trajectories.npz');p=folder/'natural_metrics.json';m=json.loads(p.read_text());dt=float(d['dt']);valid=d['valid'];yaw=np.unwrap(Rotation.from_quat(d['root_quaternion'][:,[1,2,3,0]]).as_euler('xyz')[:,2]);root=d['root_position'];m['root_displacement_world_m']=(root[-1]-root[0]).tolist();m['heading_change_deg']=float(np.degrees(yaw[-1]-yaw[0]));m['cadence_note']='Reference loop is 2.12s and contains multiple per-leg strokes. Touchdown intervals below are geometric proxy estimates, filtering intervals under 0.20s; they are not force-confirmed steps.'
 for j,l in enumerate(['fl','fr','bl','br']):
  c=d['collision_clearance'][:,j]<.003;touch=np.flatnonzero(c[1:]&~c[:-1]&valid[1:])+1;interval=np.diff(touch)*dt;interval=interval[interval>.20]
  m['feet'][l]['median_touchdown_interval_s']=float(np.median(interval)) if len(interval) else None
 p.write_text(json.dumps(m,indent=2))
print('Added per-leg touchdown cadence and root displacement/heading audit.')
