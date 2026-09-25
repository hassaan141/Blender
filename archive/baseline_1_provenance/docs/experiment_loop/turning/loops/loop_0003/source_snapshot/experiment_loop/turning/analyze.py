import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(sys.argv[1]);m=json.loads((p/'metrics.json').read_text());n=json.loads((p/'natural_metrics.json').read_text());z=np.load(p/'paw_trajectories.npz');valid=z['valid'];q=z['root_quaternion'][valid];dt=float(z['dt']);yaw=np.unwrap(Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz')[:,2]);rate=np.diff(yaw)/dt;cmd=m['command']['yaw'];trace=np.load(p/'diagnostic_trace.npz');vx=trace['vx'].reshape(-1)[valid]
m.update(yaw_mean=float(rate.mean()),yaw_std=float(rate.std()),yaw_rmse=float(np.sqrt(np.mean((rate-cmd)**2))),heading_change_deg=float(np.degrees(yaw[-1]-yaw[0])),vx_abs_max=float(abs(vx).max()),mean_slip_m_s=float(np.mean([v['contact_slip_rms_m_s'] for v in n['feet'].values()])),mean_saturation_pct=float(np.mean(list(m['torque_saturation_pct_per_joint'].values()))))
(p/'metrics.json').write_text(json.dumps(m,indent=2))
