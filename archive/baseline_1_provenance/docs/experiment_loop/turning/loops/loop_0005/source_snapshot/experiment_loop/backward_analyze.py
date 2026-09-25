"""Reference audit and signed-motion diagnostics. Isaac Python, no simulator import."""
import argparse,sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import ROOT,read,save
p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','evaluation']);p.add_argument('arm',type=Path);a=p.parse_args();arm=a.arm
if a.stage=='reference':
    src=np.load(ROOT/'docs/natural_walk/reference_02b/reference.npz');d={k:src[k].copy() for k in src.files};rate=1.4*(-.2)/.66
    velocity_keys=[k for k in d if 'velocit' in k]
    for k in velocity_keys:d[k]=d[k]*rate
    np.savez_compressed(arm/'reference.npz',**d)
    sys.path.insert(0,str(ROOT/'stage2'));from v4_kinematics import V4Kin,LEGS
    kin=V4Kin(ROOT/'URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf')
    q=d['dof_positions'];tips=np.array([[kin.leg_fk(l,row[3*j:3*j+3])[0] for j,l in enumerate(LEGS)] for row in q]);dt=1/float(d['fps']);dx=np.gradient(tips[:,:,0],dt,axis=0)
    report={'method':'Reverse time via existing signed phase; reverse/scale all stored velocity channels',
            'signed_phase_rate':rate,'cycle_period_s':(len(q)-1)*dt/abs(rate),'velocity_keys_changed':velocity_keys,
            'positions_contacts_unchanged':all(np.array_equal(d[k],src[k]) for k in src.files if k not in velocity_keys),
            'feet':{l:{'contact_duty':float(d['contacts'][:,j].mean()),'stance_local_vx_forward_m_s':float(dx[d['contacts'][:,j].astype(bool),j].mean()),
                      'stance_local_vx_backward_m_s':float((dx*rate)[d['contacts'][:,j].astype(bool),j].mean()),
                      'paw_range_mm':(np.ptp(tips[:,j],axis=0)*1000).tolist()} for j,l in enumerate(LEGS)}}
    save(arm/'reference_audit.json',report);print(json.dumps(report,indent=2))
else:
    out=arm/'evaluation';m=read(out/'metrics.json');n=read(out/'natural_metrics.json');d=np.load(out/'paw_trajectories.npz');dt=float(d['dt']);valid=d['valid']
    root=d['root_position'][valid];quat=d['root_quaternion'][valid];yaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    trace=np.load(out/'diagnostic_trace.npz');print('trace keys',trace.files)
    vx=trace['vx'].reshape(-1)[valid] if 'vx' in trace.files else None
    if vx is None:
        delta=np.diff(root,axis=0)/dt;local=Rotation.from_quat(quat[1:,[1,2,3,0]]).inv().apply(delta);vx=local[:,0]
        m['vx_extrema_source']='finite-difference root trajectory (24Hz)'
    m['vx_min']=float(vx.min());m['vx_abs_max']=float(np.abs(vx).max())
    n['root_displacement_world_m']=(root[-1]-root[0]).tolist();n['heading_change_deg']=float(np.degrees(yaw[-1]-yaw[0]));n['yaw_rate_rms_rad_s']=float(np.sqrt(np.mean((np.diff(yaw)/dt)**2)))
    for j,l in enumerate(['fl','fr','bl','br']):
        contact=d['collision_clearance'][:,j]<.003;touch=np.flatnonzero(contact[1:]&~contact[:-1]&valid[1:])+1;interval=np.diff(touch)*dt;interval=interval[interval>.20]
        n['feet'][l]['median_touchdown_interval_s']=float(np.median(interval)) if len(interval) else None
    n['cadence_note']='Multiple strokes per reference cycle; touchdown intervals from collision-height proxy, not contact force.'
    save(out/'metrics.json',m);save(out/'natural_metrics.json',n)
    baseline=read(ROOT/'docs/natural_walk/attempt_03/evaluation/metrics.json');bn=read(ROOT/'docs/natural_walk/attempt_03/evaluation/natural_metrics.json')
    slips=[v['contact_slip_rms_m_s'] for v in n['feet'].values()];bslips=[v['contact_slip_rms_m_s'] for v in bn['feet'].values()]
    sat=np.mean(list(m['torque_saturation_pct_per_joint'].values()));bsat=np.mean(list(baseline['torque_saturation_pct_per_joint'].values()))
    failures=[]
    if np.mean(slips)>np.mean(bslips)*1.1:failures.append('mean slip exceeds seed +10%')
    if sat>bsat*1.2:failures.append('mean torque saturation exceeds seed +20%')
    for l,v in n['feet'].items():
        if v['duty']>=.85 or v['clearance_p95_mm']<5:failures.append(l+' insufficient sustained clearance / high contact duty')
    for j,v in m['torque_saturation_pct_per_joint'].items():
        if v>max(baseline['torque_saturation_pct_per_joint'][j]*1.5,baseline['torque_saturation_pct_per_joint'][j]+5):failures.append(j+' major saturation regression')
    if abs(n['heading_change_deg'])>10 or n['yaw_rate_rms_rad_s']>.15:failures.append('yaw drift/oscillation')
    if read(out/'quality_metrics.json')['warmup_falls']>0:failures.append('warmup fall')
    save(arm/'backward_gates.json',{'failures':failures,'mean_slip_m_s':float(np.mean(slips)),'seed_mean_slip_m_s':float(np.mean(bslips)),
         'mean_torque_saturation_pct':float(sat),'seed_mean_torque_saturation_pct':float(bsat),'natural_metrics':n})
