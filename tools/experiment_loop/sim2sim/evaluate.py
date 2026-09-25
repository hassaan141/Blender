#!/usr/bin/env python3
"""Evaluate BASELINE_1-compatible policy in the existing MuJoCo MJCF."""
import argparse,csv,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np, torch
from mujoco_env import BingoMujoco,CASES,ROOT
from train_ppo import Actor,normalizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=10);p.add_argument('--settle',type=float,default=1);p.add_argument('--video',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--expression',choices=['neutral','reference'],default='neutral',help='neutral = browser ExpressionController (gate); reference = attempts 01-03 adapter');p.add_argument('--start',choices=['stand_then_command','command_first'],default='stand_then_command',help='stand_then_command = browser probe: settle at [0,0], then command');p.add_argument('--init-noise',type=float,default=0.,help='std (rad) of Gaussian noise on the 21 joint positions at reset');p.add_argument('--seed',type=int,default=20260923);p.add_argument('--expr-obs',choices=['training_mean','live'],default='training_mean');a=p.parse_args()
 path=Path(a.checkpoint);c=torch.load(path,map_location='cpu',weights_only=False);dev=torch.device(a.device);actor=Actor().to(dev);actor.load_state_dict(c['policy']);actor.eval();stats=c['observation_preprocessor'];out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 summary={}; steps=int((a.seconds+a.settle)*24); renderer=None  # one GL context for all cases: re-creating it fails under EGL
 for ci,(name,vx,yaw) in enumerate(CASES):
  env=BingoMujoco(a.seed+ci,expression=a.expression,expr_obs=a.expr_obs);obs=env.reset((vx,yaw) if a.start=='command_first' else (0.,0.))
  if a.init_noise>0:
   import mujoco
   for n in env.qa: env.data.qpos[env.qa[n]]+=env.rng.normal(0,a.init_noise)
   mujoco.mj_forward(env.model,env.data);obs=env.observe()
  histories=[];writer=None;camera=None
  if a.video:
   import mujoco
   renderer=renderer or mujoco.Renderer(env.model,height=480,width=640);  # every case loads the same MJCF
   camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[0,0,.16];camera.distance=.75;camera.azimuth=135;camera.elevation=-12
   scene_option=mujoco.MjvOption();scene_option.geomgroup[:]=1  # include the browser MJCF's group-3 collision meshes
   video=out/f'{name}.mp4';writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','12','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
  for t in range(steps):
   if t==int(a.settle*24): env.target=np.array([vx,yaw],dtype=np.float64)
   with torch.no_grad():
    x=torch.as_tensor(obs,device=dev).unsqueeze(0);action=actor(normalizer(x,stats))[0].cpu().numpy()
   obs,rew,done,info=env.step(action)
   if t>=int(a.settle*24): histories.append(info)
   if a.video and (t%2==0):
    camera.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.16]
    renderer.update_scene(env.data,camera=camera,scene_option=scene_option);writer.stdin.write(renderer.render().tobytes())
   if done: break
  if a.video:
   writer.stdin.close();rc=writer.wait()
   if rc: raise RuntimeError(f'ffmpeg failed for {name}')
  h=histories or [{}]; vxv=np.array([x.get('vx',0) for x in h]); yawv=np.array([x.get('yaw_rate',0) for x in h]); acc=np.array([x.get('accel',np.zeros(12)) for x in h]); rate=np.array([x.get('action_rate',np.zeros(12)) for x in h]);torque=np.array([x.get('torque',np.zeros(12)) for x in h]); slip=[s for x in h for s in x.get('slip',[])]; q=np.array([x.get('joint_pos',np.zeros(12)) for x in h]);
  limits=np.array([[-.42, .42],[-1.56,1.56],[-1.56,1.56]]*4); viol=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(min=0)
  result={'command':[vx,yaw],'survival':not env.fallen,'survival_seconds':env.elapsed,'fall_status':bool(env.fallen),'distance_m':float(env.data.qpos[0]),'vx_mean':float(vxv.mean()),'vx_std':float(vxv.std()),'vx_max_abs':float(np.max(np.abs(vxv))),'yaw_rate_mean':float(yawv.mean()),'yaw_error_mean_abs':float(np.mean(np.abs(yawv-yaw))),'joint_acceleration_rms':float(np.sqrt(np.mean(acc**2))),'action_rate_rms':float(np.sqrt(np.mean(rate**2))),'torque_saturation_pct':float(np.mean(torque>=2.97)*100),'slip_mean_mps':float(np.mean(slip)) if slip else 0.,'joint_limit_violations':int(np.count_nonzero(viol>1e-6)),'joint_limit_max_violation_rad':float(viol.max())}
  summary[name]=result;(out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n');(out/f'{name}.txt').write_text('\n'.join(f'{k}: {v}' for k,v in result.items())+'\n');print(f'{name}: survive={result["survival"]} vx={result["vx_mean"]:.3f} yaw_err={result["yaw_error_mean_abs"]:.3f} fall={result["fall_status"]}',flush=True)
 (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (out/'metrics.csv').open('w',newline='') as f:
  fields=['command','vx','yaw','survival','survival_seconds','fall_status','distance_m','vx_mean','vx_std','vx_max_abs','yaw_rate_mean','yaw_error_mean_abs','slip_mean_mps','joint_acceleration_rms','action_rate_rms','torque_saturation_pct','joint_limit_violations','joint_limit_max_violation_rad'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for name,met in summary.items():w.writerow({'command':name,'vx':met['command'][0],'yaw':met['command'][1],**{k:v for k,v in met.items() if k!='command'}})
 (out/'evaluation.txt').write_text(f'checkpoint: {path}\nsha256: {hashlib.sha256(path.read_bytes()).hexdigest()}\nmeasured_seconds: {a.seconds}\nsettle_seconds: {a.settle}\nexpression: {a.expression}\nstart: {a.start}\ninit_noise: {a.init_noise}\nexpr_obs: {a.expr_obs}\nseed: {a.seed}\nvideo_requested: {a.video}\nall_survive: {all(m["survival"] for m in summary.values())}\n')
if __name__=='__main__':main()
