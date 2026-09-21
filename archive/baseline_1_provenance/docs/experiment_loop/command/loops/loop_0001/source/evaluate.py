"""Parallel 60-second command battery plus per-command close-up videos."""
import argparse,sys,json
from pathlib import Path
from common import ROOT,CASES,load_policy,normalized
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--duration',type=float,default=60);p.add_argument('--video',action='store_true');p.add_argument('--video_seconds',type=float,default=12)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=a.video;app=AppLauncher(a).app
import torch,numpy as np,subprocess
sys.path.insert(0,str(ROOT/'rl/bingo_rl'))
from env import CommandEnv,CommandCfg
from isaaclab.utils.math import quat_rotate_inverse,quat_apply,euler_xyz_from_quat
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=CommandCfg();cfg.scene.num_envs=10;cfg.scene.env_spacing=5.;cfg.sim.device=a.device;cfg.random_start_frame=False;cfg.episode_length_s=a.duration+10;cfg.seed=42;cfg.viewer.resolution=(1280,720);cfg.viewer.origin_type="world"
env=CommandEnv(cfg,render_mode='rgb_array' if a.video else None);env.reset();env.held_commands=torch.tensor([[vx,yaw] for _,vx,yaw in CASES]+[[0,0]],device=env.device);env.targets[:]=env.held_commands;env.commands[:]=env.targets
policy,c=load_policy(a.checkpoint,env.device)
if a.video:
 import time
 for _ in range(10):env.render();time.sleep(.1)
 env.reset();env.commands[:]=env.held_commands;env.targets[:]=env.held_commands
writers={};names=[n for n,_,_ in CASES]+['transitions'];history=[];alive=torch.ones(10,dtype=torch.bool,device=env.device);prev_q=None;prev_tips=None;prev_actions=None
hulls=np.load(ROOT/'stage4/out/collision_hulls.npz');hulls=[torch.tensor(hulls[x+'_knee'],device=env.device,dtype=torch.float32) for x in ['fl','fr','bl','br']]
with torch.inference_mode():
 for step in range(int(a.duration/env.step_dt)):
  sec=step*env.step_dt;trans=int(sec/5)%len(CASES);env.held_commands[-1]=torch.tensor(CASES[trans][1:],device=env.device)
  obs=env._get_observations()['policy'];action=policy(normalized(obs,c));_,_,done,_,_=env.step(action)
  d=env.robot.data;idx=env.ref_body_index;q=d.body_quat_w[:,idx];v=quat_rotate_inverse(q,d.body_lin_vel_w[:,idx]);av=quat_rotate_inverse(q,d.body_ang_vel_w[:,idx]);roll,pitch,_=euler_xyz_from_quat(q);roll=torch.atan2(roll.sin(),roll.cos());pitch=torch.atan2(pitch.sin(),pitch.cos())
  qdot=d.joint_vel[:,env.robot_ctrl_indexes];torque=d.applied_torque[:,env.robot_ctrl_indexes];jpos=d.joint_pos[:,env.robot_ctrl_indexes];tips,_=env._foot_tips_local();contact=[]
  for j,h in enumerate(hulls):
   bi=env.key_body_indexes[j];n=len(h);verts=quat_apply(d.body_quat_w[:,bi,None,:].expand(-1,n,-1).reshape(-1,4),h[None].expand(10,-1,-1).reshape(-1,3)).reshape(10,n,3)
   low=verts[:,:,2].min(1).values+d.body_pos_w[:,bi,2]-env.scene.env_origins[:,2];contact.append(low<.003)
  contact=torch.stack(contact,1);acc=torch.zeros_like(qdot) if prev_q is None else (qdot-prev_q)/env.step_dt;speed=torch.zeros(10,4,device=env.device) if prev_tips is None else ((tips-prev_tips)/env.step_dt)[:,:,:2].norm(dim=-1);rate=torch.zeros_like(action) if prev_actions is None else (env._filtered_action-prev_actions)/env.step_dt
  if sec>=1:
   history.append({k:t.detach().cpu().numpy() for k,t in {'velocity':v,'yaw':av[:,2],'roll':roll,'pitch':pitch,'height':d.body_pos_w[:,idx,2]-env.scene.env_origins[:,2],'acc':acc,'torque':torque,'jpos':jpos,'contact':contact,'slip':speed,'action_rate':rate,'alive':alive&~done,'commands':env.commands}.items()})
  alive &= ~done;prev_q=qdot.clone();prev_tips=tips.clone();prev_actions=env._filtered_action.clone()
  if a.video and 1<=sec and step%2==0:
   for i,name in enumerate(names):
    if i<9 and sec>=1+a.video_seconds:continue
    pos=d.body_pos_w[i,idx].cpu().numpy();env.sim.set_camera_view(eye=pos+np.array([.75,.85,.42]),target=pos)
    env.sim.render();frame=env.render();h,w=frame.shape[:2]
    if name=='stand' and step==24:
     assert frame.std()>2, 'Black/empty camera capture'
     from PIL import Image
     Image.fromarray(frame).save(out/'camera_check.png')
    if name not in writers:
     writers[name]=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(1/env.step_dt/2),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p',str(out/(name+'.mp4'))],stdin=subprocess.PIPE)
    writers[name].stdin.write(np.ascontiguousarray(frame[:,:,:3]).tobytes())
  if step%240==0:print('eval seconds',round(sec,1),'survivors',alive.cpu().tolist(),flush=True)
for writer in writers.values():writer.stdin.close();assert writer.wait()==0
hist={k:np.stack([row[k] for row in history]) for k in history[0]};np.savez_compressed(out/'trajectories.npz',**hist)
limits=env.robot.data.soft_joint_pos_limits[0,env.robot_ctrl_indexes].cpu().numpy();metrics={}
for i,name in enumerate(names):
 valid=hist['alive'][:,i];mask=valid if valid.any() else np.ones_like(valid);v=hist['velocity'][mask,i];contact=hist['contact'][mask,i];sat=(abs(hist['torque'][mask,i])>=2.97).mean(0)*100;j=hist['jpos'][mask,i]
 metrics[name]={'command':CASES[i][1:] if i<9 else '5-second command transitions','survival':bool(alive[i]),'valid_seconds':float(valid.sum()*env.step_dt),'vx_mean':float(v[:,0].mean()),'vx_std':float(v[:,0].std()),'yaw_mean':float(hist['yaw'][mask,i].mean()),'yaw_std':float(hist['yaw'][mask,i].std()),'roll_rms_deg':float(np.sqrt(np.mean(hist['roll'][mask,i]**2))*180/np.pi),'pitch_rms_deg':float(np.sqrt(np.mean(hist['pitch'][mask,i]**2))*180/np.pi),'height_mean':float(hist['height'][mask,i].mean()),'height_std':float(hist['height'][mask,i].std()),'joint_acceleration_rms':float(np.sqrt(np.mean(hist['acc'][mask,i]**2))),'action_rate_rms':float(np.sqrt(np.mean(hist['action_rate'][mask,i]**2))),'torque_saturation_max_pct':float(sat.max()),'torque_saturation_per_joint_pct':sat.tolist(),'contact_duty':contact.mean(0).tolist(),'slip_mean':float(hist['slip'][mask,i][contact].mean()) if contact.any() else None,'joint_limit_violation_rad':float(max(0,(limits[:,0]-j).max(),(j-limits[:,1]).max()))}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2),flush=True);env.close();app.close()
