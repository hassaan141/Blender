"""Collect real native-teacher transitions, never load a teacher into a student."""
import argparse,os,sys,json
from pathlib import Path
from common import ROOT,HOME,TEACHERS
p=argparse.ArgumentParser();p.add_argument('--teacher',choices=TEACHERS,required=True);p.add_argument('--steps',type=int,default=1800);p.add_argument('--num_envs',type=int,default=64)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();os.environ['BINGO_NATURAL_REFERENCE']=str(TEACHERS[a.teacher][1]);os.environ['BINGO_NATURAL_CONTACT_FIX']='1'
app=AppLauncher(a).app
import torch,numpy as np
sys.path.insert(0,str(ROOT/'rl/bingo_rl'))
from common import load_policy,normalized,observation
from bingo_rl.natural_walk.env import NaturalWalkEnv
from bingo_rl.natural_walk import NaturalTrainCfg
if a.teacher=='turning':
 sys.path.insert(0,str(ROOT/'rl/bingo_rl/experiment_loop/turning'))
 from env import TurningEnv,TrainCfg
 cfg=TrainCfg();cfg.turn_heading_local=1;cfg.turn_bl_recovery=1;cls=TurningEnv
else:cfg=NaturalTrainCfg();cls=NaturalWalkEnv
cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=42;cfg.episode_length_s=30;cfg.command_resample_s=1000
if a.teacher=='forward':cfg.vx_range=(.25,.25)
elif a.teacher=='backward':cfg.vx_range=(-.2,-.2)
else:cfg.vx_range=(.215,.215)
env=cls(cfg);env.reset();policy,checkpoint=load_policy(TEACHERS[a.teacher][0],env.device)
commands=torch.zeros(a.num_envs,2,device=env.device);commands[:,0]=.2 if a.teacher=='forward' else -.2 if a.teacher=='backward' else .15
# The forward teacher also supplies standing demonstrations, with its original runtime.
if a.teacher=='forward':commands[::4,0]=0
if a.teacher=='turning':commands[:,1]=torch.linspace(-.6,.6,a.num_envs,device=env.device)
xs=[];ys=[];falls=0
with torch.inference_mode():
 for t in range(a.steps):
  env._cmd_timer.fill_(1000);env._cmd_vx[:]=torch.where(commands[:,0]==0,0.,.25) if a.teacher=='forward' else (-.2 if a.teacher=='backward' else .215)
  if a.teacher=='turning':env.held_yaw=commands[:,1]
  if a.teacher=='turning':env._cmd_yaw[:]=commands[:,1]
  obs=env._get_observations()['policy'];action=policy(normalized(obs,checkpoint))
  nexttime=torch.remainder(env._clip_time+env.step_dt*1.4*env._cmd_vx/.66,env.motion_duration)
  q=env._sample_ref(nexttime.cpu().numpy())[0];w=(env._cmd_vx.abs()/.198).clamp(0,1)[:,None];ff=w*q+(1-w)*env._stand_pose
  x=observation(env,commands,ff)
  if t>48:xs.append(x.cpu());ys.append(action.cpu())
  _,_,terminated,_,_=env.step(action);falls+=int(terminated.sum())
  if t%300==0:print(f'{a.teacher}: {t}/{a.steps}, falls={falls}',flush=True)
HOME.joinpath('teachers').mkdir(parents=True,exist_ok=True)
torch.save({'x':torch.cat(xs),'y':torch.cat(ys),'teacher':a.teacher,'falls':falls,'reference':str(TEACHERS[a.teacher][1])},HOME/'teachers'/f'{a.teacher}.pt')
print('COLLECTION COMPLETE',a.teacher,torch.cat(xs).shape,'falls',falls,flush=True)
env.close();app.close()
