"""One heading-relative residual policy and continuous, signed reference runtime.

Physics and all actuator settings are inherited unchanged. Commands are physical
m/s and rad/s; internal phase drive remains calibrated to the teacher cadence.
"""
import os,sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse
from common import TEACHERS,HOME,CASES,observation
from bingo_rl.natural_walk.env import NaturalWalkEnv
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import BingoWalkRefEnvCfg_C
class CommandEnv(NaturalWalkEnv):
 def __init__(self,cfg,render_mode=None,**kwargs):
  super().__init__(cfg,render_mode,**kwargs)
  self.commands=torch.zeros(self.num_envs,2,device=self.device);self.targets=self.commands.clone();self.ticks=0
  self.loaders={'forward':self._motion_loader}
  for key in ['backward','turning']:self.loaders[key]=type(self._motion_loader)(motion_file=str(TEACHERS[key][1]),device=self.device)
  for loader in self.loaders.values():
   assert abs(float(loader.duration)-self.motion_duration)<1e-5
   assert list(loader.dof_names)==list(self._motion_loader.dof_names)
  z=np.load(TEACHERS['turning'][1].with_name('turn_bank.npz'))
  self.bank={k:torch.tensor(z[k],device=self.device,dtype=torch.float32) for k in ['qdelta','vdelta','tipdelta']}
  refdir=Path(os.getenv('COMMAND_REFERENCE_DIR',str(HOME/'references')))
  pivot=np.load(refdir/'pivot_bank.npz');self.pivot_bank={k:torch.tensor(pivot[k],device=self.device,dtype=torch.float32) for k in ['qdelta','vdelta','tipdelta']}
  backward=np.load(refdir/'backward_turn_bank.npz');self.backward_bank={k:torch.tensor(backward[k],device=self.device,dtype=torch.float32) for k in ['qdelta','vdelta','tipdelta']}
 def _resample_command(self,ids):
  if not hasattr(self,'commands'):return super()._resample_command(ids)
  n=len(ids);c=torch.zeros(n,2,device=self.device)
  stage=min(4,int(self.ticks/max(1,self.cfg.curriculum_steps)))
  pick=torch.randint(0,3,(n,),device=self.device)
  c[:,0]=torch.where(pick==0,0.,torch.where(pick==1,.2,-.2))
  if stage>=1:
   turns=torch.rand(n,device=self.device)<.5;c[turns,1]=torch.where(torch.rand(n,device=self.device)[turns]<.5,-.4,.4)
   c[turns,0]=0 if stage==1 else .15
   if stage>=3:c[turns,0]=torch.where(torch.rand(n,device=self.device)[turns]<.5,-.15,.15)
  if stage>=4:
   core=torch.tensor([[vx,yaw] for _,vx,yaw in CASES],device=self.device)
   c=core[torch.randint(len(core),(n,),device=self.device)].clone()
   random=torch.rand(n,device=self.device)<.5;c[random,0]=torch.rand(int(random.sum()),device=self.device)*.5-.2;c[random,1]=torch.rand(int(random.sum()),device=self.device)*1.2-.6
  self.targets[ids]=c;self._cmd_timer[ids]=self._resample_s
 def _drive(self):
  vx,yaw=self.commands.unbind(-1);turn=(yaw.abs()/.4).clamp(0,1)
  positive=vx*(1.25+( .215/.15-1.25)*turn)
  negative=-(vx.abs()*(1+turn/3)).clamp(max=.20)
  drive=torch.where(vx<0,negative,positive)
  # Nonzero stepping clock for near-zero-vx turns; yaw geometry is scaled below.
  pivot=(1-vx.abs()/.1).clamp(0,1)*turn
  drive=drive*(1-pivot)+torch.where(vx<-.02,-.20,.20)*pivot
  return drive,turn,pivot
 def _sample_ref(self,times):
  if not hasattr(self,'loaders') or len(times)!=self.num_envs:return super()._sample_ref(times)
  results={};original=self._motion_loader
  for name,loader in self.loaders.items():
   self._motion_loader=loader;results[name]=super()._sample_ref(times)
  self._motion_loader=original
  drive,turn,pivot=self._drive();back=(-self.commands[:,0]/.05).clamp(0,1)
  out=[]
  for j in range(7):
   shape=(-1,)+(1,)*(results['forward'][j].ndim-1);b=back.reshape(shape);w=turn.reshape(shape)
   base=results['forward'][j]*(1-b)+results['backward'][j]*b
   # Turning posture retained in forward turns; backward retains its own geometry.
   out.append(base*(1-w*(1-b))+results['turning'][j]*w*(1-b))
  t=torch.as_tensor(times,device=self.device,dtype=torch.float32)*self._ref_fps;lo=t.long().clamp(0,self._n_ref_frames-2);hi=lo+1;tw=t-lo
  signed_yaw=self.commands[:,1]*torch.where(drive<0,-.5,1.)
  signed_yaw=signed_yaw*(1+.5*pivot)
  y=((signed_yaw+.6)/.2).clamp(0,6);yl=y.long().clamp(0,5);yh=yl+1;yw=y-yl
  for key,j in [('qdelta',0),('vdelta',1),('tipdelta',6)]:
   bank=self.bank[key];shape=(-1,)+(1,)*(bank.ndim-2);w=tw.reshape(shape);wy=yw.reshape(shape)
   low=bank[yl,lo]*(1-w)+bank[yl,hi]*w;high=bank[yh,lo]*(1-w)+bank[yh,hi]*w;delta=low*(1-wy)+high*wy
   bb=self.backward_bank[key];bl=bb[yl,lo]*(1-w)+bb[yl,hi]*w;bh=bb[yh,lo]*(1-w)+bb[yh,hi]*w
   wb=back.reshape(shape);delta=delta*(1-wb)+(bl*(1-wy)+bh*wy)*wb
   pb=self.pivot_bank[key];pl=pb[yl,lo]*(1-w)+pb[yl,hi]*w;ph=pb[yh,lo]*(1-w)+pb[yh,hi]*w
   wp=pivot.reshape(shape);delta=delta*(1-wp)+(pl*(1-wy)+ph*wy)*wp
   if j==1:delta*= (1.4*drive/.66)[:,None]
   out[j]=out[j]+delta
  return tuple(out)
 def next_ff(self):
  drive,_,_=self._drive();times=torch.remainder(self._clip_time+self.step_dt*1.4*drive/.66,self.motion_duration)
  q=self._sample_ref(times.detach().cpu().numpy())[0];w=(drive.abs()/.198).clamp(0,1)[:,None]
  return w*q+(1-w)*self._stand_pose
 def _get_observations(self):
  if not hasattr(self,'commands'):
   base=super()._get_observations()['policy'];return {'policy':torch.cat([base[:,:69],torch.zeros(self.num_envs,26,device=self.device)],-1)}
  return {'policy':observation(self,self.commands,self.next_ff())}
 def _pre_physics_step(self,actions):
  self.ticks+=1;self._cmd_timer-=self.step_dt;ids=(self._cmd_timer<=0).nonzero().flatten()
  if len(ids):self._resample_command(ids)
  if hasattr(self,'held_commands'):self.targets[:]=self.held_commands
  self.commands.lerp_(self.targets,self.cfg.command_alpha)
  drive,_,_=self._drive();self._cmd_vx[:]=drive
  self._prev_actions=getattr(self,'actions',torch.zeros_like(actions));self.actions=actions.clone();self._prev_filtered_action=self._filtered_action.clone()
  self._filtered_action=self._residual_ema_alpha*actions+(1-self._residual_ema_alpha)*self._prev_filtered_action
  self._ff_leg_q=self.next_ff();self._clip_time=torch.remainder(self._clip_time+self.step_dt*1.4*drive/.66,self.motion_duration)
 def _get_rewards(self):
  # Inherited gait/smoothness terms, but velocity reward uses physical command.
  saved=self._cmd_vx.clone();self._cmd_vx[:]=self.commands[:,0];reward=super()._get_rewards();self._cmd_vx[:]=saved
  d=self.robot.data;q=d.body_quat_w[:,self.ref_body_index];vel=quat_rotate_inverse(q,d.body_lin_vel_w[:,self.ref_body_index]);ang=quat_rotate_inverse(q,d.body_ang_vel_w[:,self.ref_body_index])
  reward+=self.cfg.yaw_weight*torch.exp(-((ang[:,2]-self.commands[:,1])/.3)**2)
  reward-=.05*(vel[:,1]/.2)**2
  return reward
@configclass
class CommandCfg(BingoWalkRefEnvCfg_C):
 motion_file:str=str(TEACHERS['forward'][1])
 ear_file:str=str(TEACHERS['forward'][1])
 observation_space=95
 correct_bl_contact=True
 vx_range=(.25,.25)
 stand_prob=0.
 command_resample_s=4.
 command_alpha:float=.10
 curriculum_steps:int=1920
 yaw_weight:float=.3
for name in ['Train','Play']:
 gym.register(id=f'Bingo-Command-{name}-v0',entry_point=CommandEnv,disable_env_checker=True,kwargs={'env_cfg_entry_point':CommandCfg,'skrl_cfg_entry_point':'bingo_rl.track.agents:skrl_ppo_cfg.yaml'})
