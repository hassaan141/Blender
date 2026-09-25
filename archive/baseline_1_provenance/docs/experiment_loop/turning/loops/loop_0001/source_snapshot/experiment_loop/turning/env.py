"""One yaw command and task-space reference adaptation on the native controller."""
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse
from bingo_rl.walk_ref.bingo_walk_ref_env import BingoWalkRefEnv
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import BingoWalkRefEnvCfg_C,BingoWalkRefPlayEnvCfg
REFERENCE=os.environ['BINGO_NATURAL_REFERENCE']
class TurningEnv(BingoWalkRefEnv):
 def __init__(self,cfg,render_mode=None,**kw):
  super().__init__(cfg,render_mode,**kw)
  self._cmd_yaw=torch.zeros(self.num_envs,device=self.device);self._target_yaw=self._cmd_yaw.clone()
  bank=np.load(Path(REFERENCE).with_name('turn_bank.npz'))
  self._turn_bank={k:torch.tensor(bank[k],dtype=torch.float32,device=self.device) for k in ['qdelta','vdelta','tipdelta']}
  self._prev_turn_rate=torch.zeros(self.num_envs,device=self.device)
 def _resample_command(self,ids):
  super()._resample_command(ids)
  if hasattr(self,'_target_yaw'):
   self._target_yaw[ids]=torch.rand(len(ids),device=self.device)*1.2-.6
 def _pre_physics_step(self,actions):
  if hasattr(self,'held_yaw'):self._target_yaw[:]=self.held_yaw;self._cmd_yaw[:]=self.held_yaw
  else:self._cmd_yaw.lerp_(self._target_yaw,.1)
  super()._pre_physics_step(actions)
 def _get_observations(self):
  o=super()._get_observations();yaw=getattr(self,'_cmd_yaw',torch.zeros(self.num_envs,device=self.device));o['policy']=torch.cat([o['policy'],yaw[:,None]/.6],dim=-1);return o
 def _sample_ref(self,times):
  result=list(super()._sample_ref(times))
  if not hasattr(self,'_turn_bank') or len(times)!=self.num_envs:return tuple(result)
  t=torch.as_tensor(times,device=self.device,dtype=torch.float32)*self._ref_fps;lo=t.long().clamp(0,self._n_ref_frames-2);hi=lo+1;tw=t-lo
  y=((self._cmd_yaw+.6)/.2).clamp(0,6);yl=y.long().clamp(0,5);yh=yl+1;yw=y-yl
  for key,outidx in [('qdelta',0),('vdelta',1),('tipdelta',6)]:
   b=self._turn_bank[key];shape=(-1,)+(1,)*(b.ndim-2);w=tw.reshape(shape);wy=yw.reshape(shape)
   low=b[yl,lo]*(1-w)+b[yl,hi]*w;high=b[yh,lo]*(1-w)+b[yh,hi]*w;delta=low*(1-wy)+high*wy
   if key=='vdelta':delta*= (1.4*self._cmd_vx/.66)[:,None]
   result[outidx]=result[outidx]+delta
  return tuple(result)
 def _get_rewards(self):
  r=super()._get_rewards();d=self.robot.data;quat=d.body_quat_w[:,self.ref_body_index]
  ang=quat_rotate_inverse(quat,d.body_ang_vel_w[:,self.ref_body_index]);yaw=ang[:,2]
  r=r+self.cfg.turn_yaw_weight*torch.exp(-((yaw-self._cmd_yaw)/.3)**2)
  r=r-self.cfg.turn_yaw_smooth_weight*((yaw-self._prev_turn_rate)/self.step_dt/5.)**2
  self._prev_turn_rate=yaw.detach().clone();return r
@configclass
class TrainCfg(BingoWalkRefEnvCfg_C):
 motion_file:str=REFERENCE
 ear_file:str=REFERENCE
 observation_space=71
 vx_range=(.15,.20)
 stand_prob=0.
 command_resample_s=4.
 turn_yaw_weight:float=.3
 turn_yaw_smooth_weight:float=0.
@configclass
class PlayCfg(BingoWalkRefPlayEnvCfg):
 motion_file:str=REFERENCE
 ear_file:str=REFERENCE
 observation_space=71
 vx_range=(.175,.175)
 stand_prob=0.
 command_resample_s=1000.
 turn_yaw_weight:float=.3
 turn_yaw_smooth_weight:float=0.
for name,cfg in [('Train',TrainCfg),('Play',PlayCfg)]:
 gym.register(id=f'Bingo-Turning-{name}-v0',entry_point=TurningEnv,disable_env_checker=True,kwargs={'env_cfg_entry_point':cfg,'skrl_cfg_entry_point':'bingo_rl.track.agents:skrl_ppo_cfg.yaml'})
