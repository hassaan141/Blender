"""Shared explicit student interface and immutable teacher locations."""
from pathlib import Path
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'rl/bingo_rl/bingo_rl').is_dir())
HOME=ROOT/'docs/experiment_loop/command'
TEACHERS={name:(None,Path(__file__).parent/'references'/name/'reference.npz') for name in ['forward','backward','turning']}
CASES=[('stand',0.,0.),('forward',.2,0.),('backward',-.2,0.),('left',0.,.4),('right',0.,-.4),('forward_left',.15,.4),('forward_right',.15,-.4),('backward_left',-.15,.4),('backward_right',-.15,-.4)]
class Policy(torch.nn.Module):
 def __init__(self,n=95,out=12):
  super().__init__();self.net_container=torch.nn.Sequential(torch.nn.Linear(n,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,out))
  if out==12:self.log_std_parameter=torch.nn.Parameter(torch.full((12,),-2.3))
 def forward(self,x):return self.net_container(x)
def load_policy(path,device):
 c=torch.load(path,map_location=device,weights_only=False);p=Policy(c['policy']['net_container.0.weight'].shape[1]).to(device);p.load_state_dict(c['policy']);p.eval();return p,c

def normalized(x,c):
 s=c['observation_preprocessor'];return ((x-s['running_mean'].float())/(s['running_variance'].float().sqrt()+1e-8)).clamp(-5,5)

def proprio(env,local=True):
 from isaaclab.utils.math import quat_mul,quat_apply
 from bingo_rl.amp.bingo_amp_env import compute_obs
 d=env.robot.data;q=d.body_quat_w[:,env.ref_body_index];root=d.body_pos_w[:,env.ref_body_index];tips,_=env._foot_tips_local();lv=d.body_lin_vel_w[:,env.ref_body_index];av=d.body_ang_vel_w[:,env.ref_body_index]
 if local:
  angle=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));inv=torch.zeros_like(q);inv[:,0]=torch.cos(angle/2);inv[:,3]=-torch.sin(angle/2)
  tips=quat_apply(inv[:,None,:].expand(-1,4,-1).reshape(-1,4),(tips-root[:,None,:]).reshape(-1,3)).reshape(-1,4,3)+root[:,None,:]
  q=quat_mul(inv,q);lv=quat_apply(inv,lv);av=quat_apply(inv,av)
 return compute_obs(d.joint_pos[:,env.obs_dof_indexes],d.joint_vel[:,env.obs_dof_indexes],root,q,lv,av,tips)
def observation(env,commands,ff):
 phase=env._clip_time/env.motion_duration*2*torch.pi
 return torch.cat([proprio(env),torch.stack([phase.sin(),phase.cos()],-1),commands/torch.tensor([.3,.6],device=env.device),ff,env._filtered_action],-1)
