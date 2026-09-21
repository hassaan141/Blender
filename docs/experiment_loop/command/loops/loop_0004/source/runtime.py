"""Inference-only public interface for the unified command-conditioned policy.

Launch Isaac Lab first and construct the matching CommandEnv/reference bundle.
This class does not load teacher policies. It returns 12 raw residual actions;
the matching environment applies reference + EMA + residual scale.
"""
import torch
from common import load_policy,normalized
class CommandController:
 def __init__(self,checkpoint,device):
  self.policy,self.checkpoint=load_policy(checkpoint,device)
  if self.policy.net_container[0].in_features!=95:
   raise ValueError('Expected unified 95-observation checkpoint; teacher checkpoints are incompatible')
 @torch.inference_mode()
 def act(self,env,command):
  command=torch.as_tensor(command,dtype=torch.float32,device=env.device)
  if command.shape==(2,):command=command.expand(env.num_envs,2)
  if command.shape!=(env.num_envs,2) or not torch.isfinite(command).all():raise ValueError('Expected finite [vx,yaw] or [num_envs,2]')
  if ((command[:,0]<-.2)|(command[:,0]>.3)|(command[:,1].abs()>.6)).any():raise ValueError('Command outside vx [-.20,.30], yaw [-.60,.60]')
  env.held_commands=command.clone()
  return self.policy(normalized(env._get_observations()['policy'],self.checkpoint))

def keyboard_command(keys):
 """Default keyboard mapping; opposite keys cancel, release commands stand."""
 keys={str(k).upper() for k in keys};direction=int('W' in keys)-int('S' in keys);turn=int('A' in keys)-int('D' in keys)
 return [direction*(.15 if turn else .20),turn*.4]
