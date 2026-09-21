"""Use the existing Isaac Lab/skrl PPO trainer with an isolated command task."""
import sys,runpy,os
from common import ROOT
import gymnasium as gym
original_make,original_spec=gym.make,gym.spec
registered=False
def register(task):
 global registered
 if isinstance(task,str) and task.startswith('Bingo-Command-') and not registered:
  import env
  import torch
  from skrl.agents.torch.ppo import PPO
  from common import HOME
  original_update=PPO.update
  def update(agent,*,timestep,timesteps):
   original_update(agent,timestep=timestep,timesteps=timesteps)
   if not hasattr(agent,'command_replay'):
    data=[torch.load(HOME/'teachers'/f'{t}.pt',weights_only=False,map_location=agent.device) for t in ['forward','backward','turning']]
    # Replace unreliable native stand targets with complete surviving champion
    # stand trajectories. Keep all three moving-teacher demonstrations intact.
    moving=data[0]['x'][:,69:71].abs().amax(dim=1)>1e-6
    data[0]={'x':data[0]['x'][moving],'y':data[0]['y'][moving]}
    data.append(torch.load(HOME/'teachers/stand_champion3_clean.pt',weights_only=False,map_location=agent.device))
    agent.command_replay=(torch.cat([d['x'] for d in data]),torch.cat([d['y'] for d in data]))
    agent.command_bc_optimizer=torch.optim.Adam(agent.policy.parameters(),lr=1e-4)
   x,y=agent.command_replay;ids=torch.randint(len(x),(2048,),device=agent.device)
   prediction=agent.policy.net_container(agent._observation_preprocessor(x[ids]))
   loss=(prediction-y[ids]).square().mean();agent.command_bc_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.policy.parameters(),1.);agent.command_bc_optimizer.step()
   agent.track_data('Loss/teacher_replay',loss.item())
  PPO.update=update
  registered=True
def spec(task,*a,**kw):register(task);return original_spec(task,*a,**kw)
def make(task,*a,**kw):
 register(task)
 if isinstance(task,str) and task.startswith('Bingo-Command-'):
  kw['cfg'].curriculum_steps=int(os.getenv('COMMAND_CURRICULUM_STEPS','1920'))
  kw['cfg'].yaw_weight=float(os.getenv('COMMAND_YAW_WEIGHT','.3'))
 return original_make(task,*a,**kw)
gym.spec=spec;gym.make=make
runpy.run_path(str(ROOT/'rl/bingo_rl/bingo_rl/natural_walk/train.py'),run_name='__main__')
