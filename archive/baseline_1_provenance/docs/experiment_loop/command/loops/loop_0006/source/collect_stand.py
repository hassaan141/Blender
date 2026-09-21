"""Collect surviving unified-champion stand states to repair stand replay."""
import argparse,sys,os
from common import ROOT,HOME,load_policy,normalized
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--references',required=True);p.add_argument('--out',required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();os.environ['COMMAND_REFERENCE_DIR']=a.references;app=AppLauncher(a).app
sys.path.insert(0,str(ROOT/'rl/bingo_rl'))
import torch
from env import CommandEnv,CommandCfg
cfg=CommandCfg();cfg.scene.num_envs=128;cfg.sim.device=a.device;cfg.random_start_frame=True;cfg.episode_length_s=25;cfg.seed=42
env=CommandEnv(cfg);env.reset();env.held_commands=torch.zeros(128,2,device=env.device);env.commands.zero_();env.targets.zero_()
policy,checkpoint=load_policy(a.checkpoint,env.device);alive=torch.ones(128,dtype=torch.bool,device=env.device);xs=[];ys=[]
with torch.inference_mode():
 for step in range(480):
  obs=env._get_observations()['policy'];action=policy(normalized(obs,checkpoint));_,_,done,_,_=env.step(action);alive&=~done
  if step>=24:xs.append(obs.cpu());ys.append(action.cpu())
assert alive.any(),'No surviving stand demonstrations'
x=torch.stack(xs)[:,alive.cpu()].flatten(0,1);y=torch.stack(ys)[:,alive.cpu()].flatten(0,1)
torch.save({'x':x,'y':y,'checkpoint':a.checkpoint,'surviving_envs':int(alive.sum()),'initial_envs':128,'seconds':20,'filter':'Only complete surviving trajectories'},a.out)
print('STAND_COLLECTION',int(alive.sum()),'/128 survive;',len(x),'states',flush=True)
env.close();app.close()
