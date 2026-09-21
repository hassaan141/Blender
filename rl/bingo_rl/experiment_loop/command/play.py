"""Run one policy with a held physical command; no teacher policy switching."""
import argparse,os,sys
from pathlib import Path
from common import ROOT
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--references',required=True);p.add_argument('--vx',type=float,default=.2);p.add_argument('--yaw',type=float,default=0);p.add_argument('--seconds',type=float,default=20)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();os.environ['COMMAND_REFERENCE_DIR']=str(Path(a.references).resolve());app=AppLauncher(a).app
sys.path.insert(0,str(ROOT/'rl/bingo_rl'))
from env import CommandEnv,CommandCfg
from runtime import CommandController
cfg=CommandCfg();cfg.scene.num_envs=1;cfg.sim.device=a.device;cfg.random_start_frame=False;cfg.episode_length_s=a.seconds+5
env=CommandEnv(cfg);env.reset();controller=CommandController(a.checkpoint,env.device)
for _ in range(int(a.seconds/env.step_dt)):
 action=controller.act(env,[a.vx,a.yaw]);_,_,done,_,_=env.step(action)
 if bool(done.any()):print('FALL: command replay failed',flush=True);break
env.close();app.close()
