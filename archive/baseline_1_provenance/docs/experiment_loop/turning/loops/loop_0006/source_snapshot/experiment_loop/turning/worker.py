import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read,save
stage=sys.argv.pop(1);i=sys.argv.index('--context');c=read(sys.argv[i+1]);del sys.argv[i:i+2]
import gymnasium as gym
make0,spec0=gym.make,gym.spec
registered=False
def register(task):
 global registered
 if isinstance(task,str) and task.startswith('Bingo-Turning-') and not registered:
  import env
  registered=True
def spec(task,*a,**kw):register(task);return spec0(task,*a,**kw)
def make(task,*a,**kw):
 register(task)
 if task.startswith('Bingo-Turning-'):
  cfg=kw['cfg']
  for k,v in c['settings'].items():
   if hasattr(cfg,k):setattr(cfg,k,v)
  if 'turn_command_vx' in c['settings']:
   speed=c['settings']['turn_command_vx'];cfg.vx_range=(speed-.025,speed+.025) if stage=='train' else (speed,speed)
  save(Path(c['arm_dir'])/(stage+'.effective_settings.json'),{'settings':c['settings'],'motion_file':cfg.motion_file,'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,'observation_space':cfg.observation_space,'vx_range':list(cfg.vx_range)})
 return make0(task,*a,**kw)
gym.make=make;gym.spec=spec
path=ROOT/'rl/bingo_rl/bingo_rl/natural_walk/train.py' if stage=='train' else Path(__file__).with_name('evaluate.py')
sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
