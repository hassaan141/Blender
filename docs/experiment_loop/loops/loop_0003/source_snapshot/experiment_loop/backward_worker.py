"""One fixed-speed backward experiment; native signed phase and residual PPO unchanged."""
import sys,runpy
from pathlib import Path
from core import ROOT,read,save,ALLOWED
stage=sys.argv.pop(1);i=sys.argv.index('--context');context=Path(sys.argv[i+1]);del sys.argv[i:i+2]
c=read(context)
import gymnasium as gym
original=gym.make

def make(task,*args,**kwargs):
    if task.startswith('Bingo-NaturalWalk-'):
        cfg=kwargs['cfg']
        for k,v in c['settings'].items():
            if k in ALLOWED and hasattr(cfg,k):setattr(cfg,k,v)
        cfg.vx_range=(-.2,-.2);cfg.stand_prob=0.;cfg.command_resample_s=1000.
        save(Path(c['arm_dir'])/(stage+'.effective_settings.json'),
             {'vx_range':list(cfg.vx_range),'stand_prob':cfg.stand_prob,'motion_file':cfg.motion_file,
              'correct_bl_contact':cfg.correct_bl_contact,'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,
              'settings':c['settings'],'phase_rate':'native signed 1.4 * command / 0.66'})
    return original(task,*args,**kwargs)
gym.make=make
path=ROOT/'rl/bingo_rl/bingo_rl/natural_walk'/('train.py' if stage=='train' else 'evaluate.py')
sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
