"""Apply the SAME bounded config to existing train/eval scripts without editing them."""
import sys, json, runpy
from pathlib import Path
# This script is intentionally launched only by a real train/evaluate hook.
from core import ROOT, ALLOWED, read, save
stage=sys.argv.pop(1)
i=sys.argv.index('--context'); context=Path(sys.argv[i+1]);del sys.argv[i:i+2]
c=read(context)
if c.get('smoke'):raise RuntimeError('Simulator forbidden during smoke test')
import gymnasium as gym
original_make=gym.make

def make(task,*args,**kwargs):
    if task.startswith('Bingo-NaturalWalk-'):
        cfg=kwargs['cfg']
        for k,v in c['settings'].items():
            if k in {'backward_command_vx','signed_reference_velocity_scale'}:raise ValueError('Use backward_worker.py for backward settings')
            if k not in ALLOWED or not ALLOWED[k][0]<=v<=ALLOWED[k][1]:raise ValueError(k)
            setattr(cfg,k,v)
        save(Path(c['arm_dir'])/(stage+'.effective_settings.json'),
             {**{k:getattr(cfg,k) for k in c['settings']},'motion_file':cfg.motion_file,
              'correct_bl_contact':cfg.correct_bl_contact,'physics_dt':cfg.sim.dt,'decimation':cfg.decimation})
    return original_make(task,*args,**kwargs)
gym.make=make
path=ROOT/'rl/bingo_rl/bingo_rl/natural_walk'/('train.py' if stage=='train' else 'evaluate.py')
sys.argv[0]=str(path)
# Existing source uses __file__ for imports/assets; run_path preserves its real location.
runpy.run_path(str(path),run_name='__main__')
