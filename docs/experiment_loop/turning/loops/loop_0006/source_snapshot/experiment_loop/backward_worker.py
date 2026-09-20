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
        if c['settings'].get('all_contacts_corrected',0):
            import numpy as np,torch
            from isaaclab.utils.math import quat_apply
            from bingo_rl.natural_walk.env import NaturalWalkEnv
            class GeometryContactEnv(NaturalWalkEnv):
                def _get_rewards(self):
                    reward=super()._get_rewards()  # already corrects BL
                    if not hasattr(self,'_backward_hulls'):
                        z=np.load(ROOT/'stage4/out/collision_hulls.npz')
                        self._backward_hulls={j:torch.tensor(z[l+'_knee'],dtype=torch.float32,device=self.device) for j,l in [(0,'fl'),(1,'fr'),(3,'br')]}
                    d=self.robot.data;tips,_=self._foot_tips_local()
                    fidx=np.clip(np.round(self._current_times()*self._ref_fps).astype(int),0,self._n_ref_frames-1)
                    for j,h in self._backward_hulls.items():
                        idx=self.key_body_indexes[j];N=self.num_envs;V=len(h)
                        quat=d.body_quat_w[:,idx,None,:].expand(N,V,4).reshape(-1,4)
                        verts=quat_apply(quat,h[None].expand(N,V,3).reshape(-1,3)).reshape(N,V,3)
                        low=verts[:,:,2].min(dim=1).values+d.body_pos_w[:,idx,2]-self.scene.env_origins[:,2]
                        old=tips[:,j,2]<.03;new=low<.003;desired=self._ref_contacts[fidx,j]>.5
                        reward=reward+.025*((new==desired).float()-(old==desired).float())
                    return reward
            gym.registry[task].entry_point=GeometryContactEnv
        cfg.vx_range=(-.2,-.2);cfg.stand_prob=0.;cfg.command_resample_s=1000.
        save(Path(c['arm_dir'])/(stage+'.effective_settings.json'),
             {'vx_range':list(cfg.vx_range),'stand_prob':cfg.stand_prob,'motion_file':cfg.motion_file,
              'correct_bl_contact':cfg.correct_bl_contact,'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,
              'settings':c['settings'],'phase_rate':'native signed 1.4 * command / 0.66'})
    return original(task,*args,**kwargs)
gym.make=make
path=ROOT/'rl/bingo_rl/bingo_rl/natural_walk'/('train.py' if stage=='train' else 'evaluate.py')
sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
