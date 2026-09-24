"""Completion rate from browser-like stand starts (mm-level leg deviation, random expression phase).
Usage: success_rate.py <policy.pt> <runs> [reference.npz]"""
import sys,numpy as np,torch
from multiprocessing import Pool
from skill_env import SkillEnv
from train_skill import Actor
ck,n=sys.argv[1],int(sys.argv[2]); REF=sys.argv[3] if len(sys.argv)>3 else 'out/timid_ref.npz'
def run(seed):
    torch.set_num_threads(1)
    c=torch.load(ck,map_location='cpu',weights_only=False);a=Actor();a.load_state_dict(c['actor']);a.eval()
    mu,sd=c['obs_mean'],np.sqrt(c['obs_var']+1e-8)
    e=SkillEnv(REF,seed,residual_scale=c['config'].get('residual_scale')); o=e.reset(stand_start=True,noise=True)
    e.d.qpos[e.qa[:12]]=e.stand[e.qa[:12]]+e.rng.normal(0,.003,12)   # browser-like: mm-level deviation
    import mujoco; mujoco.mj_forward(e.m,e.d); o=e.observe()
    while True:
        with torch.no_grad(): act=a(torch.as_tensor(np.clip((o-mu)/sd,-5,5),dtype=torch.float32)).numpy()
        o,r,d,i=e.step(act)
        if d: return (not i['fell'], e.k)
with Pool(20) as p: res=p.map(run,range(1000,1000+n))
ok=[r[0] for r in res]; print(f'completed {sum(ok)}/{n}; fall frames: {sorted(r[1] for r in res if not r[0])[:30]}')
