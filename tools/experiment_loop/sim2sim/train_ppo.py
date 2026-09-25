#!/usr/bin/env python3
"""Bounded PPO fine-tune of BASELINE_1 in the browser's MuJoCo MJCF."""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from mujoco_env import BingoMujoco, CASES, ROOT

class Net(nn.Module):
    def __init__(self, out):
        super().__init__(); self.net_container=nn.Sequential(nn.Linear(95,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
    def forward(self,x): return self.net_container(x)

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net_container=nn.Sequential(nn.Linear(95,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)); self.log_std_parameter=nn.Parameter(torch.zeros(12))
    def forward(self,x): return self.net_container(x)
    def dist(self,x): return Normal(self(x),self.log_std_parameter.clamp(-3.5,0.0).exp())

def normalizer(obs, stats):
    mean=stats['running_mean'].to(device=obs.device,dtype=torch.float32)
    var=stats['running_variance'].to(device=obs.device,dtype=torch.float32)
    return ((obs.to(torch.float32)-mean)/(var.sqrt()+1e-8)).clamp(-5,5)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--iterations',type=int,default=80); p.add_argument('--envs',type=int,default=12); p.add_argument('--horizon',type=int,default=128); p.add_argument('--seed',type=int,default=20260923); p.add_argument('--device',default='cuda:0'); p.add_argument('--lr',type=float,default=3e-5); p.add_argument('--teacher-penalty',type=float,default=.08); p.add_argument('--episode-seconds',type=float,default=8.); p.add_argument('--backward-left-oversample',type=float,default=1.); p.add_argument('--expression',choices=['neutral','reference'],default='neutral'); p.add_argument('--stand-min',type=float,default=0.); p.add_argument('--stand-max',type=float,default=0.,help='browser gate: stand at [0,0] for U(min,max) s, then command'); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dev=torch.device(args.device); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'BASELINE_1/policy.pt'; source=torch.load(base,map_location='cpu',weights_only=False)
    actor=Actor().to(dev); actor.load_state_dict(source['policy']); actor.train()
    teacher=Actor().to(dev); teacher.load_state_dict(source['policy']); teacher.eval()
    # BASELINE_1's value weights were trained with RSL-RL's own value
    # preprocessing; the actor is the transferable part, so initialize a fresh
    # PPO critic and avoid silently mixing incompatible return scales.
    critic=Net(1).to(dev); critic.train()
    stats=source['observation_preprocessor']; opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=args.lr)
    envs=[BingoMujoco(args.seed+i,expression=args.expression) for i in range(args.envs)]
    choices=np.array([[x,y] for _,x,y in CASES],dtype=np.float64)
    probs=np.ones(len(CASES),dtype=np.float64);probs[7]=max(1.,args.backward_left_oversample);probs/=probs.sum()
    episode_limit=max(1,round(args.episode_seconds*24));episode_steps=np.zeros(args.envs,dtype=np.int32)
    initial=np.random.choice(len(CASES),size=args.envs,p=probs)
    pending=[choices[k] for k in initial]; stand_steps=np.zeros(args.envs,dtype=np.int32)
    def begin(i,cmd):
        # Stand-then-command like the browser gate; the episode limit counts from the command.
        pending[i]=cmd; stand_steps[i]=round(np.random.uniform(args.stand_min,args.stand_max)*24)
        return envs[i].reset((0.,0.) if stand_steps[i] else tuple(cmd))
    obs=np.stack([begin(i,choices[initial[i]]) for i in range(args.envs)])
    t0=time.time(); report={'base_checkpoint':str(base),'base_sha256':__import__('hashlib').sha256(base.read_bytes()).hexdigest(),'config':vars(args),'contract':{'obs':95,'actions':12,'residual_scale':.3,'ema_alpha':.3,'command_alpha':.1,'physics_hz':120,'control_hz':24,'command_cases':CASES},'updates':[]}
    for it in range(args.iterations):
        ro=[]; ra=[]; rlp=[]; rv=[]; rr=[]; rd=[]; teacher_acts=[]; extra=[]
        for step in range(args.horizon):
            ot=torch.as_tensor(obs,device=dev); norm=normalizer(ot,stats)
            with torch.no_grad():
                dist=actor.dist(norm); action=dist.sample(); logp=dist.log_prob(action).sum(-1); val=critic(norm).squeeze(-1); teach=teacher(norm)
            act=action.clamp(-1,1).cpu().numpy(); nxt=[]; rewards=[]; dones=[]; infos=[]
            for i,e in enumerate(envs):
                o,r,d,info=e.step(act[i]);episode_steps[i]+=1
                # Penalize departure from the frozen baseline action. This is one
                # targeted MuJoCo dynamics adaptation, with no architecture change.
                r-=args.teacher_penalty*float(np.mean((action[i].detach().cpu().numpy()-teach[i].detach().cpu().numpy())**2))
                if episode_steps[i]==stand_steps[i]: e.target=np.array(pending[i],dtype=np.float64)
                episode_end=bool(d) or episode_steps[i]>=episode_limit+stand_steps[i]
                if episode_end:
                    o=begin(i,choices[np.random.choice(len(choices),p=probs)]);episode_steps[i]=0
                nxt.append(o); rewards.append(r); dones.append(float(episode_end)); infos.append(info)
            ro.append(norm.detach()); ra.append(action.detach()); rlp.append(logp.detach()); rv.append(val.detach()); rr.append(torch.tensor(rewards,device=dev)); rd.append(torch.tensor(dones,device=dev)); teacher_acts.append(teach.detach()); extra.extend(infos); obs=np.stack(nxt)
        with torch.no_grad(): last=critic(normalizer(torch.as_tensor(obs,device=dev),stats)).squeeze(-1)
        adv=[]; gae=torch.zeros(args.envs,device=dev)
        for t in reversed(range(args.horizon)):
            nv=last if t==args.horizon-1 else rv[t+1]
            alive=1-rd[t]; delta=rr[t]+.99*nv*alive-rv[t]; gae=delta+.99*.95*alive*gae; adv.append(gae)
        adv=torch.stack(adv[::-1]); ret=adv+torch.stack(rv); X=torch.cat(ro); A=torch.cat(ra); LP=torch.cat(rlp); ADV=adv.reshape(-1); RET=ret.reshape(-1); TA=torch.cat(teacher_acts)
        ADV=(ADV-ADV.mean())/(ADV.std(unbiased=False)+1e-8); n=len(X); inds=np.arange(n); losses=[]
        for _ in range(4):
            np.random.shuffle(inds)
            for ids in np.array_split(inds,8):
                ix=torch.as_tensor(ids,device=dev); d=actor.dist(X[ix]); newlp=d.log_prob(A[ix]).sum(-1); ratio=(newlp-LP[ix]).exp()
                pg=-torch.minimum(ratio*ADV[ix],ratio.clamp(.8,1.2)*ADV[ix]).mean(); vl=.5*(critic(X[ix]).squeeze(-1)-RET[ix]).square().mean(); entropy=d.entropy().sum(-1).mean(); distill=(actor(X[ix])-TA[ix]).square().mean()
                loss=pg+vl*.5-.002*entropy+args.teacher_penalty*distill
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.0); opt.step(); losses.append([float(pg),float(vl),float(distill)])
        recent=extra[-min(len(extra),args.envs*args.horizon):]
        statsrow={'iteration':it+1,'mean_reward':float(torch.stack(rr).mean()),'mean_vx':float(np.mean([x['vx'] for x in recent])),'fall_fraction':float(np.mean([x['fallen'] for x in recent])),'losses':np.mean(losses,axis=0).tolist()}
        report['updates'].append(statsrow)
        if (it+1)%10==0: print(json.dumps({'progress':it+1,'iterations':args.iterations,'elapsed_s':round(time.time()-t0,1),**statsrow}),flush=True)
    ckpt=copy.deepcopy(source); ckpt['policy']={k:v.detach().cpu() for k,v in actor.state_dict().items()}; ckpt['value']={k:v.detach().cpu() for k,v in critic.state_dict().items()}; ckpt['sim2sim']={'method':'MuJoCo PPO fine-tune with baseline-action regularization','config':vars(args),'base_sha256':report['base_sha256']}
    torch.save(ckpt,out/'policy.pt'); report['elapsed_s']=time.time()-t0; (out/'train_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(f"CHECKPOINT={out/'policy.pt'}",flush=True)
    for e in envs: del e

if __name__=='__main__': main()
