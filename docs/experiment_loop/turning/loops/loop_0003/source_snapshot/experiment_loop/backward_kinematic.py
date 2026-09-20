"""Close-up kinematic preview on unchanged v4 visual asset; not a physics test."""
import argparse,sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--out',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(a).app
import numpy as np,torch,gymnasium as gym,imageio.v2 as imageio
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'rl/bingo_rl'));import bingo_rl;import bingo_rl.natural_walk
from bingo_rl.natural_walk import NaturalPlayCfg
cfg=NaturalPlayCfg();cfg.motion_file=a.reference;cfg.ear_file=a.reference
E=gym.make('Bingo-NaturalWalk-Play-v0',cfg=cfg,render_mode='rgb_array');E.reset();b=E.unwrapped;d=np.load(a.reference);names=list(d['dof_names']);idx=[b.robot.data.joint_names.index(str(n)) for n in names];rate=-1.4*.20/.66;dur=(len(d['dof_positions'])-1)/float(d['fps']);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with imageio.get_writer(str(out),fps=24,codec='libx264') as writer:
 for k in range(24*8):
  phase=(k/24*rate)%dur;f=phase*float(d['fps']);lo=int(f);hi=min(lo+1,len(d['dof_positions'])-1);w=f-lo;q=(1-w)*d['dof_positions'][lo]+w*d['dof_positions'][hi];rp=(1-w)*d['body_positions'][lo,0]+w*d['body_positions'][hi,0];rp[:2]=[-k/24*.20,0];rq=d['body_rotations'][lo,0]
  root=torch.zeros((1,13),device=b.device);root[0,:3]=torch.tensor(rp,device=b.device);root[0,3:7]=torch.tensor(rq,device=b.device)
  b.robot.write_root_state_to_sim(root);qt=torch.tensor(q[None],device=b.device);b.robot.write_joint_state_to_sim(qt,torch.zeros_like(qt),joint_ids=idx);b.scene.write_data_to_sim();b.sim.forward();b.scene.update(0)
  if k==0:
   for _ in range(12):b.sim.render();E.render()
  b.sim.render();writer.append_data(E.render())
E.close();app.close()
