"""Stand-only causal probe: reset state x residual source, no training."""
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
import numpy as np, torch, json
ROOT=next(x for x in Path(__file__).resolve().parents if (x/'rl/bingo_rl').is_dir())
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(ROOT/'rl/bingo_rl'))
from env import CommandEnv,CommandCfg
from common import load_policy,normalized
from isaaclab.utils.math import quat_rotate_inverse
cfg=CommandCfg(); cfg.scene.num_envs=4; cfg.scene.env_spacing=3.; cfg.sim.device=a.device; cfg.random_start_frame=False; cfg.episode_length_s=8.; cfg.seed=42
env=CommandEnv(cfg); env.reset(); env.held_commands=torch.zeros(4,2,device=env.device); env.targets.zero_(); env.commands.zero_()
# Arms: walking reset + policy, walking reset + zero residual, stand reset + policy, stand reset + zero residual.
stand_ids=torch.tensor([2,3],device=env.device); root=env.robot.data.default_root_state[stand_ids].clone(); root[:,0:3]=env.scene.env_origins[stand_ids]; root[:,2]=.182; root[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device); root[:,7:]=0
pos=env.robot.data.default_joint_pos[stand_ids].clone(); vel=torch.zeros_like(pos); pos[:,env.robot_ctrl_indexes]=env._stand_pose[stand_ids]
env.robot.write_root_link_pose_to_sim(root[:,:7],stand_ids); env.robot.write_root_com_velocity_to_sim(root[:,7:],stand_ids); env.robot.write_joint_state_to_sim(pos,vel,None,stand_ids)
policy,c=load_policy(ROOT/'docs/experiment_loop/command/loops/loop_0006/policy.pt',env.device)
names=['walk_reset_policy','walk_reset_zero','stand_reset_policy','stand_reset_zero']; rows={n:[] for n in names}; alive=torch.ones(4,dtype=torch.bool,device=env.device)
with torch.inference_mode():
 for step in range(120):
  obs=env._get_observations()['policy']; action=policy(normalized(obs,c)); action[[1,3]]=0
  _,_,done,_,_=env.step(action); alive &= ~done
  d=env.robot.data; q=d.body_quat_w[:,env.ref_body_index]; lv=quat_rotate_inverse(q,d.body_lin_vel_w[:,env.ref_body_index]); av=quat_rotate_inverse(q,d.body_ang_vel_w[:,env.ref_body_index]); r22=1-2*(q[:,1]**2+q[:,2]**2); tilt=torch.rad2deg(torch.acos(r22.clamp(-1,1))); height=d.body_pos_w[:,env.ref_body_index,2]-env.scene.env_origins[:,2]; j=d.joint_pos[:,env.robot_ctrl_indexes]; err=((j-env._stand_pose)**2).mean(1).sqrt()
  for i,n in enumerate(names): rows[n].append({'t':(step+1)*env.step_dt,'alive':bool(alive[i]),'height':float(height[i]),'tilt_deg':float(tilt[i]),'vx':float(lv[i,0]),'yaw':float(av[i,2]),'joint_rmse':float(err[i]),'raw_action_rms':float(action[i].square().mean().sqrt()),'filtered_action_rms':float(env._filtered_action[i].square().mean().sqrt())})
summary={}
for n in names:
 r=rows[n]; summary[n]={'survived':r[-1]['alive'],'final':r[-1],'max_tilt_deg':max(x['tilt_deg'] for x in r),'min_height':min(x['height'] for x in r),'max_joint_rmse':max(x['joint_rmse'] for x in r)}
(Path(__file__).parent/'stand_probe.json').write_text(json.dumps({'summary':summary,'trace':rows},indent=2)+'\n'); print(json.dumps(summary,indent=2))
env.close(); app.close()
