"""Kinematic replay in the Isaac Sim GUI viewport (no camera sensor -> no RTX stall).
Run WITHOUT --headless to watch. Drives the exact reference (legs + head/tail), no policy.
    cd ~/robotics/IsaacLab
    ./isaaclab.sh -p ~/Bingo/local/scripts/gui_replay.py --motion ~/Bingo/Blender/motions/bingo_laidback.npz
"""
import argparse, importlib.util
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--motion", required=True); p.add_argument("--loops", type=int, default=20)
AppLauncher.add_app_launcher_args(p); args, _ = p.parse_known_args()
app = AppLauncher(args).app
import torch, numpy as np, sys, time
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG
_p="/home/hassaan/robotics/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py"
_s=importlib.util.spec_from_file_location("ml",_p);_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)
MotionLoader=_m.MotionLoader
HT=["head_pitch_joint","head_yaw","head_roll","tail_pitch","tail_yaw"]

def main():
    dev="cuda:0" if torch.cuda.is_available() else "cpu"
    sim=SimulationContext(SimulationCfg(dt=1/120, device=dev))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot=Articulation(BINGO_IMPROVED_CFG.replace(prim_path="/World/Robot")); sim.reset()
    m=MotionLoader(motion_file=args.motion, device=dev)
    r_idx=[robot.data.joint_names.index(n) for n in m.dof_names]; m_idx=m.get_dof_index(list(m.dof_names))
    raw=np.load(args.motion, allow_pickle=True); ht=raw["head_tail_positions"] if "head_tail_positions" in raw.files else None
    hi=[robot.data.joint_names.index(n) for n in HT] if ht is not None else []
    fps=30
    for _ in range(args.loops):
        for i in range(int(m.duration*fps)):
            t=np.array([i/fps],np.float32); dp,dv,bp,br,blv,bav=m.sample(num_samples=1,times=t)
            robot.write_root_link_pose_to_sim(torch.cat([bp[:,0],br[:,0]],dim=-1))
            robot.write_root_com_velocity_to_sim(torch.cat([blv[:,0],bav[:,0]],dim=-1))
            jp=robot.data.default_joint_pos.clone(); jv=torch.zeros_like(jp)
            jp[:,r_idx]=dp[:,m_idx]
            if ht is not None:
                k=min(int(round(t[0]*float(raw["fps"]))),len(ht)-1); jp[:,hi]=torch.tensor(ht[k],device=dev,dtype=jp.dtype)
            robot.write_joint_state_to_sim(jp,jv)
            sim.step(render=True); robot.update(1/fps); time.sleep(1/fps)
main()
