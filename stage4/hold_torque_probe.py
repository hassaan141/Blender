"""Measure the TRUE static torque each joint needs to hold the standing pose.
Very stiff gains + very high effort ceiling -> read applied_torque once settled.
That includes ground reaction and all coupling, unlike a free-hanging estimate."""
import argparse, sys, os
import numpy as np
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--sp", type=float, default=0.3); p.add_argument("--knee", type=float, default=0.6)
p.add_argument("--seconds", type=float, default=4.0)
AppLauncher.add_app_launcher_args(p); args, _ = p.parse_known_args()
app = AppLauncher(args).app
import torch, isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
sys.path.insert(0,"/home/hassaan/Bingo/Blender/rl/bingo_rl")
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage4")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
from v4_kinematics import V4Kin, LEGS, axis_rot
from contact_model import ContactModel
URDF=("/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf")
dev="cuda:0" if torch.cuda.is_available() else "cpu"; dt=1/120
kin=V4Kin(URDF); cm=ContactModel()
STAND={"fl_SP_J":-args.sp,"bl_SP_J":-args.sp,"fr_SP_J":+args.sp,"br_SP_J":-args.sp,
       "fl_knee":+args.knee,"bl_knee":+args.knee,"fr_knee":-args.knee,"br_knee":-args.knee}
lows=[]
for leg in LEGS:
    q=np.array([0.0,STAND[f"{leg}_SP_J"],STAND[f"{leg}_knee"]]); R,p=np.eye(3),np.zeros(3)
    for name,qi in zip(kin.leg_chain(leg),q):
        J=kin.j[name]; p=p+R@J["xyz"]; R=R@J["R"]@axis_rot(J["axis"],qi)
    lows.append((cm.hull[f"{leg}_knee"]@R.T+p)[:,2].min())
spawn_z=-min(lows)+0.002
sim=SimulationContext(SimulationCfg(dt=dt,device=dev,gravity=(0,0,-9.81)))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot=Articulation(BINGO_V4_CFG.replace(prim_path="/World/Robot")); sim.reset()
names=list(robot.data.joint_names); n=len(names)
# stiff hold + huge effort ceiling
big=torch.full((1,n),400.0,device=dev); bigd=torch.full((1,n),8.0,device=dev)
robot.write_joint_stiffness_to_sim(big); robot.write_joint_damping_to_sim(bigd)
for fn in ("write_joint_effort_limit_to_sim","write_joint_max_effort_to_sim"):
    if hasattr(robot,fn):
        try:
            getattr(robot,fn)(torch.full((1,n),200.0,device=dev)); print("[[ raised effort via",fn,flush=True); break
        except Exception as e: print("[[",fn,"failed",e,flush=True)
for a_ in robot.actuators.values():
    try: a_.effort_limit=torch.full_like(torch.as_tensor(a_.effort_limit,device=dev,dtype=torch.float),200.0)
    except Exception: pass
tgt=torch.zeros((1,n),device=dev)
for jn,v in STAND.items(): tgt[0,names.index(jn)]=v
robot.write_root_link_pose_to_sim(torch.tensor([[0,0,spawn_z,1.0,0,0,0]],device=dev))
robot.write_root_com_velocity_to_sim(torch.zeros((1,6),device=dev))
robot.write_joint_state_to_sim(tgt.clone(),torch.zeros_like(tgt))
for s in range(int(args.seconds/dt)):
    robot.set_joint_position_target(tgt); robot.write_data_to_sim()
    sim.step(render=False); robot.update(dt)
tq=robot.data.applied_torque[0].cpu().numpy(); q=robot.data.joint_pos[0].cpu().numpy()
w=tgt[0].cpu().numpy()
bp=robot.data.body_pos_w[0].cpu().numpy(); bq=robot.data.body_quat_w[0].cpu().numpy()
pz,ncon,other=cm.support_summary(bp,bq,list(robot.data.body_names))
print(f"[[ settled: contacts {ncon}/4  paw z {np.round(pz*1000,2)} mm  root z {float(robot.data.root_pos_w[0,2]):.4f}",flush=True)
print(f"[[ max |pos err| {np.abs(q-w).max():.4f} rad (stiff hold)",flush=True)
print("[[ TRUE static torque to hold the standing pose:",flush=True)
for j in np.argsort(np.abs(tq))[::-1]:
    print(f"     {names[j]:18s} {tq[j]:+8.3f} Nm   (err {q[j]-w[j]:+.4f})",flush=True)
os._exit(0)
