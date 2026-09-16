"""Physics playback of a motion on Bingo (PD targets + gravity + contacts), dumping
body-link poses each control step (NO camera). Prints a feasibility verdict AND writes
poses for offscreen rendering. Answers: could the real machine hold/track this motion?
"""
import argparse, os, importlib.util
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", required=True)
parser.add_argument("--out", required=True, help="output .npz of body poses")
parser.add_argument("--loops", type=int, default=1)
parser.add_argument("--settle", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)  # headless, no camera
simulation_app = app_launcher.app

import torch, numpy as np, sys
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG

_p = "/home/hassaan/robotics/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py"
_s = importlib.util.spec_from_file_location("ml", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
MotionLoader = _m.MotionLoader

PHYS_DT, DECIMATION = 1.0 / 120.0, 4
HT_NAMES = ["head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw"]


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(dt=PHYS_DT, device=dev))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(BINGO_IMPROVED_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    motion = MotionLoader(motion_file=args.motion, device=dev)
    r_idx = [robot.data.joint_names.index(n) for n in motion.dof_names]
    m_idx = motion.get_dof_index(list(motion.dof_names))
    _raw = np.load(args.motion, allow_pickle=True)
    ht = _raw["head_tail_positions"] if "head_tail_positions" in _raw.files else None
    ht_idx = [robot.data.joint_names.index(n) for n in HT_NAMES] if ht is not None else []

    # start on the reference's first frame
    dp, dv, bp, br, blv, bav = motion.sample(num_samples=1, times=np.array([0.0], dtype=np.float32))
    robot.write_root_link_pose_to_sim(torch.cat([bp[:, 0], br[:, 0]], dim=-1))
    robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
    jp = robot.data.default_joint_pos.clone(); jv = torch.zeros_like(jp)
    jp[:, r_idx] = dp[:, m_idx]
    if ht is not None:
        jp[:, ht_idx] = torch.tensor(ht[0], device=dev, dtype=jp.dtype)
    robot.write_joint_state_to_sim(jp, jv)
    robot.reset()

    body_names = list(robot.data.body_names)
    n_ctrl = int((args.settle + motion.duration * args.loops) * 30)
    poses = np.zeros((n_ctrl, len(body_names), 7), dtype=np.float32)
    err_hist, h_hist = [], []

    for i in range(n_ctrl):
        t = max(0.0, i / 30.0 - args.settle) % motion.duration
        dp, dv, bp, br, blv, bav = motion.sample(num_samples=1, times=np.array([t], dtype=np.float32))
        target = robot.data.default_joint_pos.clone()
        target[:, r_idx] = dp[:, m_idx]
        if ht is not None:
            k = min(int(round(t * float(_raw["fps"]))), len(ht) - 1)
            target[:, ht_idx] = torch.tensor(ht[k], device=dev, dtype=target.dtype)
        robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYS_DT)
        actual = robot.data.joint_pos[:, r_idx]
        err_hist.append((actual - dp[:, m_idx]).abs().max().item())
        h_hist.append(robot.data.root_pos_w[0, 2].item())
        poses[i, :, :3] = robot.data.body_pos_w[0].cpu().numpy()
        poses[i, :, 3:] = robot.data.body_quat_w[0].cpu().numpy()
        if i % 60 == 0:
            print(f"  step {i}/{n_ctrl} h={h_hist[-1]:.3f} err={err_hist[-1]:.3f}", flush=True)

    e = np.array(err_hist); h = np.array(h_hist)
    fell = bool(h.min() < 0.15)
    print(f"PHYS tracking_err mean {e.mean():.4f} rad ({np.degrees(e.mean()):.1f} deg) max {e.max():.4f} rad", flush=True)
    print(f"PHYS base_height start {h[0]:.4f} end {h[-1]:.4f} min {h.min():.4f} max {h.max():.4f} m", flush=True)
    print(f"PHYS VERDICT {'FELL' if fell else 'STAYED_UP'} (floor 0.15 m)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, poses=poses, body_names=np.array(body_names), fps=30,
             tracking_err_mean=float(e.mean()), tracking_err_max=float(e.max()),
             h_min=float(h.min()), h_end=float(h[-1]), fell=fell)
    print(f"WROTE_POSES {args.out} shape={poses.shape}", flush=True)


_ok = True
try:
    main()
except Exception:
    import traceback; traceback.print_exc(); _ok = False
# Isaac Sim hangs on simulation_app.close() on this box; data is already written,
# so force-exit to guarantee the process terminates (frees the GPU for the next job).
os._exit(0 if _ok else 1)
