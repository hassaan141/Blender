"""Play a motion .npz on Bingo WITH PHYSICS. No RL, no reward, no training.

The reference joint angles are fed to the real PD actuators as position targets
while gravity and contacts are on, and the robot is left to cope. This answers
the question kinematic replay cannot: is this motion something the actual
machine could perform?

    python scripts/playback_physics.py --motion motions/clip.npz --out output/<name> --headless \
        --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"
"""
import argparse, os, importlib.util
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", required=True)
parser.add_argument("--out", default=None, help="if set, write video frames here")
parser.add_argument("--loops", type=int, default=3, help="times to repeat the clip")
parser.add_argument("--kp_scale", type=float, default=1.0, help="multiply PD stiffness/damping")
parser.add_argument("--settle", type=float, default=0.5, help="seconds held at frame 0 first")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = args.out is not None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch, numpy as np, sys
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

sys.path.insert(0, "/pub0/muhammadf/BingoRobotics/bingo_rl")
from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG

_p = "/pub0/muhammadf/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py"
_s = importlib.util.spec_from_file_location("ml", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
MotionLoader = _m.MotionLoader

PHYS_DT, DECIMATION = 1.0 / 120.0, 4          # 30 Hz control, matches the env


def main():
    dev = args.device if hasattr(args, "device") else "cuda:0"
    sim = SimulationContext(SimulationCfg(dt=PHYS_DT, device=dev))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    cfg = BINGO_IMPROVED_CFG.replace(prim_path="/World/Robot")
    if args.kp_scale != 1.0:
        import copy as _c
        cfg = _c.deepcopy(cfg)
        for act in cfg.actuators.values():
            for attr in ("stiffness", "damping"):
                v = getattr(act, attr)
                if isinstance(v, dict):
                    setattr(act, attr, {k: x * args.kp_scale for k, x in v.items()})
                elif v is not None:
                    setattr(act, attr, v * args.kp_scale)
    robot = Articulation(cfg)

    cam = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        cam = Camera(CameraCfg(prim_path="/World/cam", height=720, width=1280,
                               data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                                               clipping_range=(0.01, 100.0)),
                               offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0))))
    sim.reset()
    motion = MotionLoader(motion_file=args.motion, device=dev)
    r_idx = [robot.data.joint_names.index(n) for n in motion.dof_names]
    m_idx = motion.get_dof_index(list(motion.dof_names))

    # start exactly on the reference's first frame so we test tracking, not recovery
    dp, dv, bp, br, blv, bav = motion.sample(num_samples=1, times=np.array([0.0], dtype=np.float32))
    root = torch.cat([bp[:, 0], br[:, 0]], dim=-1)
    robot.write_root_link_pose_to_sim(root)
    robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
    jp = robot.data.default_joint_pos.clone(); jv = torch.zeros_like(jp)
    jp[:, r_idx] = dp[:, m_idx]
    robot.write_joint_state_to_sim(jp, jv)
    robot.reset()

    n_ctrl = int((args.settle + motion.duration * args.loops) * 30)
    err_hist, h_hist, frames_written = [], [], 0

    for i in range(n_ctrl):
        t = max(0.0, i / 30.0 - args.settle) % motion.duration
        dp, dv, bp, br, blv, bav = motion.sample(num_samples=1, times=np.array([t], dtype=np.float32))
        target = robot.data.default_joint_pos.clone()
        target[:, r_idx] = dp[:, m_idx]
        robot.set_joint_position_target(target)

        for _ in range(DECIMATION):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYS_DT)

        actual = robot.data.joint_pos[:, r_idx]
        err_hist.append((actual - dp[:, m_idx]).abs().max().item())
        h_hist.append(robot.data.root_pos_w[0, 2].item())

        if cam is not None and i % 1 == 0:
            rp = robot.data.root_pos_w[0].cpu().numpy()
            eye = torch.tensor([[rp[0] - 0.55, rp[1] - 0.75, rp[2] + 0.35]], device=dev, dtype=torch.float32)
            tgt = torch.tensor([[rp[0], rp[1], rp[2]]], device=dev, dtype=torch.float32)
            cam.set_world_poses_from_view(eye, tgt)
            sim.render()
            cam.update(1 / 30.0)
            from PIL import Image
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(args.out, f"f{frames_written:05d}.png"))
            frames_written += 1

    e = np.array(err_hist); h = np.array(h_hist)
    print(f"[[ control steps {n_ctrl} ({n_ctrl/30:.2f}s, {args.loops} loops)")
    print(f"[[ joint tracking error: mean {e.mean():.4f} rad ({np.degrees(e.mean()):.2f} deg) "
          f"max {e.max():.4f} rad ({np.degrees(e.max()):.2f} deg)")
    print(f"[[ base height: start {h[0]:.4f} end {h[-1]:.4f} min {h.min():.4f} max {h.max():.4f} m")
    fell = h.min() < 0.15
    print(f"[[ VERDICT: {'FELL (base under 0.15 m) - motion not achievable as-is' if fell else 'stayed up'}"
          f"; final height {h[-1]:.4f} m")
    if frames_written:
        print(f"[[ wrote {frames_written} frames to {args.out}")


try:
    main()
finally:
    simulation_app.close()
