"""Kinematic replay of a motion .npz on the Bingo robot (spec B.3 Stage 7).

Writes the reference root pose + joint state into the articulation every step and
renders it. No policy, no physics tracking -- this shows exactly what the
reference asks the robot to do, which is the thing to inspect before spending
GPU time on training.

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p <repo>/scripts/replay_motion.py \
        --motion <repo>/motions/bingo_deadpan.npz \
        --out <repo>/output/video/deadpan_replay --headless
"""

import argparse, os, importlib.util
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", required=True)
parser.add_argument("--out", required=True, help="output dir for frames + mp4")
parser.add_argument("--fps", type=int, default=30, help="output video fps")
parser.add_argument("--max_seconds", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.utils.math as math_utils

import sys
sys.path.insert(0, "/pub0/muhammadf/BingoRobotics/bingo_rl")
from bingo_rl.improved_walking_cfg import BINGO_IMPROVED_CFG

_p = "/pub0/muhammadf/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py"
_s = importlib.util.spec_from_file_location("ml", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
MotionLoader = _m.MotionLoader


def main():
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=dev))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    robot = Articulation(BINGO_IMPROVED_CFG.replace(prim_path="/World/Robot"))

    cam = Camera(CameraCfg(
        prim_path="/World/cam", height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    ))

    sim.reset()
    motion = MotionLoader(motion_file=args.motion, device=dev)
    print(f"motion: {motion.duration:.2f}s  dofs={motion.num_dofs}  bodies={motion.num_bodies}")

    # map motion dof order -> robot articulation joint order (spec A.4)
    m_idx = motion.get_dof_index(list(motion.dof_names))
    r_idx = [robot.data.joint_names.index(n) for n in motion.dof_names]
    print("robot joints:", len(robot.data.joint_names), "| driven by motion:", len(r_idx))

    # MotionLoader only carries the 12 leg DOF. Head and tail are written by
    # bake_conform.py into the `head_tail_positions` extension key, so read them
    # straight from the npz -- otherwise the head and tail just sit at default
    # and the animator's expressive work is invisible in the replay.
    HT_NAMES = ["head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw"]
    _raw = np.load(args.motion, allow_pickle=True)
    ht = _raw["head_tail_positions"] if "head_tail_positions" in _raw.files else None
    ht_idx = [robot.data.joint_names.index(n) for n in HT_NAMES] if ht is not None else []
    if ht is not None:
        print(f"head/tail channels found: driving {len(ht_idx)} extra joints")
    else:
        print("no head_tail_positions in this clip; head/tail stay at default")

    dur = motion.duration if args.max_seconds is None else min(motion.duration, args.max_seconds)
    n = int(dur * args.fps)
    from PIL import Image

    for i in range(n):
        t = np.array([i / args.fps], dtype=np.float32)
        dp, dv, bp, br, blv, bav = motion.sample(num_samples=1, times=t)

        root_pos, root_quat = bp[:, 0], br[:, 0]
        robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_com_velocity_to_sim(torch.cat([blv[:, 0], bav[:, 0]], dim=-1))

        jp = robot.data.default_joint_pos.clone()
        jv = torch.zeros_like(jp)
        jp[:, r_idx] = dp[:, m_idx]
        jv[:, r_idx] = dv[:, m_idx]
        if ht is not None:
            k = min(int(round(t[0] * float(_raw["fps"]))), len(ht) - 1)
            jp[:, ht_idx] = torch.tensor(ht[k], device=dev, dtype=jp.dtype)
        robot.write_joint_state_to_sim(jp, jv)

        # chase cam: 0.9 m back-left of the root, looking at it
        rp = root_pos[0].cpu().numpy()
        # dtype must be float32: create_rotation_matrix_from_view cross-products
        # against a float32 up-vector and errors on float64.
        eye = torch.tensor([[rp[0] - 0.55, rp[1] - 0.75, rp[2] + 0.35]],
                           device=dev, dtype=torch.float32)
        tgt = torch.tensor([[rp[0], rp[1], rp[2]]], device=dev, dtype=torch.float32)
        cam.set_world_poses_from_view(eye, tgt)

        sim.step(render=True)
        robot.update(1.0 / args.fps)
        cam.update(1.0 / args.fps)

        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(args.out, f"f{i:05d}.png"))
        if i % 60 == 0:
            print(f"  frame {i}/{n}  root z={rp[2]:.3f}")

    print(f"wrote {n} frames to {args.out}")


try:
    main()
finally:
    # Always close, even on an exception -- otherwise Isaac Sim hangs forever
    # holding the GPU instead of exiting with the traceback.
    simulation_app.close()
