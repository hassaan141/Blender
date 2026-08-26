"""Stage 3 - kinematic replay of a baked Blender v4 motion on the Isaac v4 asset.

Proves that Isaac reproduces the exact Blender configuration frame-for-frame.
Nothing here is dynamics: gravity is off, no actuator targets are set, and the
full state (root pose + all 21 joints) is written every frame.

Joints are matched BY NAME - Isaac orders its DOFs breadth-first
(bl_SY_J, br_SY_J, fl_SY_J, ...) while the .npz is per-leg (fl_SY_J, fl_SP_J,
fl_knee, ...), so positional indexing would silently scramble the legs.

  --verify         check selected frames and print the error table (default)
  --frames a,b,c   which frames to check      (default 0,30,60,90,120,150,179)
  --all            replay every frame and report worst-case error
  --loops N        with --all, loop the clip N times (for watching in the GUI)

Run headless to verify, or without --headless to watch:
  ./isaaclab.sh -p rl/tools/replay_v4.py --motion <clip.npz> --all --headless
"""
import argparse, sys
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--motion", required=True)
p.add_argument("--frames", default="0,30,60,90,120,150,179")
p.add_argument("--all", action="store_true")
p.add_argument("--loops", type=int, default=1)
p.add_argument("--follow", action="store_true",
               help="camera chases the robot. OFF by default: re-setting the view every "
                    "frame fights your mouse, so the camera is now framed once and then "
                    "left alone and you can orbit/pan freely.")
p.add_argument("--ground", action="store_true",
               help="spawn the ground plane. Off for verification: a paw resting on "
                    "or through the floor generates CONTACT forces, which are physics "
                    "and would perturb a kinematic replay.")
p.add_argument("--urdf", default="/home/hassaan/Bingo/Blender/URDF/"
                                 "bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
from bingo_rl.bingo_v4 import BINGO_V4_CFG
from v4_kinematics import V4Kin, LEGS, quat_to_mat


def quat_angle_deg(qa, qb):
    """Angle between two wxyz quaternions, sign-ambiguity aware."""
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


def urdf_body_positions(kin, q, root_pos, root_quat):
    """World positions of the leg bodies implied by the Blender motion, via URDF FK.
    Used as an independent check that the whole robot - not just the joint scalars -
    lands where Blender put it."""
    R = quat_to_mat(root_quat)
    out = {}
    for k, leg in enumerate(LEGS):
        sp, knee, ankle, contact, _ = kin.leg_points(leg, q[3 * k:3 * k + 3])
        out[f"{leg}_shoulder_pitch"] = root_pos + R @ sp
        out[f"{leg}_knee"] = root_pos + R @ knee
    return out


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Gravity off + no actuator targets: the replay must be purely kinematic.
    sim = SimulationContext(SimulationCfg(dt=1 / 120, device=dev, gravity=(0.0, 0.0, 0.0)))
    # Ground is ON for watching (visual reference) and OFF for headless
    # verification, where paw contact would perturb a kinematic replay.
    if args.ground or not args.headless:
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(BINGO_V4_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    m = np.load(args.motion, allow_pickle=True)
    dof = m["dof_positions"].astype(np.float64)
    # Two equivalent schemas: the solver writes root_pos/root_quat directly, while
    # bake_conform.py (read straight off the .blend) stores the root as body 0 of
    # the AMP-style body arrays. Same numbers, different field names.
    if "root_pos" in m.files:
        root_pos = m["root_pos"].astype(np.float64)
        root_quat = m["root_quat"].astype(np.float64)    # wxyz
    else:
        root_pos = m["body_positions"][:, 0].astype(np.float64)
        root_quat = m["body_rotations"][:, 0].astype(np.float64)   # wxyz
    npz_names = [str(n) for n in m["dof_names"]]
    fps = float(m["fps"]); T = dof.shape[0]

    isaac_names = list(robot.data.joint_names)
    missing = [n for n in npz_names if n not in isaac_names]
    if missing:
        print(f"[[ FAIL missing joints in Isaac: {missing}", flush=True)
        import os; os._exit(1)
    # name -> index remap (the whole point)
    idx = torch.tensor([isaac_names.index(n) for n in npz_names], device=dev, dtype=torch.long)
    print(f"[[ motion {args.motion}", flush=True)
    print(f"[[ {T} frames @ {fps:g} fps | {len(npz_names)} joints mapped by name", flush=True)
    print(f"[[ npz order  : {npz_names}", flush=True)
    print(f"[[ isaac order: {isaac_names}", flush=True)

    if not args.headless:
        # frame the clip ONCE, then hand the camera to the user
        c0 = root_pos.mean(0)
        sim.set_camera_view(eye=(c0[0] + 0.6, c0[1] - 0.6, c0[2] + 0.30),
                            target=(c0[0], c0[1], c0[2]))
        print("[[ camera framed once - orbit/pan freely (use --follow to chase)", flush=True)
    kin = V4Kin(args.urdf)
    body_names = list(robot.data.body_names)

    def apply(i):
        """Write the full kinematic state for frame i, then read it back."""
        jp = torch.zeros((1, len(isaac_names)), device=dev, dtype=torch.float32)
        jp[0, idx] = torch.tensor(dof[i], device=dev, dtype=torch.float32)
        jv = torch.zeros_like(jp)
        pose = torch.tensor(np.concatenate([root_pos[i], root_quat[i]]),
                            device=dev, dtype=torch.float32).unsqueeze(0)
        robot.write_root_link_pose_to_sim(pose)
        robot.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(jp, jv)
        # The asset carries implicit PD actuators. With no target set they default
        # to default_joint_pos (the crouch) and drag the pose off during the step,
        # which also kicks the floating base. Command the replayed pose so the
        # drives have zero error and the step is a no-op.
        robot.set_joint_position_target(jp)
        robot.write_data_to_sim()
        pre = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        sim.step(render=not args.headless)
        robot.update(1 / 120)
        if not args.headless and args.follow:
            c = root_pos[i]
            sim.set_camera_view(eye=(c[0] + 0.55, c[1] - 0.55, c[2] + 0.30),
                                target=(c[0], c[1], c[2]))
        return jp, pre

    sel = ([int(x) for x in args.frames.split(",")] if not args.all
           else list(range(T)))
    sel = [f for f in sel if 0 <= f < T]

    j_err = []; p_err = []; q_err = []; b_err = []; pre_err = []
    print(f"\n[[ {'frame':>5s} {'joint max':>11s} {'joint mean':>11s} "
          f"{'root pos':>10s} {'root ori':>10s} {'body max':>10s}", flush=True)
    for i in sel:
        jp, pre = apply(i)
        got_j = robot.data.joint_pos[0].detach().cpu().numpy()
        want_j = jp[0].detach().cpu().numpy()
        dj = np.abs(got_j - want_j)
        dpre = np.abs(pre - want_j)
        got_p = robot.data.root_pos_w[0].detach().cpu().numpy()
        got_q = robot.data.root_quat_w[0].detach().cpu().numpy()   # wxyz
        dp = float(np.linalg.norm(got_p - root_pos[i]))
        dq = quat_angle_deg(got_q, root_quat[i])
        # independent whole-body check against URDF FK from the Blender motion
        want_b = urdf_body_positions(kin, dof[i], root_pos[i], root_quat[i])
        gotb = robot.data.body_pos_w[0].detach().cpu().numpy()
        db = max(float(np.linalg.norm(gotb[body_names.index(bn)] - wp))
                 for bn, wp in want_b.items() if bn in body_names)
        j_err.append(dj); p_err.append(dp); q_err.append(dq); b_err.append(db)
        pre_err.append(dpre)
        if not args.all or i in (0, T - 1) or i % 30 == 0:
            print(f"[[ {i:5d} {dj.max():8.2e} {dj.mean():11.2e} "
                  f"{dp*1000:8.3f}mm {dq:8.4f}d {db*1000:8.3f}mm", flush=True)

    J = np.concatenate([e[None] for e in j_err], 0)
    print(f"\n[[ ===== STAGE 3 RESULT over {len(sel)} frame(s) =====", flush=True)
    P = np.concatenate([e[None] for e in pre_err], 0)
    print(f"[[ joint as-written: max {P.max():.3e} rad   mean {P.mean():.3e} rad", flush=True)
    print(f"[[ joint after-step: max {J.max():.3e} rad   mean {J.mean():.3e} rad", flush=True)
    print(f"[[ root position   : max {max(p_err)*1000:.4f} mm  mean {np.mean(p_err)*1000:.4f} mm", flush=True)
    print(f"[[ root orientation: max {max(q_err):.5f} deg  mean {np.mean(q_err):.5f} deg", flush=True)
    print(f"[[ body world pos  : max {max(b_err)*1000:.4f} mm (URDF FK vs Isaac)", flush=True)
    ok = J.max() < 1e-4 and max(p_err) < 1e-4 and max(q_err) < 0.05 and max(b_err) < 1e-3
    print(f"[[ STAGE3 {'PASS' if ok else 'FAIL'}", flush=True)

    if not args.headless:
        # Play at wall-clock speed, then hold the window open. Without this the
        # verification pass blasts through in a second and the app exits, which
        # looks like "it never showed anything".
        import time
        for n in range(max(1, args.loops)):
            print(f"[[ playing loop {n + 1}/{max(1, args.loops)} at {fps:g} fps", flush=True)
            for i in range(T):
                if not app.is_running():
                    break
                apply(i)
                time.sleep(1.0 / fps)
        print("[[ done - close the window to quit", flush=True)
        while app.is_running():
            sim.step(render=True)
    import os
    os._exit(0 if ok else 1)


main()
