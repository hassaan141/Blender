"""Instrumented gait-cycle check: does the trained policy actually swing its legs,
or achieve forward velocity via a static/dragging pose? Logs per-joint dof_pos and
per-paw height over a walking rollout and reports peak-to-peak amplitude + dominant
frequency (FFT) per leg -- unambiguous, unlike squinting at rendered video frames.

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p /pub0/muhammadf/Blender/rl/bingo_rl/scripts/diagnose_gait_cycle.py \
        --checkpoint <run>/checkpoints/best_agent.pt --headless --cmd_vx 0.25 --steps 300
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Bingo-Locomotion2-AMP-Direct-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--cmd_vx", type=float, default=0.25)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup", type=int, default=60)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import functools  # noqa: E402
print = functools.partial(print, flush=True)  # noqa: A001,E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bingo_rl  # noqa: F401,E402
from bingo_rl.locomotion2_amp.bingo_dog_amp_env_cfg import DOF_ORDER  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = 4
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wenv, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    print(f"[[ loaded {args_cli.checkpoint}")

    base = env.unwrapped
    robot = base.robot
    obs, _ = wenv.reset()

    dof_pos_hist, paw_y_hist = [], []
    with torch.inference_mode():
        for k in range(args_cli.warmup + args_cli.steps):
            base.commands[:, 0] = args_cli.cmd_vx
            base.commands[:, 1] = 0.0
            base.commands[:, 2] = 0.0
            outputs = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = wenv.step(actions)
            if k < args_cli.warmup:
                continue
            dof_pos_hist.append(robot.data.joint_pos[0, base.robot_ctrl_indexes].cpu().numpy())
            _, _, paw_pos = base._leg_world_positions()
            paw_y_hist.append(paw_pos[0, :, 2].cpu().numpy())  # env 0, 4 legs, world Z (up)

    dof_pos = np.array(dof_pos_hist)   # (T, 12)
    paw_y = np.array(paw_y_hist)       # (T, 4)
    dt = base.dt_ctrl
    fps = 1.0 / dt

    print("\n================ GAIT CYCLE DIAGNOSTIC ================")
    print(f"{args_cli.steps} steps @ {fps:.0f} Hz = {args_cli.steps*dt:.1f}s, cmd_vx={args_cli.cmd_vx}")
    print(f"actual vx (env0 mean over window): {robot.data.root_lin_vel_b[0,0].item():.3f} m/s (last step)")
    print("\nper-joint dof_pos peak-to-peak amplitude (rad) and dominant frequency (Hz):")
    for i, name in enumerate(DOF_ORDER):
        x = dof_pos[:, i]
        ptp = x.max() - x.min()
        xf = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x), d=dt)
        dom = freqs[1:][np.argmax(xf[1:])] if len(xf) > 1 else 0.0
        print(f"  {name:10s} ptp={ptp:.4f} rad  dominant_freq={dom:.2f} Hz")

    print("\nper-paw world-Z (height) peak-to-peak amplitude (m) and dominant frequency (Hz):")
    for i, leg in enumerate(["fl", "fr", "bl", "br"]):
        y = paw_y[:, i]
        ptp = y.max() - y.min()
        yf = np.abs(np.fft.rfft(y - y.mean()))
        freqs = np.fft.rfftfreq(len(y), d=dt)
        dom = freqs[1:][np.argmax(yf[1:])] if len(yf) > 1 else 0.0
        print(f"  {leg:4s} ptp={ptp:.4f} m  dominant_freq={dom:.2f} Hz")

    verdict_legs_move = all((dof_pos[:, i].max() - dof_pos[:, i].min()) > 0.05 for i in range(12))
    verdict_paws_lift = all((paw_y[:, i].max() - paw_y[:, i].min()) > 0.01 for i in range(4))
    print(f"\nVERDICT: all 12 joints show >0.05 rad ptp swing: {verdict_legs_move}")
    print(f"VERDICT: all 4 paws show >0.01 m ptp height change: {verdict_paws_lift}")
    print("=========================================================\n")

    if args_cli.out:
        np.savez(args_cli.out, dof_pos=dof_pos, paw_y=paw_y, dt=dt, dof_names=np.array(DOF_ORDER))
        print(f"[[ wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
