#!/usr/bin/env python3
"""Side-by-side stills: authored reference pose (kinematic, top) vs simulated policy (bottom).

Usage: compare_frames.py <policy.pt> <reference.npz> <out.png> [frames...]
"""
import sys
import numpy as np
import torch
import mujoco
from PIL import Image
from skill_env import SkillEnv
from train_skill import Actor

ck, ref, out = sys.argv[1:4]
frames = [int(x) for x in sys.argv[4:]] or [24, 48, 72, 100, 130, 160, 190, 215, 239]
c = torch.load(ck, map_location="cpu", weights_only=False); a = Actor(); a.load_state_dict(c["actor"]); a.eval()
mu, sd = c["obs_mean"], np.sqrt(c["obs_var"] + 1e-8)
env = SkillEnv(ref, 0, residual_scale=c["config"].get("residual_scale")); o = env.reset(stand_start=True, noise=False)
kin = mujoco.MjData(env.m)
r = mujoco.Renderer(env.m, 240, 320); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = .75, 150, -15; opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
top, bot = [], []
def shot(d, look):
    cam.lookat[:] = [look[0], look[1], .1]; r.update_scene(d, cam, opt); return r.render().copy()
while env.k < max(frames):
    with torch.no_grad(): act = a(torch.as_tensor(np.clip((o - mu)/sd, -5, 5), dtype=torch.float32)).numpy()
    o, _, done, _ = env.step(act)
    if env.k in frames:
        kin.qpos[:] = env.stand; kin.qpos[:3] = env.ref_pos(env.k); kin.qpos[3:7] = env.ref_quat(env.k)
        kin.qpos[env.qa[:12]] = env.ref.legs[env.k]; kin.qpos[env.qa[12:]] = env.ref_expr(env.k); mujoco.mj_kinematics(env.m, kin)
        top.append(shot(kin, kin.qpos[:2])); bot.append(shot(env.d, env.ref_pos(env.k)[:2]))
    if done: break
Image.fromarray(np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)).save(out)
print(f"saved {out}: frames {frames[:len(top)]} (top = authored reference, bottom = simulated)")
