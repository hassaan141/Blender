"""Close-up kinematic preview of an authored lateral reference, in place -- no
physics, no net root displacement (matches how the reference is actually
authored and used, see AUTONOMOUS_LATERAL_PROTOCOL.md). Adapted from
backward_kinematic.py; the only real difference is the root stays fixed rather
than being driven forward/backward externally, since this campaign's reference
authoring keeps the root static by design.
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--reference", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "rl/bingo_rl"))
import bingo_rl  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env as lateral_env  # noqa: F401  (registers Bingo-Lateral-Play-v0)
from env import LateralPlayCfg

cfg = LateralPlayCfg()
cfg.motion_file = a.reference
cfg.ear_file = a.reference
E = gym.make("Bingo-Lateral-Play-v0", cfg=cfg, render_mode="rgb_array")
E.reset()
b = E.unwrapped
d = np.load(a.reference)
names = list(d["dof_names"])
idx = [b.robot.data.joint_names.index(str(n)) for n in names]
dur = (len(d["dof_positions"]) - 1) / float(d["fps"])
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

with imageio.get_writer(str(out), fps=24, codec="libx264") as writer:
    for k in range(int(24 * a.seconds)):
        phase = (k / 24.0) % dur
        f = phase * float(d["fps"])
        lo = int(f); hi = min(lo + 1, len(d["dof_positions"]) - 1); w = f - lo
        q = (1 - w) * d["dof_positions"][lo] + w * d["dof_positions"][hi]
        rp = d["body_positions"][0, 0].copy()  # root static, by design -- see module docstring
        rq = d["body_rotations"][0, 0]
        root = torch.zeros((1, 13), device=b.device)
        root[0, :3] = torch.tensor(rp, device=b.device)
        root[0, 3:7] = torch.tensor(rq, device=b.device)
        b.robot.write_root_state_to_sim(root)
        qt = torch.tensor(q[None], device=b.device)
        b.robot.write_joint_state_to_sim(qt, torch.zeros_like(qt), joint_ids=idx)
        b.scene.write_data_to_sim(); b.sim.forward(); b.scene.update(0)
        if k == 0:
            for _ in range(12):
                b.sim.render(); E.render()
        b.sim.render()
        writer.append_data(E.render())
E.close()
app.close()
