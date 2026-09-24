#!/usr/bin/env python3
"""Python-vs-browser parity for a tracked skill.

Starts SkillEnv from the browser's exact state at skill start (qpos, qvel, anchor,
live expression) and runs the same policy closed-loop, comparing per step:
observation, action, actuator targets and physical state against probe_skill.mjs.
Usage: parity_skill.py <policy.pt> <reference.npz> <probe.json>
"""
import json, sys
import numpy as np
import torch
import mujoco
from skill_env import SkillEnv, qz
from train_skill import Actor

ckpt, ref, probe = sys.argv[1:4]
c = torch.load(ckpt, map_location="cpu", weights_only=False); actor = Actor(); actor.load_state_dict(c["actor"]); actor.eval()
mu, sd = c["obs_mean"], np.sqrt(c["obs_var"] + 1e-8)
pol = lambda o: actor(torch.as_tensor(np.clip((o - mu)/sd, -5, 5), dtype=torch.float32)).detach().numpy()
P = json.load(open(probe)); meta = P["runs"][0]; run = P["runs"][1]
env = SkillEnv(ref, 0, residual_scale=c["config"].get("residual_scale")); env.reset(stand_start=True, noise=False)
d = env.d; d.qpos[:] = run["start"]["qpos"]; d.qvel[:] = run["start"]["qvel"]; d.ctrl[:] = run["start"]["ctrl"]
env.anchor = np.array(meta["anchor"]); env.aq = qz(env.anchor[2]); env.expr0 = np.array(meta["expr0"]); env.k = 0
mujoco.mj_forward(env.m, d)
seg = {"gravity": (0, 3), "angvel": (3, 6), "linvel": (6, 9), "z": (9, 10), "leg_q": (10, 22), "leg_qd": (22, 34), "prev_a": (34, 46),
       "ref_legs": (46, 70), "ref_root": (70, 74), "pos_err": (74, 77), "heading": (77, 79), "phase": (79, 82)}
rows = []; first_big = None
for i, st in enumerate(run["steps"]):
    o = env.observe(); bo = np.array(st["obs"]); a = pol(o)
    rows.append({"step": i+1, "obs_max": float(np.abs(o - bo).max()), "act_py_vs_br": float(np.abs(a - np.array(st["act"])).max()),
                 "act_same_obs": float(np.abs(pol(bo) - np.array(st["act"])).max()),
                 "seg": {k: float(np.abs(o[s:e] - bo[s:e]).max()) for k, (s, e) in seg.items()}})
    _, _, done, info = env.step(a)
    rows[-1]["ctrl_max"] = float(np.abs(d.ctrl - np.array(st["ctrl"])).max()); rows[-1]["qpos_max"] = float(np.abs(d.qpos - np.array(st["qpos"])).max())
    if first_big is None and rows[-1]["qpos_max"] > 1e-3: first_big = i + 1
    if done: break
show = range(int(sys.argv[4]), int(sys.argv[5]) + 1) if len(sys.argv) > 5 else (1, 2, 12, 24, 48, 96, 150, 200, len(rows))
for r in rows:
    if r["step"] in show:
        print(f"step {r['step']:3d}: obs {r['obs_max']:.1e} act {r['act_py_vs_br']:.1e} (same-obs {r['act_same_obs']:.1e}) ctrl {r['ctrl_max']:.1e} qpos {r['qpos_max']:.1e}")
worst = max(rows[:24], key=lambda r: r["obs_max"])
print("worst obs segment (first 24 steps):", max(worst["seg"].items(), key=lambda x: x[1]))
print(f"python done={done} fell={info['fell']} steps={len(rows)} | browser steps={len(run['steps'])} completed={run['completed']} | first qpos>1e-3 at step {first_big}")
# ---- permanent gate --------------------------------------------------------------
# Full trajectories legitimately bifurcate at contact events (float-level noise amplified),
# so the gate checks what must be deterministic: same obs -> same action (ONNX vs torch)
# everywhere, tight state/obs lock through the entering transition, and the same outcome.
tin = env.ref.tin
checks = {"same_obs_action<1e-5": max(r["act_same_obs"] for r in rows) < 1e-5,
          "entering_obs<1e-3": max(r["obs_max"] for r in rows[:tin]) < 1e-3,
          "entering_qpos<1e-4": max(r["qpos_max"] for r in rows[:tin]) < 1e-4,
          "same_outcome": (not info["fell"]) == bool(run["completed"])}
print("PARITY", "PASS" if all(checks.values()) else "FAIL", checks)
sys.exit(0 if all(checks.values()) else 1)
