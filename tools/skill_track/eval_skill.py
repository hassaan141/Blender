#!/usr/bin/env python3
"""Evaluate a skill policy from the browser's starting condition (standing), then hand
back to plain standing (legs -> stand pose, Neutral expression, no residual) for 2 s.

Reports completion, tracking, slip, torque saturation and residual size vs the
authored motion; optional video. `--zero-residual` evaluates open-loop playback.
"""
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
import torch
from skill_env import SkillEnv, RESIDUAL_SCALE, neutral_expression, qconj, qrot
from train_skill import Actor


def rollout(env, policy, noise, seed_anchor, video=None):
    env.rng = np.random.default_rng(seed_anchor)
    obs = env.reset(stand_start=True, noise=noise)
    H = []; stand_legs = env.stand[env.qa[:12]]; done = False
    while not done:
        a = policy(obs) if policy else np.zeros(12); obs, r, done, info = env.step(a, residual=policy is not None); info["k"] = env.k; H.append(info)
        if video: video(env)
    completed = not H[-1]["fell"]
    handoff_ok = None
    if completed:  # browser: skill ends -> stand controller (legs at stand pose, live expression)
        d = env.d; t0 = np.random.default_rng(seed_anchor).uniform(0, 20)
        for s in range(48):
            d.ctrl[env.aid[:12]] = stand_legs; d.ctrl[env.aid[12:]] = neutral_expression(t0 + s/24)
            for _ in range(5): __import__("mujoco").mj_step(env.m, d)
            if video: video(env)
        q = d.qpos[3:7]; tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2*(q[1]**2 + q[2]**2)))))
        handoff_ok = bool(d.qpos[2] > .15 and tilt < 20)
    clip = [h for h in H if env.ref.tin <= h["k"] < env.ref.tin + env.ref.meta["clip_frames"]]
    slip = []
    for h0, h1 in zip(H, H[1:]):
        for c in range(4):
            if h0["contact"][c] and h1["contact"][c]: slip.append(np.linalg.norm(h1["tips"][c, :2] - h0["tips"][c, :2]) * 24)
    ref_motion = env.ref.legs[env.ref.tin:env.ref.tin + env.ref.meta["clip_frames"]]
    res = np.array([h["residual"] for h in H])
    return {"completed": completed, "frames": len(H), "of": env.ref.n - 1, "handoff_stand_ok": handoff_ok,
            "joint_rms": float(np.sqrt(np.mean([h["joint_rms"]**2 for h in clip]))) if clip else None,
            "ori_err_deg_mean": float(np.degrees(np.mean([h["ang"] for h in clip]))) if clip else None,
            "ori_err_deg_max": float(np.degrees(np.max([h["ang"] for h in clip]))) if clip else None,
            "root_xy_err_final_m": float(H[-1]["dxy"]), "root_z_err_mean_m": float(np.mean([abs(h["dz"]) for h in H])),
            "tip_rms_m": float(np.sqrt(np.mean([h["tip_rms"]**2 for h in H]))), "contact_match": float(np.mean([h["cmatch"] for h in H])),
            "slip_mps_mean": float(np.mean(slip)) if slip else 0., "torque_sat_pct": float(100*np.mean(np.abs([h["torque"] for h in H]) > 2.97)),
            "residual_rms_rad": float(np.sqrt(np.mean(res**2))), "residual_max_rad": float(np.abs(res).max()),
            "ref_motion_rms_rad": float(np.sqrt(np.mean((ref_motion - ref_motion.mean(0))**2)))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint"); p.add_argument("--zero-residual", action="store_true")
    p.add_argument("--ref", default=str(Path(__file__).parent / "out/timid_ref.npz")); p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=10); p.add_argument("--video", action="store_true")
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    policy = None
    if not a.zero_residual:
        c = torch.load(a.checkpoint, map_location="cpu", weights_only=False); act = Actor(); act.load_state_dict(c["actor"]); act.eval()
        mu, sd = c["obs_mean"], np.sqrt(c["obs_var"] + 1e-8)
        policy = lambda o: act(torch.as_tensor(np.clip((o - mu)/sd, -5, 5), dtype=torch.float32)).detach().numpy()
    env = SkillEnv(a.ref, 0, residual_scale=c["config"].get("residual_scale") if not a.zero_residual else None); results = {}
    video = None
    if a.video:
        import mujoco
        r = mujoco.Renderer(env.m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = .8, 135, -15; opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
        ff = subprocess.Popen(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "24",
                               "-i", "-", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(out / "deterministic.mp4")], stdin=subprocess.PIPE)
        def video(e):
            cam.lookat[:] = [e.d.qpos[0], e.d.qpos[1], .12]; r.update_scene(e.d, cam, opt); ff.stdin.write(r.render().tobytes())
    results["deterministic"] = rollout(env, policy, False, 0, video)
    if a.video: ff.stdin.close(); ff.wait()
    for s in range(1, a.runs + 1): results[f"noisy_{s}"] = rollout(env, policy, True, s)
    noisy = [v for k, v in results.items() if k.startswith("noisy")]
    summary = {"completed": f"{sum(v['completed'] for v in noisy)}/{len(noisy)} noisy, deterministic={results['deterministic']['completed']}",
               "handoff_ok": f"{sum(bool(v['handoff_stand_ok']) for v in noisy)}/{len(noisy)}",
               **{k: float(np.mean([v[k] for v in noisy if v[k] is not None])) for k in
                  ("joint_rms", "ori_err_deg_mean", "ori_err_deg_max", "root_xy_err_final_m", "tip_rms_m", "contact_match", "slip_mps_mean",
                   "torque_sat_pct", "residual_rms_rad", "ref_motion_rms_rad") if any(v[k] is not None for v in noisy)}}
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
