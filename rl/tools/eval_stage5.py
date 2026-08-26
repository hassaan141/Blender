"""Stage 5 evaluation: PD-only vs PD + residual RL, in ONE environment.

Both conditions run in the SAME Stage 5 env (same physics, same BINGO_V4_CFG
IdealPD actuators, same ground material, same 24 Hz first-order-hold reference),
so the only difference is whether the residual comes from the policy or is zero:

    --checkpoint <ckpt.pt>   PD + residual RL   (Stage 5)
    (no --checkpoint)        zero residual      (== Stage 4 PD-only)

Metrics use the Stage 4 convention exactly (state after the decimation window is
compared against the reference frame that STARTED the window) so the numbers are
directly comparable to stage4/out/timid_v4_stage4.csv.

  ./isaaclab.sh -p rl/tools/eval_stage5.py --headless
  ./isaaclab.sh -p rl/tools/eval_stage5.py --headless --checkpoint <ckpt.pt>
"""
import argparse, os, sys
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Bingo-Stage5-Timid-Play-v0")
p.add_argument("--checkpoint", default=None,
               help="skrl checkpoint. Omitted => zero residual (Stage 4 PD-only).")
p.add_argument("--out", default="/home/hassaan/Bingo/Blender/stage5/out")
p.add_argument("--label", default=None)
p.add_argument("--stage4-csv", default="/home/hassaan/Bingo/Blender/stage4/out/timid_v4_stage4.csv")
p.add_argument("--ref-vel-init", type=int, default=None,
               help="1 = seed root/joint velocity from the reference, 0 = zeros (Stage 4 exact)")
p.add_argument("--replicate-physics", type=int, default=None,
               help="scene.replicate_physics override (diagnostic)")
p.add_argument("--no-early-term", action="store_true",
               help="run the full clip even if the fall/drift criterion trips")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch
import gymnasium as gym

sys.path.insert(0, "/home/hassaan/Bingo/Blender/rl/bingo_rl")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage4")
import bingo_rl  # noqa: F401  registers the tasks
from contact_model import ContactModel
from v4_kinematics import LEGS
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

CONTACT_H = 0.005          # same as Stage 4: paw within 5 mm of the floor
LABEL = args.label or ("stage5_residual" if args.checkpoint else "stage4_pd_only")


def quat_angle_deg(qa, qb):
    d = np.abs(np.sum(qa * qb, axis=-1) /
               (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1) + 1e-12))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def main():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.random_start_frame = False          # deterministic: start at frame 0
    if args.ref_vel_init is not None:
        env_cfg.rsi_use_reference_velocity = bool(args.ref_vel_init)
    if args.replicate_physics is not None:
        env_cfg.scene.replicate_physics = bool(args.replicate_physics)
    if args.no_early_term:
        env_cfg.early_termination = False
    env = gym.make(args.task, cfg=env_cfg)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    runner = None
    if args.checkpoint:
        from skrl.utils.runner.torch import Runner
        agent_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wenv, agent_cfg)
        runner.agent.load(os.path.abspath(args.checkpoint))
        runner.agent.enable_training_mode(False, apply_to_models=True)
        print(f"[[ policy loaded: {args.checkpoint}", flush=True)
    else:
        print("[[ no checkpoint: residual forced to ZERO (Stage 4 PD-only)", flush=True)

    # per-joint effort limits, for saturation accounting (as Stage 4 does)
    robot = base.robot
    names = list(robot.data.joint_names)
    eff = np.full(len(names), np.inf)
    for act in robot.actuators.values():
        ids = act.joint_indices
        ids = np.arange(len(names)) if isinstance(ids, slice) else \
            np.atleast_1d(np.asarray(ids if not isinstance(ids, torch.Tensor)
                                     else ids.detach().cpu().numpy())).reshape(-1).astype(int)
        e = act.effort_limit
        e = (e.detach().cpu().numpy().reshape(-1) if isinstance(e, torch.Tensor)
             else np.atleast_1d(np.asarray(e, dtype=float)).reshape(-1))
        for k, j in enumerate(ids):
            eff[int(j)] = e[k] if e.size > 1 else e[0]

    cm = ContactModel()
    body_names = list(robot.data.body_names)
    knee_i = {l: body_names.index(f"{l}_knee") for l in LEGS}
    ref_q = base._ref_all_q.detach().cpu().numpy()
    isaac_of_ref = base._all_dof_isaac.detach().cpu().numpy()
    ref_root_p = base._ref_root_p.detach().cpu().numpy()
    ref_root_q = base._ref_root_q.detach().cpu().numpy()
    T = base._T
    n_leg = len(base.ctrl_dof_names)
    leg_isaac = base.robot_ctrl_indexes.detach().cpu().numpy()

    obs, _ = wenv.reset()
    L = {k: [] for k in ("frame", "qerr21", "qerr_leg", "root_p", "root_q", "tilt",
                         "paw_z", "paw_xy", "contacts", "torque", "resid")}
    zero = torch.zeros((1, n_leg), device=base.device)
    ended_reason = "ran out of steps"
    completed = False

    for step in range(T):
        f0 = int(base._frames(0)[0].item())      # frame that STARTS this window
        if runner is not None:
            with torch.inference_mode():
                out = runner.agent.act(obs, wenv.state(), timestep=0, timesteps=0)
                action = out[-1].get("mean_actions", out[0])
        else:
            action = zero
        resid = (base._res_scale * action)[0].detach().cpu().numpy().copy()
        obs, _, term, tout, _ = wenv.step(action)

        # DirectRLEnv resets INSIDE step() when the episode ends, so on an ending
        # step the robot buffers already hold the post-reset (frame 0) state.
        # That sample must be discarded, not logged.
        if bool(term[0]) or bool(tout[0]):
            completed = bool(tout[0]) and not bool(term[0])
            ended_reason = ("clip end (time out)" if completed
                            else "TERMINATED (fall or root drift)")
            print(f"[[ episode ended after step {step+1} at reference frame {f0}"
                  f" -> {ended_reason}", flush=True)
            break

        q_act = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        q_ref_full = np.zeros_like(q_act)
        q_ref_full[isaac_of_ref] = ref_q[f0]
        rp = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        rq = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        r22 = 1.0 - 2.0 * (rq[1] ** 2 + rq[2] ** 2)
        bp = robot.data.body_pos_w[0].detach().cpu().numpy()
        bq = robot.data.body_quat_w[0].detach().cpu().numpy()
        pz = cm.paw_heights(bp, bq, body_names)

        L["frame"].append(f0)
        L["qerr21"].append(np.abs(q_act - q_ref_full))
        L["qerr_leg"].append(np.abs(q_act[leg_isaac] - q_ref_full[leg_isaac]))
        L["root_p"].append(rp); L["root_q"].append(rq)
        L["tilt"].append(np.degrees(np.arccos(np.clip(r22, -1, 1))))
        L["paw_z"].append(pz); L["contacts"].append(pz < CONTACT_H)
        L["paw_xy"].append(np.array([bp[knee_i[l]][:2] for l in LEGS]))
        try:
            L["torque"].append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        except Exception:
            L["torque"].append(np.zeros(len(names)))
        L["resid"].append(resid)

    D = {k: np.asarray(v) for k, v in L.items()}
    n = len(D["frame"])
    fr = D["frame"]
    rpe = np.linalg.norm(D["root_p"] - ref_root_p[fr], axis=1)
    rqe = quat_angle_deg(D["root_q"], ref_root_q[fr])
    tilt = D["tilt"]
    z0 = float(ref_root_p[0, 2])
    fallen = (D["root_p"][:, 2] < 0.5 * z0) | (tilt > 70.0)
    first_fall = int(np.where(fallen)[0][0]) if fallen.any() else -1
    tq_sat = np.abs(D["torque"]) >= (eff[None, :] - 1e-4)
    slip = []
    for i in range(1, n):
        pl = D["contacts"][i] & D["contacts"][i - 1]
        if pl.any():
            slip.append(np.linalg.norm(D["paw_xy"][i][pl] - D["paw_xy"][i - 1][pl], axis=1).max())

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"timid_v4_{LABEL}")
    np.savez(stem + ".npz", joint_names=np.array(names), **D)
    with open(stem + ".csv", "w") as f:
        f.write("frame,qerr21_max,qerr21_mean,qerr_leg_mean,root_pos_err_m,root_ori_err_deg,"
                "root_z,tilt_deg,n_contacts,resid_max,max_torque\n")
        for i in range(n):
            f.write(f"{fr[i]},{D['qerr21'][i].max():.6f},{D['qerr21'][i].mean():.6f},"
                    f"{D['qerr_leg'][i].mean():.6f},{rpe[i]:.6f},{rqe[i]:.4f},"
                    f"{D['root_p'][i,2]:.6f},{tilt[i]:.3f},{int(D['contacts'][i].sum())},"
                    f"{np.abs(D['resid'][i]).max():.6f},{np.abs(D['torque'][i]).max():.4f}\n")

    print(f"\n[[ ===== STAGE 5 EVAL [{LABEL}] frames {fr[0]}..{fr[-1]} ({n} steps) =====", flush=True)
    print(f"[[ completed clip        : {'YES' if completed else f'NO - {ended_reason} at frame {fr[-1]} of {T-1}'}", flush=True)
    print(f"[[ fallen/collapsed      : {'frame ' + str(first_fall) if first_fall >= 0 else 'no'}"
          f" | root z {D['root_p'][:,2].min():.3f}..{D['root_p'][:,2].max():.3f} m"
          f" | max tilt {tilt.max():.1f} deg", flush=True)
    print(f"[[ joint err (21 dof)    : mean {D['qerr21'].mean():.4f} rad  max {D['qerr21'].max():.4f} rad", flush=True)
    print(f"[[ joint err (12 legs)   : mean {D['qerr_leg'].mean():.4f} rad  max {D['qerr_leg'].max():.4f} rad", flush=True)
    print(f"[[ root position error   : mean {rpe.mean()*1000:.1f} mm  max {rpe.max()*1000:.1f} mm", flush=True)
    print(f"[[ root orientation err  : mean {rqe.mean():.2f} deg  max {rqe.max():.2f} deg", flush=True)
    print(f"[[ contacts per frame    : mean {D['contacts'].sum(1).mean():.2f} of 4"
          f" | frames with none {int((D['contacts'].sum(1)==0).sum())}", flush=True)
    ref_c = np.load(env_cfg.motion_file, allow_pickle=True)["contacts"][fr]
    print(f"[[ contact agreement     : {100.0*(D['contacts']==ref_c).mean():.1f}% of paw-frames match the reference", flush=True)
    if slip:
        print(f"[[ planted-paw slip      : mean {np.mean(slip)*1000:.2f} mm/frame  "
              f"max {np.max(slip)*1000:.2f} mm  total {np.sum(slip)*1000:.1f} mm", flush=True)
    print(f"[[ min paw z             : {D['paw_z'].min()*1000:+.1f} mm", flush=True)
    print(f"[[ residual magnitude    : mean {np.abs(D['resid']).mean():.4f} rad  "
          f"max {np.abs(D['resid']).max():.4f} rad  (authority {float(base._res_scale.max()):.2f})", flush=True)
    sat = [names[j] for j in range(len(names)) if tq_sat[:, j].mean() > 0.05]
    print(f"[[ torque-saturated >5%   : {sat if sat else 'none'}", flush=True)
    print(f"[[ wrote {stem}.npz and {stem}.csv", flush=True)

    # cross-check against the canonical Stage 4 run over the same frames
    if os.path.isfile(args.stage4_csv):
        c = np.genfromtxt(args.stage4_csv, delimiter=",", names=True)
        sl = slice(0, n)
        print(f"\n[[ --- canonical Stage 4 baseline over frames {fr[0]}..{fr[-1]} "
              f"(stage4/out/timid_v4_stage4.csv) ---", flush=True)
        print(f"[[   joint err mean {c['qerr_mean'][sl].mean():.4f} rad  max {c['qerr_max'][sl].max():.4f} rad", flush=True)
        print(f"[[   root pos err mean {c['root_pos_err_m'][sl].mean()*1000:.1f} mm  "
              f"max {c['root_pos_err_m'][sl].max()*1000:.1f} mm", flush=True)
        print(f"[[   root ori err mean {c['root_ori_err_deg'][sl].mean():.2f} deg  "
              f"max {c['root_ori_err_deg'][sl].max():.2f} deg", flush=True)
        print(f"[[   contacts mean {c['n_contacts'][sl].mean():.2f} | tilt max {c['tilt_deg'][sl].max():.1f} deg", flush=True)

    env.close()
    os._exit(0)


main()
