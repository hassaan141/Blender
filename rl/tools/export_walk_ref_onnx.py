"""Export a trained Bingo-WalkRef policy (reference-guided residual, skrl PPO) to
ONNX plus a manifest, for a browser/runtime that keeps the Stage 4 reference
separately and only asks the network for the residual:

    keyboard command -> phase + robot observation -> ONNX residual policy
        -> q_ref(phase) + residual -> PD control

Sibling of rl/tools/export_onnx.py (Task 1's from-scratch locomotion exporter),
adapted for this env's different stack: DirectRLEnv (not manager-based) and an
skrl-trained GaussianMixin policy (not rsl_rl's OnPolicyRunner) -- so the manifest
covers the same ground (observation order, joint order, action scale, control
rate) but the loading/export mechanics are necessarily different; see the class
docstrings in bingo_walk_ref_env.py / bingo_walk_ref_env_cfg.py for what each
manifest field actually means for this env.

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p /pub0/muhammadf/Blender/rl/tools/export_walk_ref_onnx.py \
        --checkpoint <run>/agent_XXXX.pt \
        --out /pub0/muhammadf/Blender/exports/bingo_walk_reference --headless
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a Bingo walk-ref policy to ONNX.")
parser.add_argument("--task", type=str, default="Bingo-WalkRef-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_verify", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rl" / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import NOMINAL_VX  # noqa: E402


class ONNXPolicy(torch.nn.Module):
    """Deterministic (mean-action) wrapper around an skrl GaussianMixin policy.

    Bakes the trained RunningStandardScaler observation preprocessor INTO the
    graph (skrl_ppo_cfg.yaml sets `state_preprocessor: RunningStandardScaler`,
    auto-remapped to `agent._observation_preprocessor`) so the exported ONNX
    takes raw, unnormalized observations -- the runtime should not have to
    reimplement skrl's running-mean/var normalization separately.

    NOTE (this session, measured): the policy network's YAML `input: OBSERVATIONS`
    reads `inputs["observations"]`, NOT `inputs["states"]` -- skrl's PPO.act()
    builds `{"observations": obs_preprocessor(obs), "states": state_preprocessor(states)}`
    and "states" is the (unused here) recurrent/critic-state slot, which is None
    for this non-recurrent policy. Passing obs under "states" instead of
    "observations" silently reaches the first Linear layer as None.
    """

    def __init__(self, policy, obs_preprocessor):
        super().__init__()
        self.policy = policy
        self.obs_preprocessor = obs_preprocessor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_preprocessor(obs)
        mean, _ = self.policy.compute({"observations": normalized, "states": None}, role="policy")
        return mean


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wenv = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    from skrl.utils.runner.torch import Runner
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(wenv, agent_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    print(f"[[ loaded {args_cli.checkpoint}")

    policy = runner.agent.policy
    policy.to("cpu")
    policy.eval()
    obs_preprocessor = runner.agent._observation_preprocessor
    obs_preprocessor.to("cpu")
    obs_preprocessor.eval()
    onnx_wrapper = ONNXPolicy(policy, obs_preprocessor)
    onnx_wrapper.eval()

    obs_dim = int(env_cfg.observation_space)
    action_dim = int(env_cfg.action_space)

    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "bingo_walk_reference.onnx"

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        onnx_wrapper, dummy, str(onnx_path),
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"[[ wrote {onnx_path.resolve()}")

    # ---- numeric verification: same random obs through PyTorch and ONNX ----
    import onnxruntime as ort

    torch.manual_seed(0)
    test_obs = torch.randn(args_cli.num_verify, obs_dim)
    with torch.inference_mode():
        torch_out = onnx_wrapper(test_obs).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_in_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_in_name: test_obs.numpy().astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    onnx_matches = max_diff < 1e-4
    print(f"[[ onnx vs pytorch max abs diff: {max_diff:.3e} over {args_cli.num_verify} random obs "
          f"({'PASS' if onnx_matches else 'FAIL'} <1e-4)")

    # ---- rollout sanity check: drive a real episode through ONNX, compare to PyTorch ----
    def rollout(use_onnx: bool, steps: int = 96):
        obs, _ = wenv.reset()
        vx_trace = []
        with torch.inference_mode():
            for _ in range(steps):
                base._cmd_vx[:] = float(NOMINAL_VX)
                base._cmd_timer[:] = 1e6
                if use_onnx:
                    o = sess.run(None, {onnx_in_name: obs.cpu().numpy().astype(np.float32)})[0]
                    action = torch.as_tensor(o, device=obs.device)
                else:
                    action = onnx_wrapper(obs.cpu()).to(obs.device)
                obs, _, term, tout, _ = wenv.step(action)
                root_quat = base.robot.data.body_quat_w[:, base.ref_body_index]
                root_lin_w = base.robot.data.body_lin_vel_w[:, base.ref_body_index]
                from isaaclab.utils.math import quat_rotate_inverse
                vx_trace.append(float(quat_rotate_inverse(root_quat, root_lin_w)[0, 0]))
        return np.array(vx_trace)

    vx_pt = rollout(use_onnx=False)
    vx_onnx = rollout(use_onnx=True)
    rollout_diff = float(np.max(np.abs(vx_pt - vx_onnx)))
    print(f"[[ rollout vx trace max abs diff (pytorch vs onnx-driven): {rollout_diff:.4f} m/s "
          f"over {len(vx_pt)} steps")

    # ---- manifest ----
    joint_names = list(base.ctrl_dof_names)  # == LEG_JOINT_NAMES, Isaac-resolved order
    action_scale = {n: float(base._res_scale[i]) for i, n in enumerate(joint_names)}
    default_pose = {n: float(base._stand_pose[0, i]) for i, n in enumerate(joint_names)}

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    manifest = {
        "provenance": {
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "task": args_cli.task,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "repo_commit": commit,
            "reference_clip": "motions/bingo_walk_v4_upright_grounded_loopsmooth.npz "
                               "(loop-seam-smoothed derivative of the canonical "
                               "bingo_walk_v4_upright_grounded.npz; see stage4/smooth_loop_seam.py)",
        },
        "residual_filter": {
            "note": "The ONNX graph outputs the RAW policy action every call (stateless). "
                    "The runtime MUST apply this EMA low-pass to it before scaling by "
                    "residual_scale and adding to q_ref -- training applies the identical "
                    "filter (bingo_walk_ref_env.py _pre_physics_step) to what actually "
                    "drives the robot, so skipping this step at inference will not "
                    "reproduce the trained behavior.",
            "formula": "filtered[t] = alpha*raw_action[t] + (1-alpha)*filtered[t-1]; "
                       "filtered[-1] = 0 at episode/session start",
            "alpha": float(getattr(env_cfg, "residual_ema_alpha", 0.5)),
        },
        "control": {"physics_hz": 120, "decimation": int(env_cfg.decimation),
                     "control_hz": 120 // int(env_cfg.decimation)},
        "observation": {
            "total_dim": obs_dim,
            "order": [
                {"name": "leg_dof_pos", "dim": 12, "note": "joint_order below, radians"},
                {"name": "leg_dof_vel", "dim": 12, "note": "joint_order below, rad/s"},
                {"name": "expr_dof_pos", "dim": 9, "note": "head/tail/ear, radians (feed-forward only, informational)"},
                {"name": "expr_dof_vel", "dim": 9, "note": "head/tail/ear, rad/s"},
                {"name": "root_height", "dim": 1},
                {"name": "root_quat_tangent_norm", "dim": 6},
                {"name": "root_lin_vel_body", "dim": 3},
                {"name": "root_ang_vel_body", "dim": 3},
                {"name": "foot_tips_local", "dim": 12, "note": "4 feet x xyz, root frame"},
                {"name": "phase_sin_cos", "dim": 2},
                {"name": "commanded_vx_normalized", "dim": 1, "note": "cmd_vx / NOMINAL_VX"},
            ],
            "note": "exact per-dim composition is bingo_rl.amp.bingo_amp_env.compute_obs "
                    "(proprio) + [sin(phase),cos(phase),cmd_vx/NOMINAL_VX]; verify against "
                    "that function before hand-porting, this listing is descriptive only.",
        },
        "action": {
            "dim": action_dim,
            "joint_order": joint_names,
            "residual_scale_rad": action_scale,
            "default_pose_rad": default_pose,
            "note": "q_target[joint] = q_ref(phase)[joint] + residual_scale[joint] * "
                    "ema_filter(onnx_output[joint])  -- see residual_filter above, apply it "
                    "BEFORE scaling, not after. q_ref(phase) must come from the runtime's own "
                    "copy of the reference clip (see provenance.reference_clip), NOT from the "
                    "network. At "
                    "|cmd_vx| below stand_blend_vx_frac*NOMINAL_VX, blend q_ref toward "
                    "default_pose_rad (see phase_update_rule below) instead of using the raw "
                    "cyclic reference -- this is what makes zero command produce standing.",
        },
        "phase_update_rule": {
            "nominal_vx_m_s": float(NOMINAL_VX),
            "clip_duration_s": float(base.motion_duration),
            "rule": "phase_time_s += control_dt_s * (cmd_vx / nominal_vx_m_s); "
                    "phase_time_s %= clip_duration_s; phase = phase_time_s / clip_duration_s",
            "stand_blend": {
                "stand_blend_vx_frac": float(base.cfg.stand_blend_vx_frac),
                "rule": "w = clip(|cmd_vx| / (stand_blend_vx_frac*nominal_vx_m_s), 0, 1); "
                        "q_ref_used = w * q_ref_cyclic(phase) + (1-w) * default_pose_rad",
            },
        },
        "expressive_joints_feedforward": {
            "note": "head/tail/ear (9 DOF) are NOT policy outputs; feed them q_ref(phase) "
                    "directly from the reference clip's own head_tail_positions/ear_positions "
                    "fields, same phase as the legs.",
        },
        "pd_gains_expected_by_runtime": {
            "source": "rl/bingo_rl/bingo_rl/bingo_v4.py (BINGO_V4_CFG, Stage 4 validated physics)",
            "legs": {"Kp": {"SY": 40, "SP": 120, "knee": 120}, "effort_limit_Nm": 3.0},
            "note": "see bingo_v4.py for the full derivation and head/tail/ear gains; this "
                    "policy assumes exactly these actuator dynamics (IdealPDActuator, first-"
                    "order-hold reference across the control-decimation window).",
        },
        "onnx_verification": {
            "max_abs_diff_vs_pytorch_random_obs": max_diff,
            "passes_1e-4": onnx_matches,
            "num_random_obs_tested": args_cli.num_verify,
            "rollout_vx_trace_max_abs_diff_m_s": rollout_diff,
            "rollout_steps": 96,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[[ wrote {manifest_path.resolve()}")
    print(f"[[ ONNX numeric match <1e-4 : {onnx_matches}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
