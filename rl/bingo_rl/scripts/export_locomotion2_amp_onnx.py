"""Export a trained Bingo Locomotion 2 (dog-style AMP) policy to ONNX + manifest.

Sibling of rl/tools/export_walk_ref_onnx.py (same stack: DirectRLEnv, skrl-trained
GaussianMixin policy) -- reuses its exact ONNXPolicy wrapper pattern (bake the
trained RunningStandardScaler observation preprocessor into the graph). Differs
from that script in one important way: this env is velocity-COMMANDED, not
phase/reference-driven, so there is no q_ref(phase)/residual-filter section in
the manifest -- q_target[joint] = default_joint_pos[joint] + scale[joint] * onnx_output[joint].

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p /pub0/muhammadf/Blender/rl/bingo_rl/scripts/export_locomotion2_amp_onnx.py \
        --checkpoint <run>/checkpoints/agent_XXXX.pt \
        --out /pub0/muhammadf/Blender/docs/locomotion2/amp/export --headless
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

parser = argparse.ArgumentParser(description="Export a Bingo Locomotion2 AMP policy to ONNX.")
parser.add_argument("--task", type=str, default="Bingo-Locomotion2-AMP-Direct-Play-v0")
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "rl" / "bingo_rl"))
import bingo_rl  # noqa: F401,E402
from bingo_rl.locomotion2_amp.bingo_dog_amp_env_cfg import ACTION_SCALE, DOF_ORDER  # noqa: E402


class ONNXPolicy(torch.nn.Module):
    """Deterministic (mean-action) wrapper around an skrl GaussianMixin policy,
    with the trained observation preprocessor baked into the graph -- see
    rl/tools/export_walk_ref_onnx.py's identical class for the full rationale."""

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
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
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
    onnx_path = out_dir / "bingo_locomotion2_amp.onnx"

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
                base.commands[:, 0] = 0.25
                base.commands[:, 1] = 0.0
                base.commands[:, 2] = 0.0
                if use_onnx:
                    o = sess.run(None, {onnx_in_name: obs.cpu().numpy().astype(np.float32)})[0]
                    action = torch.as_tensor(o, device=obs.device)
                else:
                    action = onnx_wrapper(obs.cpu()).to(obs.device)
                obs, _, term, tout, _ = wenv.step(action)
                vx_trace.append(float(base.robot.data.root_lin_vel_b[0, 0]))
        return np.array(vx_trace)

    vx_pt = rollout(use_onnx=False)
    vx_onnx = rollout(use_onnx=True)
    rollout_diff = float(np.max(np.abs(vx_pt - vx_onnx)))
    print(f"[[ rollout vx trace max abs diff (pytorch vs onnx-driven): {rollout_diff:.4f} m/s "
          f"over {len(vx_pt)} steps")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    joint_names = list(DOF_ORDER)
    default_pose = {n: float(base.robot.data.default_joint_pos[0, base.robot_ctrl_indexes[i]])
                     for i, n in enumerate(joint_names)}
    action_scale = {n: float(base.action_scale[i]) for i, n in enumerate(joint_names)}

    manifest = {
        "provenance": {
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "task": args_cli.task,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "repo_commit": commit,
            "expert_data": "docs/locomotion2/amp/cache/dog_amp_expert.npz (dog mocap style prior, "
                            "AMP discriminator only -- not tracked directly by this policy)",
        },
        "control": {"physics_hz": 120, "decimation": int(env_cfg.decimation),
                     "control_hz": 120 // int(env_cfg.decimation)},
        "observation": {
            "total_dim": obs_dim,
            "order": [
                {"name": "style_features", "dim": 61,
                 "note": "see docs/locomotion2/amp/extract_dog_amp_features.py's schema docstring "
                         "for the exact 61-dim layout (root-local vel/leg_scale, ang_vel, gravity, "
                         "4x2 leg segment unit dirs, paw_rel/leg_scale, paw_vel/leg_scale, contacts)"},
                {"name": "command_vx_vy_yaw", "dim": 3, "note": "m/s, m/s, rad/s"},
            ],
            "note": "this is command-CONDITIONED but style-feature-based, not the raw dof_pos/vel "
                    "observation Task 1 uses -- do not assume the two exports share a layout.",
        },
        "action": {
            "dim": action_dim,
            "joint_order": joint_names,
            "scale_rad_per_unit_action": action_scale,
            "default_pose_rad": default_pose,
            "note": "q_target[joint] = default_pose_rad[joint] + scale[joint] * onnx_output[joint]. "
                    "No reference clip, no phase, no residual filter -- this policy is a direct "
                    "velocity-command controller (unlike walk_ref's residual-on-reference).",
        },
        "pd_gains_expected_by_runtime": {
            "source": "rl/bingo_rl/bingo_rl/bingo_v4.py (BINGO_V4_CFG, Stage 4 validated physics, unmodified)",
            "legs": {"Kp": {"SY": 40, "SP": 120, "knee": 120}, "effort_limit_Nm": 3.0},
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
