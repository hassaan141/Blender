"""Export a trained Bingo velocity policy (Task 1, locomotion only) to ONNX,
plus a manifest describing exactly how to drive it from outside Isaac Lab
(a browser MuJoCo runtime, in particular).

Locomotion only. This script has nothing to do with the Stage 1-5 animation
pipeline or the expression (head/tail/ear) layer -- it exports the 12-leg
command-conditioned policy from rl/bingo_rl/bingo_rl/locomotion/.

    cd /pub0/muhammadf/IsaacLab
    ./isaaclab.sh -p /pub0/muhammadf/Blender/rl/tools/export_onnx.py \
        --checkpoint <run>/model_XXXX.pt \
        --out /pub0/muhammadf/Blender/exports/<name> --headless

Why a manifest and not just the .onnx: the ONNX graph only knows "66 floats
in, 12 floats out." It says nothing about WHICH 66 floats, in what order, at
what scale, or which of the 12 outputs drives which joint. Isaac Lab resolves
joint name -> index at runtime and is NOT guaranteed to match the URDF's own
authoring order (v4_kinematics.DOF_ORDER) -- TASK1_DESIGN.md already flags
this ("Isaac orders DOFs breadth-first, which is not this order"). Any
external runtime that assumes DOF_ORDER without checking will not crash, it
will just apply the wrong action to the wrong joint and produce a policy that
"looks bad" for a reason that has nothing to do with training. This script
queries Isaac Lab's OWN resolved order at runtime and writes THAT into the
manifest, then separately reports whether it happens to match DOF_ORDER --
never assumes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a Bingo velocity policy to ONNX.")
parser.add_argument("--task", type=str, default="Bingo-Velocity-Flat-v4-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True,
                     help="output directory; writes policy.onnx + manifest.json here")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_verify", type=int, default=64,
                     help="random observations used for the PyTorch-vs-ONNX numeric check")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

_INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rl" / "bingo_rl"))
sys.path.insert(0, str(REPO_ROOT / "stage2"))
import bingo_rl  # noqa: F401,E402  (registers the Bingo tasks)
from bingo_rl.locomotion import bingo_velocity_env_cfg as vcfg  # noqa: E402
from v4_kinematics import DOF_ORDER  # noqa: E402  (canonical URDF-authoring order, for comparison only)


def term_dims(dims_list):
    out = []
    for d in dims_list:
        out.append(int(np.prod(d)))
    return out


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"[[ loaded {args_cli.checkpoint}")

    uenv = env.unwrapped
    om = uenv.observation_manager
    am = uenv.action_manager
    robot = uenv.scene["robot"]

    # ---- ground truth: Isaac Lab's OWN resolved order, not an assumption ----
    obs_term_names = om.active_terms["policy"]
    obs_term_sizes = term_dims(om.group_obs_term_dim["policy"])
    obs_dim_total = int(np.prod(om.group_obs_dim["policy"]))
    obs_layout = []
    off = 0
    for name, d in zip(obs_term_names, obs_term_sizes):
        obs_layout.append({"name": name, "dim": d, "start": off, "end": off + d})
        off += d

    action_term = am.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)  # resolved at runtime; no public accessor
    action_dim = am.total_action_dim
    scale_t = action_term._scale
    if isinstance(scale_t, torch.Tensor):
        action_scale = {n: float(v) for n, v in zip(action_joint_names, scale_t[0].tolist())}
    else:
        action_scale = {n: float(scale_t) for n in action_joint_names}

    full_joint_names = list(robot.data.joint_names)  # order used by joint_pos_rel / joint_vel_rel

    dof_order_full = list(DOF_ORDER)
    dof_order_legs = dof_order_full[:12]
    full_matches_dof_order = full_joint_names == dof_order_full
    legs_match_dof_order = action_joint_names == dof_order_legs

    stance_pose = {**vcfg.STAND_SOLVED, **vcfg.EXPR_NEUTRAL}

    # ---- export ----
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(out_dir), filename="policy.onnx")
    onnx_path = out_dir / "policy.onnx"
    print(f"[[ wrote {onnx_path.resolve()}")

    # ---- numeric verification: same random obs through PyTorch and ONNX ----
    import onnxruntime as ort

    policy = runner.get_inference_policy(device="cpu")
    onnx_wrapper = policy.as_onnx(verbose=False)
    onnx_wrapper.to("cpu")
    onnx_wrapper.eval()

    torch.manual_seed(0)
    test_obs = torch.randn(args_cli.num_verify, obs_dim_total)
    with torch.inference_mode():
        torch_out = onnx_wrapper(test_obs).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_in_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_in_name: test_obs.numpy().astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    onnx_matches = max_diff < 1e-5
    print(f"[[ onnx vs pytorch max abs diff: {max_diff:.3e} over {args_cli.num_verify} random obs "
          f"({'PASS' if onnx_matches else 'FAIL'} <1e-5)")

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
        },
        "control": {"physics_hz": 120, "decimation": 5, "control_hz": 24},
        "observation": {
            "total_dim": obs_dim_total,
            "terms": obs_layout,
            "note": "concatenate terms in this exact order to build the 'obs' input",
            "full_joint_order_for_joint_pos_rel_and_joint_vel_rel": full_joint_names,
        },
        "action": {
            "dim": action_dim,
            "joint_order": action_joint_names,
            "scale_rad_per_unit_action": action_scale,
            "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
            "note": "q_target[joint] = stance_pose_rad[joint] + scale[joint] * onnx_output[joint]",
        },
        "stance_pose_rad": stance_pose,
        "canonical_dof_order_reference": {
            "source": "stage2/v4_kinematics.DOF_ORDER (== bake_conform.DOF_ORDER_21)",
            "full_21": dof_order_full,
            "legs_12": dof_order_legs,
            "warning": "This is the URDF-authoring order used by the Blender/retargeting "
                       "pipeline. It is NOT guaranteed to match Isaac Lab's runtime order "
                       "below -- always use observation/action orders above, never this one, "
                       "to build the actual obs/action vectors.",
        },
        "joint_order_check": {
            "full_obs_order_matches_DOF_ORDER": full_matches_dof_order,
            "action_order_matches_DOF_ORDER_legs": legs_match_dof_order,
        },
        "onnx_verification": {
            "max_abs_diff_vs_pytorch": max_diff,
            "passes_1e-5": onnx_matches,
            "num_random_obs_tested": args_cli.num_verify,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[[ wrote {manifest_path.resolve()}")
    print(f"[[ full-obs joint order matches DOF_ORDER : {full_matches_dof_order}")
    print(f"[[ action joint order matches DOF_ORDER[:12] : {legs_match_dof_order}")
    print(f"[[ ONNX numeric match <1e-5 : {onnx_matches}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
