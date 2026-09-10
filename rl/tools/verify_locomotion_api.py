"""Verify every Isaac Lab API assumption the Bingo locomotion task makes.

The locomotion module was written without access to the Isaac Lab install (see
docs/locomotion/TASK1_DESIGN.md). Every assumption it makes about the local API is
checked here, so a mismatch is a named failure in this report rather than a stack
trace forty minutes into a training run.

Run this FIRST, on the training machine:

    cd ~/robotics/IsaacLab
    ./isaaclab.sh -p ~/Bingo/Blender/rl/tools/verify_locomotion_api.py --headless

Exits non-zero if anything fails. Nothing here trains, writes, or modifies any file.
"""
from __future__ import annotations

import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--strict", action="store_true",
               help="treat WARN as failure too")
AppLauncher.add_app_launcher_args(p)
args, _ = p.parse_known_args()
app = AppLauncher(args).app

import torch  # noqa: E402

RESULTS = []


def check(name, fn, warn_only=False):
    """Run one check; record PASS / FAIL / WARN with a detail string."""
    try:
        detail = fn()
        RESULTS.append(("PASS", name, detail if isinstance(detail, str) else ""))
        return True
    except Exception as e:  # noqa: BLE001 - this is a diagnostic harness
        RESULTS.append(("WARN" if warn_only else "FAIL", name,
                        f"{type(e).__name__}: {e}"))
        if not warn_only:
            traceback.print_exc(limit=2)
        return False


# =============================================================================
# 1. Module paths and stock classes
# =============================================================================
def _velocity_env_cfg():
    from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (  # noqa: F401
        LocomotionVelocityRoughEnvCfg, RewardsCfg, TerminationsCfg,
    )
    return f"{LocomotionVelocityRoughEnvCfg.__module__} (+RewardsCfg, TerminationsCfg)"


def _velocity_mdp():
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    return mdp.__name__


def _rsl_rl_cfgs():
    from isaaclab_rl.rsl_rl import (  # noqa: F401
        RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,
    )
    return "isaaclab_rl.rsl_rl"


def _actuator_cfgs():
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: F401
    return "IdealPDActuatorCfg, ImplicitActuatorCfg"


check("import velocity_env_cfg (LocomotionVelocityRoughEnvCfg/Rewards/Terminations)",
      _velocity_env_cfg)
check("import locomotion.velocity.mdp", _velocity_mdp)
check("import isaaclab_rl.rsl_rl PPO cfgs", _rsl_rl_cfgs)
check("import isaaclab.actuators PD cfgs", _actuator_cfgs)


# =============================================================================
# 2. mdp functions the config references by name
# =============================================================================
def _mdp_funcs():
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    needed = [
        "is_terminated", "feet_slide", "push_by_setting_velocity",
        "root_height_below_minimum", "bad_orientation", "terrain_levels_vel",
        "UniformVelocityCommandCfg",
    ]
    missing = [n for n in needed if not hasattr(mdp, n)]
    if missing:
        raise AttributeError(f"mdp is missing {missing}")
    return f"all {len(needed)} present"


def _command_class_type():
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    ct = mdp.UniformVelocityCommandCfg.class_type
    if not hasattr(ct, "_resample_command"):
        raise AttributeError(f"{ct.__name__} has no _resample_command to override")
    return f"{ct.__module__}.{ct.__name__}"


def _command_ranges_fields():
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    r = mdp.UniformVelocityCommandCfg.Ranges
    for f in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if f not in r.__annotations__ and not hasattr(r, f):
            raise AttributeError(f"Ranges has no {f}")
    return "lin_vel_x, lin_vel_y, ang_vel_z"


check("mdp functions referenced by the config exist", _mdp_funcs)
check("UniformVelocityCommandCfg.class_type is subclassable", _command_class_type)
check("UniformVelocityCommandCfg.Ranges fields", _command_ranges_fields)


# =============================================================================
# 3. The Bingo cfg builds, and its interfaces are what the design assumes
# =============================================================================
CFG = None


def _build_cfg():
    global CFG
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                           / "bingo_rl"))
    from bingo_rl.locomotion.bingo_velocity_env_cfg import BingoVelocityFlatEnvCfg
    CFG = BingoVelocityFlatEnvCfg()
    CFG.scene.num_envs = 4
    return "BingoVelocityFlatEnvCfg() constructed"


def _timing():
    if CFG.decimation != 5:
        raise ValueError(f"decimation {CFG.decimation} != 5")
    if abs(CFG.sim.dt - 1 / 120) > 1e-12:
        raise ValueError(f"sim.dt {CFG.sim.dt} != 1/120")
    hz = 1.0 / (CFG.sim.dt * CFG.decimation)
    if abs(hz - 24.0) > 1e-6:
        raise ValueError(f"control rate {hz} Hz != 24 Hz")
    return f"120 Hz physics / decimation 5 = {hz:.1f} Hz policy"


def _action_cfg():
    a = CFG.actions.joint_pos
    if not isinstance(a.scale, dict):
        raise TypeError(f"action scale is {type(a.scale).__name__}, expected a per-joint dict "
                        "- this Isaac Lab version may not support dict scales")
    if not getattr(a, "use_default_offset", False):
        raise ValueError("use_default_offset must be True for q = q_stand + scale*a")
    return f"joint_names={a.joint_names} scale={a.scale}"


check("build BingoVelocityFlatEnvCfg", _build_cfg)
if CFG is not None:
    check("control timing 120 Hz / dec 5 / 24 Hz", _timing)
    check("action term: per-joint dict scale + default offset", _action_cfg)


# =============================================================================
# 4. Instantiate the env and check the real shapes against the design
# =============================================================================
ENV = None


def _make_env():
    global ENV
    from isaaclab.envs import ManagerBasedRLEnv
    ENV = ManagerBasedRLEnv(cfg=CFG)
    return f"{ENV.num_envs} envs on {ENV.device}"


def _obs_dim():
    obs, _ = ENV.reset()
    dim = obs["policy"].shape[1]
    # 3 lin vel + 3 ang vel + 3 gravity + 3 command + 21 q + 21 qd + 12 actions
    expected = 3 + 3 + 3 + 3 + 21 + 21 + 12
    if dim != expected:
        raise ValueError(
            f"policy obs dim {dim} != {expected}. If it is {expected - 9 * 2} the "
            "joint terms are observing only the 12 actioned joints, which breaks "
            "Task 2; if larger, an extra term is enabled (height_scan?)")
    return f"policy obs = {dim} (all 21 joints observed)"


def _action_dim():
    n = ENV.action_manager.total_action_dim
    if n != 12:
        raise ValueError(f"action dim {n} != 12 (legs only)")
    return "action dim = 12"


def _joint_names():
    names = list(ENV.scene["robot"].data.joint_names)
    if len(names) != 21:
        raise ValueError(f"robot has {len(names)} joints, expected 21")
    canonical = [
        "fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
        "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
        "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
        "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
    ]
    missing = [n for n in canonical if n not in names]
    if missing:
        raise ValueError(f"missing joints {missing}")
    same = names == canonical
    return (f"21 joints, all canonical names present; Isaac order "
            f"{'matches' if same else 'DIFFERS from'} the npz order "
            f"(name mapping is required{'' if same else ' - as expected'})")


def _actuator_model():
    robot = ENV.scene["robot"]
    kinds = {k: type(v).__name__ for k, v in robot.actuators.items()}
    if not any("IdealPD" in v for v in kinds.values()):
        raise ValueError(f"expected IdealPDActuator, got {kinds}. Implicit actuators "
                         "do NOT enforce effort limits on this robot (measured: "
                         "3/8/15 N m gave byte-identical motion)")
    return str(kinds)


def _stance_pose():
    """The default pose must be STAND_SOLVED and inside the soft limits."""
    from bingo_rl.locomotion.bingo_velocity_env_cfg import STAND_SOLVED
    robot = ENV.scene["robot"]
    names = list(robot.data.joint_names)
    q = robot.data.default_joint_pos[0].cpu().numpy()
    worst = 0.0
    for jn, want in STAND_SOLVED.items():
        got = float(q[names.index(jn)])
        worst = max(worst, abs(got - want))
    if worst > 1e-3:
        raise ValueError(f"default joint pos differs from STAND_SOLVED by {worst:.4f} rad "
                         "- the rev_3 pose is probably still in place")
    lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    bad = [names[i] for i in range(len(names)) if q[i] < lo[i] - 1e-6 or q[i] > hi[i] + 1e-6]
    if bad:
        raise ValueError(f"stance pose outside soft limits at {bad}")
    return f"STAND_SOLVED applied (max dev {worst:.2e} rad), inside soft limits"


def _action_scale_headroom():
    """|action| = 3 must stay inside the soft limits for every leg joint."""
    from bingo_rl.locomotion.bingo_velocity_env_cfg import ACTION_SCALE
    import re
    robot = ENV.scene["robot"]
    names = list(robot.data.joint_names)
    q = robot.data.default_joint_pos[0].cpu().numpy()
    lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    worst, worst_j = 1e9, None
    for i, jn in enumerate(names):
        scale = next((s for pat, s in ACTION_SCALE.items() if re.fullmatch(pat, jn)), None)
        if scale is None:
            continue
        margin = min(hi[i] - (q[i] + 3 * scale), (q[i] - 3 * scale) - lo[i])
        if margin < worst:
            worst, worst_j = margin, jn
    if worst < -1e-3:
        raise ValueError(f"|action|=3 exceeds the soft limit at {worst_j} by "
                         f"{-worst:.4f} rad")
    return f"tightest margin at |a|=3 is {worst:+.4f} rad ({worst_j})"


def _command_categories():
    """Every category must actually be reachable, including exact zero."""
    cm = ENV.command_manager
    term = cm.get_term("base_velocity")
    n = 4000
    ids = torch.arange(ENV.num_envs, device=ENV.device)
    seen_zero = seen_turn = seen_lat = seen_back = 0
    for _ in range(n // max(1, ENV.num_envs)):
        term._resample_command(ids)
        c = term.command.clone()
        lin = torch.norm(c[:, :2], dim=1)
        seen_zero += int(((lin < 1e-6) & (c[:, 2].abs() < 1e-6)).sum())
        seen_turn += int(((lin < 1e-6) & (c[:, 2].abs() > 0.1)).sum())
        seen_lat += int(((c[:, 1].abs() > 0.05) & (c[:, 0].abs() < 1e-6)).sum())
        seen_back += int((c[:, 0] < -0.05).sum())
    for label, v in (("exact zero", seen_zero), ("turn-in-place", seen_turn),
                     ("lateral", seen_lat), ("backward", seen_back)):
        if v == 0:
            raise ValueError(f"category '{label}' never sampled in ~{n} draws")
    return (f"zero={seen_zero} turn-in-place={seen_turn} lateral={seen_lat} "
            f"backward={seen_back} of ~{n}")


def _step_env():
    """A zero action must run without NaN and must not instantly fall."""
    obs, _ = ENV.reset()
    z0 = float(ENV.scene["robot"].data.root_pos_w[0, 2])
    act = torch.zeros(ENV.num_envs, ENV.action_manager.total_action_dim,
                      device=ENV.device)
    for _ in range(48):  # 2 s at 24 Hz
        obs, rew, term, trunc, _ = ENV.step(act)
        if torch.isnan(obs["policy"]).any():
            raise ValueError("NaN in observations")
        if torch.isnan(rew).any():
            raise ValueError("NaN in rewards")
    z1 = float(ENV.scene["robot"].data.root_pos_w[0, 2])
    return f"48 steps at zero action, base z {z0:.4f} -> {z1:.4f} m (sank {1000*(z0-z1):+.1f} mm)"


if CFG is not None:
    if check("instantiate ManagerBasedRLEnv", _make_env) and ENV is not None:
        check("observation dimension == 66", _obs_dim)
        check("action dimension == 12", _action_dim)
        check("21 joints with canonical names", _joint_names)
        check("actuators are explicit IdealPD", _actuator_model)
        check("default pose is STAND_SOLVED and legal", _stance_pose)
        check("action scale leaves headroom at |a|=3", _action_scale_headroom)
        check("command categories all reachable", _command_categories)
        check("env steps cleanly at zero action", _step_env)


# =============================================================================
# Report
# =============================================================================
print("\n" + "=" * 78)
print("BINGO LOCOMOTION API VERIFICATION")
print("=" * 78)
w = max(len(n) for _, n, _ in RESULTS) if RESULTS else 10
for status, name, detail in RESULTS:
    mark = {"PASS": "ok  ", "FAIL": "FAIL", "WARN": "warn"}[status]
    print(f"[{mark}] {name:<{w}}  {detail}")
n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
print("-" * 78)
print(f"{len(RESULTS) - n_fail - n_warn} passed, {n_warn} warned, {n_fail} failed")
if n_fail:
    print("\nFix the config against the LOCAL Isaac Lab API. Do not adjust these checks\n"
          "to make them pass - they encode what the design in\n"
          "docs/locomotion/TASK1_DESIGN.md assumes.")

if ENV is not None:
    ENV.close()
app.close()
sys.exit(1 if (n_fail or (args.strict and n_warn)) else 0)
