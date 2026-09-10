"""Bingo-specific MDP terms for command-conditioned locomotion (Task 1).

Everything here is either (a) a term the stock Isaac Lab velocity task does not
provide, or (b) a Bingo-specific variant with its justification attached. Stock
terms are used unchanged from
``isaaclab_tasks.manager_based.locomotion.velocity.mdp`` and are NOT re-implemented.

Nothing in this module touches the Stage 1-5 animation pipeline or the validated v4
physics model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Command: categorical velocity sampling
# =============================================================================
# The stock UniformVelocityCommand samples each of vx, vy, yaw_rate independently
# and uniformly. Two cases the brief requires explicitly then essentially never
# occur in training:
#
#   exact zero      P(|v| < eps) ~ 0 under a continuous uniform, so "stand still"
#                   is never actually trained, only approached.
#   turn in place   requires vx AND vy to be ~0 while yaw_rate is large - the
#                   product of two near-zero draws, i.e. vanishingly rare.
#
# So sample a CATEGORY first, then values within it. This also lets the curriculum
# switch categories on and off by weight, which is how stages 1-6 are implemented.
#
# The base class is resolved through the local cfg's own ``class_type`` rather than
# imported by path: the module layout of the velocity command has moved between
# Isaac Lab versions, but the cfg always names its own implementation, so this picks
# up whatever the locally installed version uses.
_UniformVelocityCommand = mdp.UniformVelocityCommandCfg.class_type

# Category order is fixed so that a weight vector is unambiguous.
CATEGORIES = (
    "stand",           # exact [0, 0, 0]
    "forward",
    "backward",
    "lateral",
    "diagonal",
    "turn_in_place",   # exact zero linear, non-zero yaw
    "turn_and_move",
)


class BingoVelocityCommand(_UniformVelocityCommand):
    """Velocity command that samples a movement category, then values within it.

    ``_resample_command`` defers to the base implementation first so that whatever
    per-version bookkeeping it does (standing-env mask, heading resampling, metric
    buffers) still happens, then overwrites the numbers. That keeps this subclass
    robust to the base class gaining fields.
    """

    cfg: BingoVelocityCommandCfg

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        dev = self.device
        w = torch.tensor([max(0.0, float(self.cfg.category_weights.get(c, 0.0)))
                          for c in CATEGORIES], device=dev)
        if float(w.sum()) <= 0.0:
            return  # no categories enabled: leave the base class's uniform sample
        cat = torch.multinomial(w / w.sum(), n, replacement=True)

        def uni(lo, hi, mask):
            """Uniform in [lo, hi] for the masked entries."""
            return torch.empty(int(mask.sum()), device=dev).uniform_(lo, hi)

        def signed(lo, hi, mask):
            """Uniform magnitude in [lo, hi] with a random sign."""
            m = torch.empty(int(mask.sum()), device=dev).uniform_(lo, hi)
            s = torch.randint(0, 2, (int(mask.sum()),), device=dev) * 2 - 1
            return m * s

        vx = torch.zeros(n, device=dev)
        vy = torch.zeros(n, device=dev)
        wz = torch.zeros(n, device=dev)
        r = self.cfg.category_ranges

        m = cat == CATEGORIES.index("forward")
        if m.any():
            vx[m] = uni(*r["forward_vx"], m)

        m = cat == CATEGORIES.index("backward")
        if m.any():
            vx[m] = -uni(*r["backward_vx"], m)

        m = cat == CATEGORIES.index("lateral")
        if m.any():
            vy[m] = signed(*r["lateral_vy"], m)

        m = cat == CATEGORIES.index("diagonal")
        if m.any():
            vx[m] = signed(*r["diagonal_vx"], m)
            vy[m] = signed(*r["diagonal_vy"], m)

        m = cat == CATEGORIES.index("turn_in_place")
        if m.any():
            wz[m] = signed(*r["turn_wz"], m)

        m = cat == CATEGORIES.index("turn_and_move")
        if m.any():
            vx[m] = uni(*r["turn_move_vx"], m)
            wz[m] = signed(*r["turn_move_wz"], m)

        # "stand" leaves all three at exactly zero, which is the point of it.
        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 1] = vy
        self.vel_command_b[env_ids, 2] = wz


@configclass
class BingoVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Config for :class:`BingoVelocityCommand`.

    ``ranges`` from the base class is still honoured by the base implementation but
    is immediately overwritten by the categorical draw, so it only matters as a
    fallback when every category weight is zero.
    """

    class_type: type = BingoVelocityCommand

    # Shares are relative; they are normalised before sampling. Setting one to 0
    # disables that category, which is how the training curriculum stages work.
    category_weights: dict[str, float] = {
        "stand": 0.10,
        "forward": 0.25,
        "backward": 0.10,
        "lateral": 0.10,
        "diagonal": 0.10,
        "turn_in_place": 0.15,
        "turn_and_move": 0.20,
    }

    # Magnitude ranges, sized for a 0.18 m / 2.5 kg quadruped rather than for a Go1.
    # The measured walking speed of every authored Bingo clip is 0.06-0.14 m/s
    # (rl/tools/analyze_style_motions.py), so these ask for meaningfully more than
    # the animation does while staying far below the Froude-3 ceiling of ~2.3 m/s.
    category_ranges: dict[str, tuple[float, float]] = {
        "forward_vx": (0.10, 0.40),
        "backward_vx": (0.10, 0.25),
        "lateral_vy": (0.10, 0.25),
        "diagonal_vx": (0.10, 0.30),
        "diagonal_vy": (0.10, 0.20),
        "turn_wz": (0.40, 1.20),
        "turn_move_vx": (0.10, 0.35),
        "turn_move_wz": (0.30, 1.00),
    }


# =============================================================================
# Rewards
# =============================================================================
# air_time_over_limit and contact_time_over_limit are carried over verbatim from
# bingo_rl/improved_walking_cfg.py, where they were written to kill two failure
# modes MEASURED on this robot:
#   v3  held one leg permanently in the air and dragged on three (tripod limp).
#       feet_air_time only *rewards* feet that touch down, so a leg that never
#       comes down is never punished by it.
#   v6  parked the front-left foot on the ground 85% of the time as a static prop
#       while the other three stepped.
# Those are properties of Bingo, not of that config, so they come along. They are
# re-declared here rather than imported so this module does not depend on a rev_3
# configuration.


def air_time_over_limit(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                        max_air_time: float) -> torch.Tensor:
    """Charge a foot for staying airborne longer than ``max_air_time`` seconds."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(air - max_air_time, min=0.0), dim=1)


def contact_time_over_limit(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                            max_contact_time: float) -> torch.Tensor:
    """Charge a foot for staying grounded longer than ``max_contact_time`` seconds."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_t = cs.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.sum(torch.clamp(contact_t - max_contact_time, min=0.0), dim=1)


def stand_still_on_zero_command(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise joint deviation from the stance pose while the command is zero.

    The brief requires the exact-zero command to be trained, not merely approached.
    Velocity tracking alone does not do it: a policy that marches on the spot scores
    a perfect tracking reward at zero command, because its *base* velocity is zero.
    This charges the pose deviation instead, so standing still is cheaper than
    marching. Returns 0 for every env whose command is non-zero, so it never
    competes with locomotion.
    """
    cmd = env.command_manager.get_command(command_name)
    is_zero = (torch.norm(cmd[:, :3], dim=1) < command_threshold).float()
    asset: Articulation = env.scene[asset_cfg.name]
    dev = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]
                  - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return dev * is_zero


def joint_position_limit_margin(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared violation of the SOFT joint limits, summed over the given joints.

    The stock ``dof_pos_limits`` term does the same thing; this exists so the term
    can be pointed at the leg joints only. It matters on Bingo because SY has a
    +-0.42 rad range against +-1.56 for every other leg joint, so a limit penalty
    averaged over all 21 joints is dominated by the joints that are nowhere near
    their stops.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lo = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    hi = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out = (lo - q).clamp(min=0.0) + (q - hi).clamp(min=0.0)
    return torch.sum(out.square(), dim=1)
