"""Config for the reference-guided walk env: residual RL on a CYCLIC, speed-scaled
Stage 4 walking reference (vs Stage 5's single-pass, non-cyclic Timid reference).

Inherited unchanged from bingo_rl.track_v4.bingo_track_v4_env_cfg.BingoTrackV4EnvCfg /
bingo_rl.track.bingo_track_env_cfg.BingoTrackEnvCfg:
  * robot = BINGO_V4_CFG (IdealPD actuators, Stage 4 gains) -- physics unchanged
  * reference_body "origin", KEY_BODY_NAMES = the four knee/shank tips
  * action_space 12 (legs only; head/tail/ears stay feed-forward, see bingo_walk_ref_env.py)

Changed:
  observation_space 69 -> 70
      proprio(67) + phase(sin,cos)=2 + commanded-vx(1) = 70. The extra scalar is the
      normalized forward-speed command the policy must track (see class docstring
      in bingo_walk_ref_env.py for why phase itself is no longer wall-clock).
  decimation -> 5 (24 Hz control, matches the validated Stage 4 baseline rate)
  residual_scale = 0.3
      taken verbatim from Stage 5 / BingoTrackResidualEnvCfg (the project's best
      tracker result) -- not re-tuned for this first pass, per the task's scope.
  motion_file -> motions/bingo_walk_v4_upright_grounded.npz
      the ground-fixed conform-rig walk clip, pushed through the real Stage 4
      physics validator (rl/tools/track_v4_physics.py) this session: completes
      136/136 frames without falling (max tilt 13.3 deg vs ref 17.2 deg), joint
      tracking mean 0.0103/max 0.34 rad, root position mean 202/max 418 mm, root
      orientation mean 9.3/max 16.7 deg -- comparable to or better than the
      already-accepted Timid clip (Stage 5's reference: max root error 611 mm).
      br_knee/bl_knee/fr_knee/*_SP_J torque-saturate 5-28% of frames under raw PD
      tracking alone; this is exactly the gap the RL residual is meant to close.
      See stage4/out/bingo_walk_v4_upright_grounded_stage4.{npz,csv} for the audit.
  ear_file -> same npz (carries ear_positions, like Timid's convention)

Three curriculum stages (A/B/C), per the task's explicit scope -- forward only, no
reverse/strafe/turn/domain-randomization:
  A: fixed nominal Stage 4 speed (no command variation, no standing)
  B: small forward-velocity range around nominal
  C: + standing and start/stop transitions (command resampled mid-episode)
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from bingo_rl.track_v4.bingo_track_v4_env_cfg import BingoTrackV4EnvCfg

# .../Blender/rl/bingo_rl/bingo_rl/walk_ref/this_file.py -> parents[4] == Blender/
BLENDER_ROOT = Path(__file__).resolve().parents[4]
# Loop-seam-smoothed derivative of the canonical bingo_walk_v4_upright_grounded.npz
# (stage4/smooth_loop_seam.py; canonical file itself untouched). Shaking diagnostic
# (rl/tools/diagnose_walk_ref_shaking.py) measured the raw clip's frame[-1]->frame[0]
# wrap as a 0.33 rad single-joint jump -- by far the single largest per-step
# reference jump anywhere in the clip, and a genuine defect (the clip is a single
# authored pass, never designed to loop). Retiming (uniformly slowing the whole
# clip 1.1x/1.2x/1.3x, PD-only-validated via track_v4_physics.py this session) was
# REJECTED: it reduced saturated-joint COUNT slightly but made the open-loop
# baseline fall outright (max tilt 97-119 deg vs 13.3 deg at 1.0x) -- this gait's
# balance depends on its authored timing, so slowing it made things worse, not
# better, confirming the reference is fine at its native speed and the defect is
# specifically the loop seam. Seam-smoothing (window=30 frames each side) removes
# the 0.33 rad jump (-> 0.0 rad) without introducing a fall, and drops peak joint
# tracking error 0.34->0.18 rad in the PD-only check.
WALK_V4 = str(BLENDER_ROOT / "motions" / "bingo_walk_v4_upright_grounded_loopsmooth.npz")

_DECIMATION = 5  # 120 Hz physics / 5 = 24 Hz control, matches Stage 4's own rate

# Measured (this session) from bingo_walk_v4_upright_grounded.npz: root xy moved
# 0.748 m over 136 frames / 120 fps = 1.1333 s -> 0.660 m/s average. Root x runs
# monotonically (a single forward-travelling clip, not in-place), so this is a
# straightforward displacement/duration measurement, not a body-frame projection.
# Unaffected by loop-seam smoothing (that only touches dof_positions/velocities,
# not body_positions/root trajectory).
NOMINAL_VX = 0.660

# EMA low-pass filter on the RL residual (bingo_walk_ref_env.py's _pre_physics_step):
# filtered = ema_alpha*raw + (1-ema_alpha)*prev_filtered. Diagnostic measured the
# raw residual jumping up to 0.23 rad/step (77% of its own 0.3 rad authority) in a
# single 1/24s control step -- this smooths that without touching the network.
# MUST be replicated exactly by any external runtime consuming the ONNX export
# (see manifest's "residual_filter" field) since the graph itself is stateless and
# only emits the raw pre-filter action.
RESIDUAL_EMA_ALPHA = 0.5


@configclass
class BingoWalkRefEnvCfg(BingoTrackV4EnvCfg):
    """Residual RL on a cyclic, speed-scaled Stage 4 walk reference.
    q_target = q_ref(phase) + 0.3*action, phase rate proportional to commanded vx."""

    observation_space = 70  # proprio(67) + phase(2) + commanded vx(1)

    motion_file: str = WALK_V4
    ear_file: str = WALK_V4

    decimation = _DECIMATION
    episode_length_s = 8.0

    residual_scale = 0.3
    residual_ema_alpha = RESIDUAL_EMA_ALPHA

    # --- termination: Stage 4 fall criterion (no absolute-position drift term --
    # meaningless for a clip meant to be walked/looped many times per episode) ---
    early_termination = True
    tilt_limit_deg = 70.0

    # --- command: forward-speed range this stage trains on, and how often it's
    # resampled mid-episode (small = frequent start/stop practice) -----------
    vx_range = (NOMINAL_VX, NOMINAL_VX)   # stage A default: fixed nominal speed
    stand_prob = 0.0                       # probability a resampled command is 0 (stand)
    command_resample_s = 1000.0            # effectively "never" unless overridden
    stand_blend_vx_frac = 0.3              # walk gait fully engaged by this frac of nominal

    random_start_frame = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=_DECIMATION,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    def __post_init__(self):
        # keep BINGO_V4_CFG's own init_state (Stage 4 physics); RSI overwrites the
        # root pose and joint state every reset anyway.
        pass


@configclass
class BingoWalkRefEnvCfg_B(BingoWalkRefEnvCfg):
    """Stage B: small forward-velocity range around nominal, still no standing."""
    vx_range = (0.7 * NOMINAL_VX, 1.3 * NOMINAL_VX)
    command_resample_s = 3.0


@configclass
class BingoWalkRefEnvCfg_C(BingoWalkRefEnvCfg):
    """Stage C: + standing and start/stop transitions."""
    vx_range = (0.0, 1.3 * NOMINAL_VX)
    stand_prob = 0.35
    command_resample_s = 2.5
    episode_length_s = 10.0


@configclass
class BingoWalkRefEnvCfg_D(BingoWalkRefEnvCfg_C):
    """Stage D: sustained-hold exposure fix.

    diagnose_fall_timing.py (this session) found the smoothed-residual policy
    reliably falls ~2s into a CONTINUOUS 4s hold at exactly NOMINAL_VX -- via
    gradual root-height sinking (tilt only ~4.5 deg at the fall, well under the
    70 deg tip-over limit), not a transition-specific bug (an isolated
    stand->start_walk test completed cleanly). Stage C's 2.5s command-resample
    cadence means the policy rarely if ever practices an unbroken hold as long
    as the eval battery's 4s segments -- this stage closes that specific gap by
    holding each sampled command roughly twice as long.
    """
    command_resample_s = 5.0
    episode_length_s = 12.0


@configclass
class BingoWalkRefPlayEnvCfg(BingoWalkRefEnvCfg_C):
    """Deterministic single-robot playback/eval."""
    random_start_frame = False
    command_resample_s = 1000.0  # eval scripts drive the command explicitly

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
