"""Stage 5 config: residual RL on top of the validated Stage 4 Timid reference.

Everything here is INHERITED from the existing project infrastructure; only the
values that are provably incompatible with the Stage 4 v4 reference are changed,
and each such change carries its measured justification.

Inherited unchanged from bingo_rl.track.bingo_track_env_cfg.BingoTrackEnvCfg /
bingo_rl.track_v4.bingo_track_v4_env_cfg.BingoTrackV4EnvCfg:
  * the spec-B.4.1 imitation reward and all its weights (hard-coded in the env)
  * observation_space 69, action_space 12
  * robot = BINGO_V4_CFG (IdealPD actuators, Stage 4 gains) -- physics unchanged
  * reference_body "origin", KEY_BODY_NAMES = the four knee/shank tips
  * early_termination, RSI, phase observation

Changed, with reasons (all measured on motions/timid_v4.npz):
  decimation 4 -> 5
      120 Hz / 5 = 24 Hz == the reference fps (fps field = 24.0). This is what the
      Stage 4 baseline (rl/tools/track_v4_physics.py) uses; at decimation 4 the
      control rate would not divide the reference and every step would land
      between frames.
  episode_length_s 8.0 -> 7.5
      180 frames / 24 fps = 7.5 s exactly, so one episode is at most one pass of
      the clip. Timid is NOT cyclic (root x runs -0.016 -> -0.364 m), so the
      deadpan tracker's wrap-around would teleport the reference 0.35 m mid-episode.
  termination_height 0.12 -> 0.045
      the REFERENCE root z itself ranges 0.0899..0.1897 m, so 0.12 m terminates on
      the authored motion. 0.045 = 0.5 * ref root z at frame 0 (0.0900 m), which is
      exactly the Stage 4 fall criterion.
  root_track_max 0.15 -> 0.70
      the Stage 4 PD-only baseline already reaches 618 mm of root error (mean 445 mm)
      without falling, so 0.15 m would terminate the baseline at ~frame 20 and make
      the task unlearnable. 0.70 m sits just above the measured baseline maximum, so
      it never fires for baseline-quality behaviour but still catches runaway drift.
  motion_file / ear_file -> motions/timid_v4.npz
      the Stage 4 canonical reference. It carries head_tail_positions AND
      ear_positions in the same npz, so no separate ear file is needed.
  ground friction 1.0/1.0 -> IsaacLab default 0.5/0.5  (in the env's _setup_scene)
      Stage 4 was validated on the default GroundPlaneCfg. The deadpan tracker
      forced 1.0/1.0, which would be a physics change.

residual_scale = 0.3 is taken verbatim from the existing
BingoTrackResidualEnvCfg (the README's best tracker, 0.099 rad on deadpan).
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from bingo_rl.track_v4.bingo_track_v4_env_cfg import BingoTrackV4EnvCfg

# .../Blender/rl/bingo_rl/bingo_rl/stage5/this_file.py -> parents[4] == Blender/
BLENDER_ROOT = Path(__file__).resolve().parents[4]
TIMID_V4 = str(BLENDER_ROOT / "motions" / "timid_v4.npz")

_DECIMATION = 5  # 120 Hz physics / 5 = 24 Hz control = reference fps


@configclass
class BingoStage5TimidEnvCfg(BingoTrackV4EnvCfg):
    """Residual RL on the Stage 4 Timid reference. q_target = q_ref + 0.3*action."""

    # --- reference -----------------------------------------------------------
    motion_file: str = TIMID_V4
    ear_file: str = TIMID_V4          # ear_positions live in the same npz

    # --- timing: match Stage 4 exactly --------------------------------------
    decimation = _DECIMATION
    episode_length_s = 7.5            # 180 frames / 24 fps

    # --- residual action (from BingoTrackResidualEnvCfg) --------------------
    residual_scale = 0.3

    # --- termination: Stage 4 fall criterion + measured drift bound ---------
    early_termination = True
    termination_height = 0.045        # 0.5 * reference root z at frame 0
    tilt_limit_deg = 70.0            # Stage 4 fall criterion
    root_track_max = 0.70            # above the measured 618 mm baseline maximum

    # --- reference-state initialization ------------------------------------
    random_start_frame = True         # RSI: sample a start frame across the clip
    # Zero root/joint velocity at reset, which is what the Stage 4 baseline does.
    # MEASURED: seeding the reference velocity instead (frame 0 is only 0.238 rad/s
    # of joint velocity and 0.0018 m/s of root velocity) is already enough to push
    # this clip off its stability basin -- a ZERO-residual run then falls by frame 55
    # instead of completing. Keeping zeros makes the training distribution contain
    # the exact condition the policy is evaluated under.
    rsi_use_reference_velocity = False

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
class BingoStage5TimidPlayEnvCfg(BingoStage5TimidEnvCfg):
    """Deterministic single-robot playback/eval: always start at frame 0, like Stage 4.

    rsi_use_reference_velocity is False here so that a ZERO residual reproduces the
    Stage 4 baseline exactly. Stage 4 initialises with zero root and joint velocity;
    measured, seeding the reference velocity instead (max 0.238 rad/s at frame 0) is
    enough to push this clip off its stability basin and fall by frame 55.
    """

    random_start_frame = False
    rsi_use_reference_velocity = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
