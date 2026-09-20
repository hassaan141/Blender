"""Sideways-shuffle reference generator for lateral (left/right) locomotion.

Reuses V4Kin FK (stage2/v4_kinematics.py) and the DLS solve_leg IK extracted
from scripts/retarget.py -- same machinery as backward_geometry.py -- to
re-shape the TRUE forward champion's reference clip (a proven, dynamically
stable gait TIMING/CONTACT pattern under real Stage 4 physics) into a
sideways-shuffle in place.

Key research finding (see AUTONOMOUS_LATERAL_PROTOCOL.md): BingoWalkRefEnv's
reward never uses the reference's absolute root trajectory during stepping
(only joint pose / foot-tip-local / contacts, all root-relative). So the
authored reference's root can stay static -- no net displacement needs to be
encoded here at all; real translation emerges from the RL residual policy
being pulled by a NEW reward term (LateralWalkEnv's cmd_vy tracking), guided
toward a joint-pose neighborhood that resembles this shuffle.

Transform, per leg, per frame (keeping the ORIGINAL gait's stance/swing TIMING
and CONTACT PATTERN untouched):
  local-X (fore-aft) target = clamped to the leg's own mean stance depth
      (zero net fore-aft excursion -- this axis is not what should move).
  local-Y (lateral) target  = mean_Y + lateral_stride_scale * normalized_swing_shape
      where normalized_swing_shape is the ORIGINAL fore-aft swing's own
      excursion-from-mean, RESCALED onto the lateral axis -- i.e. reusing the
      forward gait's proven swing PROFILE shape, just applied to the other axis.
  local-Z lift-during-swing bump: unchanged from backward_geometry.py's bump math.

Usage: python3 geometry.py <arm_dir> [--direction right|left]
Reads <arm_dir>/config.json (written by core.py's Runner.arm()) for 'settings'.
Always regenerates FRESH from the fixed original source (SOURCE_REFERENCE
below), using the CUMULATIVE settings dict core.py already computes -- same
"always re-derive from the same fixed base, never edit incrementally" pattern
backward_geometry.py uses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_LOOP = ROOT / "rl/bingo_rl/experiment_loop"
sys.path.insert(0, str(EXPERIMENT_LOOP))
from core import save  # noqa: E402

sys.path.insert(0, str(ROOT / "stage2"))
from v4_kinematics import V4Kin, LEGS  # noqa: E402

# The TRUE current forward champion's own reference -- fixed, always re-derived
# from here (never from an already-transformed lateral reference), matching
# turning's choice (start_from_candidate: false) to build on the real best gait,
# not the rejected natural_walk/attempt_03 seed backward used.
SOURCE_REFERENCE = ROOT / "docs/walk_ref/refinement_runs/run_05/reference.npz"
URDF_PATH = ROOT / "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"

DEFAULT_SETTINGS = dict(
    # Measured (this session): source fore-aft swing range is ~106-120mm per leg,
    # vs. the SAME gait's own NATURAL lateral sway of only ~47-58mm -- mapping the
    # full fore-aft amplitude onto lateral at scale=1.0 asks for roughly 2x the
    # leg's demonstrated comfortable lateral reach and fails IK (residual 10.4mm
    # at scale=1.0, tested). 0.5 lands close to the naturally-observed sway
    # magnitude and is the conservative starting point; loops may tune this up
    # bounded by IK feasibility (build_reference raises if it doesn't converge).
    lateral_stride_scale=0.5,
    # Measured (this session): lift=25mm combined with stride_scale>=0.3 fails IK
    # (worst residual 6.9-7.9mm, fr_knee pinned at its -1.56 hard limit -- a real
    # workspace-reach conflict between the lift bump and the lateral excursion at
    # mid-swing, not a solver-iteration issue: 400 iterations gave the same
    # limit-pinned result as 60). lift<=15mm converges cleanly (residual ~0.001mm)
    # across stride_scale 0.3-0.5. 15mm kept as the default with headroom.
    lateral_lift_mm=15.0,       # swing-phase Z lift bump, per leg (uniform default)
    lateral_stance_width_scale=1.0,  # scales each foot's OWN mean lateral offset from the body
)


def _extract_solve_leg():
    import ast
    fn = next(n for n in ast.parse((ROOT / "scripts/retarget.py").read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "solve_leg")
    # Use THIS module's own globals (already has numpy as np) so the extracted
    # function body's free variable `np` resolves -- same technique
    # backward_geometry.py uses (bare exec() there relies on the same thing).
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "scripts/retarget.py", "exec"), globals())
    return globals()["solve_leg"]


solve_leg = _extract_solve_leg()


def build_reference(settings: dict, out_path: Path, direction_sign: float, audit_path: Path | None = None):
    """direction_sign: +1 => cmd_vy target will be positive (left); -1 => negative (right).
    Only affects the audit report's stated intent; the geometry itself is
    direction-neutral (see module docstring) -- direction is set purely by the
    SIGN of cmd_vy in the env cfg, not by anything encoded here. Kept as a
    parameter so the audit file records what this reference is meant to drive."""
    s = {**DEFAULT_SETTINGS, **{k: v for k, v in settings.items() if k in DEFAULT_SETTINGS}}
    src = np.load(SOURCE_REFERENCE)
    d = {k: src[k].copy() for k in src.files}
    original = d["dof_positions"].astype(float)
    q = original.copy()
    N = len(q)

    kin = V4Kin(URDF_PATH)
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(d["body_rotations"][:, 0][:, [1, 2, 3, 0]]).as_matrix()

    tips = np.array([[kin.leg_fk(l, row[j * 3:j * 3 + 3])[0] for j, l in enumerate(LEGS)] for row in q])
    target = tips.copy()

    for j, l in enumerate(LEGS):
        mean = tips[:-1, j].mean(axis=0)
        # local-X (fore-aft): clamp to the mean stance depth -- no net fore-aft excursion.
        target[:, j, 0] = mean[0]
        # local-Y (lateral): reuse the ORIGINAL fore-aft swing's own excursion-from-mean
        # shape, rescaled onto the lateral axis. This keeps the exact stance/swing
        # TIMING of the proven forward gait; only the axis assignment changes.
        fore_aft_excursion = tips[:, j, 0] - mean[0]
        target[:, j, 1] = mean[1] * s["lateral_stance_width_scale"] + s["lateral_stride_scale"] * fore_aft_excursion

    # swing-phase Z lift bump -- identical math to backward_geometry.py.
    lift = s["lateral_lift_mm"] / 1000.0
    for j, l in enumerate(LEGS):
        swing = ~d["contacts"][:-1, j].astype(bool)
        num = len(swing)
        for start in [i for i in range(num) if swing[i] and not swing[(i - 1) % num]]:
            ids = []; i = start
            while swing[i] and len(ids) < num:
                ids.append(i); i = (i + 1) % num
            if len(ids) < 4:
                continue
            bump = np.sin(np.linspace(0, np.pi, len(ids))) ** 2 * lift
            for ix, h in zip(ids, bump):
                target[ix, j] += rot[ix].T @ np.array([0.0, 0.0, h])

    target[-1] = target[0]
    errors = []
    for t in range(N):
        for j, l in enumerate(LEGS):
            sl = slice(3 * j, 3 * j + 3)
            q[t, sl], err = solve_leg(kin, l, target[t, j], q[t, sl], kin.leg_limits(l))
            errors.append(err)
    q[-1, :12] = q[0, :12]

    max_err = max(errors)
    max_delta = float(np.abs(q[:, :12] - original[:, :12]).max())
    if max_err > 0.001:
        raise ValueError(f"IK residual too large ({max_err:.4f} m); geometry not accepted")
    if max_delta > 1.0:  # measured 0.57 rad at defaults; generous headroom, not a loose placeholder
        raise ValueError(f"Joint delta too large ({max_delta:.3f} rad); geometry not accepted")

    for t in range(N):
        for j, l in enumerate(LEGS):
            tip, kr, kp = kin.leg_fk(l, q[t, j * 3:j * 3 + 3])
            oldtip, _, oldkp = kin.leg_fk(l, original[t, j * 3:j * 3 + 3])
            oldlocal = rot[t].T @ (d["body_positions"][t, j + 1] - d["body_positions"][t, 0])
            is_tip = np.linalg.norm(oldlocal - oldtip) < np.linalg.norm(oldlocal - oldkp)
            d["body_positions"][t, j + 1] = d["body_positions"][t, 0] + rot[t] @ (tip if is_tip else kp)
            from scipy.spatial.transform import Rotation as R2
            xyzw = R2.from_matrix(rot[t] @ kr).as_quat()
            d["body_rotations"][t, j + 1] = xyzw[[3, 0, 1, 2]]

    # Root stays static (an in-place shuffle) -- see module docstring: the
    # reference's absolute root trajectory is never used during stepping, only
    # for reset placement, so encoding a net displacement here would be inert
    # for training but misleading in playback/kinematic-preview review.
    d["body_positions"][:, 0] = d["body_positions"][0, 0]
    d["body_rotations"][:, 0] = d["body_rotations"][0, 0]

    fps = float(src["fps"])
    d["dof_positions"] = q.astype(np.float32)
    d["dof_velocities"] = np.gradient(q, 1 / fps, axis=0).astype(np.float32)
    d["body_linear_velocities"] = np.gradient(d["body_positions"], 1 / fps, axis=0).astype(np.float32)
    for j in range(d["body_rotations"].shape[1]):
        rr = Rotation.from_quat(d["body_rotations"][:, j][:, [1, 2, 3, 0]])
        omega = (rr[1:] * rr[:-1].inv()).as_rotvec() * fps
        d["body_angular_velocities"][:, j] = np.vstack([omega, omega[-1]])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)

    audit = {
        "source": str(SOURCE_REFERENCE), "settings": s, "direction_sign": direction_sign,
        "ik_max_error_m": max_err, "max_joint_delta_rad": max_delta,
        "contacts_unchanged": bool(np.array_equal(d["contacts"], src["contacts"])),
        "root_static": True,
        "geometry": "local-X clamped to mean stance depth (zero net fore-aft); "
                    "local-Y = rescaled copy of the original fore-aft swing shape; "
                    "local-Z unchanged swing-lift bump. Same stance/swing TIMING and "
                    "contact pattern as the source forward gait.",
        "feet": {l: {"source_range_mm": (np.ptp(tips[:, j], axis=0) * 1000).tolist(),
                     "target_range_mm": (np.ptp(target[:, j], axis=0) * 1000).tolist()}
                 for j, l in enumerate(LEGS)},
    }
    if audit_path:
        save(audit_path, audit)
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm_dir", type=Path)
    p.add_argument("--direction", choices=["right", "left"], default="right")
    a = p.parse_args()
    cfg = json.loads((a.arm_dir / "config.json").read_text())
    sign = -1.0 if a.direction == "right" else 1.0
    audit = build_reference(cfg.get("settings", {}), a.arm_dir / "reference.npz", sign,
                             a.arm_dir / "reference_audit.json")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
