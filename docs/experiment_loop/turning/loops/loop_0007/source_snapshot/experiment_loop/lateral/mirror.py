"""Mirror a RIGHT-stepping lateral reference into its LEFT-stepping counterpart.

IMPORTANT: an initial hand-derived hypothesis (pure column swap fl<->fr/bl<->br
with NO sign change, reasoning from the SY_SIGN/SP_SIGN/KN_SIGN maps in
rl/bingo_rl/scripts/retarget_dog_to_bingo.py) was WRONG -- numerically verified
and REJECTED (395mm mismatch on a random-pose FK round-trip test, nowhere near
the noise floor). The algebra that predicted "signs cancel" for the fl<->fr /
bl<->br pairing conflated two different sign conventions; it was not checked
before being written down the first time, exactly the mistake the "verify, don't
assume" discipline in this campaign exists to catch.

The CORRECT transform was instead found by numerical search (brute-force over
all 8 sign combinations per joint, scored against the true geometric mirror of
kin.leg_fk on random test poses across the leg's full joint-limit range):

    fl <-> fr: SY unchanged, SP negated, KN negated   (residual ~1.9e-5 m)
    bl <-> br: SY unchanged, SP unchanged, KN negated (residual ~1.0e-4 m)

Both confirmed symmetric (fr->fl and br->bl use the identical sign vectors,
residual near the joint-limit-sampling noise floor). `verify_mirror_hypothesis`
below re-runs this exact check at call time before trusting the transform in
the actual pipeline, rather than hardcoding the signs on faith.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "stage2"))
from v4_kinematics import V4Kin, LEGS  # noqa: E402

URDF_PATH = ROOT / "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"
LEG_SWAP = {"fl": "fr", "fr": "fl", "bl": "br", "br": "bl"}
# Per-leg-pair joint sign correction applied to the raw angle BEFORE feeding it
# to the swapped leg's own FK -- see module docstring for how this was found
# (numerical search, not hand algebra) and its measured residual.
MIRROR_SIGN = {
    "fl": np.array([1.0, -1.0, -1.0]), "fr": np.array([1.0, -1.0, -1.0]),
    "bl": np.array([1.0, 1.0, -1.0]), "br": np.array([1.0, 1.0, -1.0]),
}


def verify_mirror_hypothesis(kin: V4Kin, n_random: int = 30, seed: int = 0) -> float:
    """Returns the max mismatch (m) between the true geometric mirror of a
    random test pose's foot positions and the positions produced by the
    swap+sign-correct transform. Raises if it's not near the joint-limit-
    sampling noise floor (~1e-4 m, measured)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_random):
        q = {}
        for l in LEGS:
            lo, hi = kin.leg_limits(l)[:, 0], kin.leg_limits(l)[:, 1]
            q[l] = lo + rng.random(3) * (hi - lo)
        for l in LEGS:
            orig_tip, *_ = kin.leg_fk(l, q[l])
            true_mirror = orig_tip * np.array([1.0, -1.0, 1.0])
            swapped_leg = LEG_SWAP[l]
            hyp_tip, *_ = kin.leg_fk(swapped_leg, q[l] * MIRROR_SIGN[l])
            mismatch = np.linalg.norm(true_mirror - hyp_tip)
            worst = max(worst, mismatch)
    return worst


def mirror_reference(src_path: Path, out_path: Path):
    kin = V4Kin(URDF_PATH)
    mismatch = verify_mirror_hypothesis(kin)
    if mismatch > 5e-4:  # measured noise floor ~1.0e-4 m; generous headroom, not a loose placeholder
        raise ValueError(
            f"Mirror hypothesis FAILED numerically (max mismatch {mismatch*1000:.4f}mm); "
            "do not trust this transform -- re-derive the sign correction instead."
        )

    src = np.load(src_path)
    d = {k: src[k].copy() for k in src.files}
    q = d["dof_positions"]
    qv = d["dof_velocities"]
    leg_idx = {l: slice(3 * j, 3 * j + 3) for j, l in enumerate(LEGS)}

    new_q = q.copy()
    new_qv = qv.copy()
    for l in LEGS:
        swapped = LEG_SWAP[l]
        new_q[:, leg_idx[l]] = q[:, leg_idx[swapped]] * MIRROR_SIGN[swapped]
        new_qv[:, leg_idx[l]] = qv[:, leg_idx[swapped]] * MIRROR_SIGN[swapped]
    d["dof_positions"] = new_q
    d["dof_velocities"] = new_qv

    # body marker positions/rotations for the 4 foot markers: recompute via FK
    # from the mirrored q (cleaner than trying to mirror the marker arrays
    # directly), exactly like geometry.py already does.
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(d["body_rotations"][:, 0][:, [1, 2, 3, 0]]).as_matrix()
    for t in range(len(new_q)):
        for j, l in enumerate(LEGS):
            tip, kr, kp = kin.leg_fk(l, new_q[t, leg_idx[l]])
            oldtip, _, oldkp = kin.leg_fk(l, q[t, leg_idx[l]])
            oldlocal = rot[t].T @ (src["body_positions"][t, j + 1] - src["body_positions"][t, 0])
            is_tip = np.linalg.norm(oldlocal - oldtip) < np.linalg.norm(oldlocal - oldkp)
            d["body_positions"][t, j + 1] = d["body_positions"][t, 0] + rot[t] @ (tip if is_tip else kp)
            xyzw = Rotation.from_matrix(rot[t] @ kr).as_quat()
            d["body_rotations"][t, j + 1] = xyzw[[3, 0, 1, 2]]

    # contacts: swap the same way (fl<->fr, bl<->br columns)
    c = d["contacts"]
    new_c = c.copy()
    for j, l in enumerate(LEGS):
        sj = LEGS.index(LEG_SWAP[l])
        new_c[:, j] = c[:, sj]
    d["contacts"] = new_c

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    return {"mirror_hypothesis_max_mismatch_mm": mismatch * 1000.0, "leg_swap": LEG_SWAP}


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    report = mirror_reference(a.src, a.out)
    print(json.dumps(report, indent=2))
