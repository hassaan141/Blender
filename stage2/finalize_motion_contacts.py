"""Copy validated Stage-2 physical contacts into an exact Stage-3 bake.

`bake_conform.py` remains unchanged and continues to validate root/joint replay.
Its legacy fixed-tip contact estimate is replaced only after confirming that the
baked root and 21 joint trajectories exactly match the contact-locked Stage-2
motion.
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baked", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = np.load(a.baked, allow_pickle=True)
    s = np.load(a.stage2, allow_pickle=True)
    qerr = float(np.max(np.abs(b["dof_positions"] - s["dof_positions"])))
    rerr = float(np.max(np.abs(b["body_positions"][:, 0] - s["root_pos"])))
    # Quaternion sign is equivalent; compare absolute dot angle.
    qb = b["body_rotations"][:, 0].astype(float)
    qs = s["root_quat"].astype(float)
    dot = np.abs(np.sum(qb * qs, axis=1) /
                 (np.linalg.norm(qb, axis=1) * np.linalg.norm(qs, axis=1)))
    aerr = float(np.degrees(2*np.arccos(np.clip(dot, -1, 1))).max())
    if qerr > 2e-5 or rerr > 2e-5 or aerr > 1e-3:
        raise RuntimeError(f"bake mismatch q={qerr:g} root={rerr:g} quat={aerr:g}deg")
    out = {k: b[k] for k in b.files}
    out["contacts"] = s["contacts"].astype(bool)
    out["source_contacts"] = s["source_contacts"].astype(bool)
    out["physical_height_contacts"] = s["physical_height_contacts"].astype(bool)
    out["contact_height_tolerance"] = s["contact_height_tolerance"]
    out["contact_speed_tolerance"] = s["contact_speed_tolerance"]
    out["contact_geometry"] = s["collision_hulls"]
    out["contact_stage2_source"] = np.array(a.stage2)
    # Carry through any Stage-4 reference modifications so the audits score the
    # motion against the schedule/tempo it was actually built to.
    for k in ("stage4_planted", "wrench_offset", "glide_removed",
              "stage4_retime_factor", "stage4_retime", "stage4_balance_window",
              "stage4_root_shift", "stage4_dz"):
        if k in s.files:
            out[k] = s[k]
    np.savez(a.out, **out)
    print(f"[[ exact bake match: joints {qerr:.3g} rad | root {rerr:.3g} m | "
          f"orientation {aerr:.3g} deg")
    print(f"[[ contacts replaced from physical hull result: {int(out['contacts'].sum())} "
          f"foot-frames | source agreement {100*(out['contacts']==out['source_contacts']).mean():.1f}%")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
