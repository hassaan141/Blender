"""Add a local, physically meaningful takeoff stroke to a Stage-4 reference.

The Blender root trajectory is not actuated in physics.  A hop therefore needs to
be encoded in the leg targets: while paws are on the floor they move backward and
down relative to the body, producing forward and upward ground reaction, then
retract during flight.  This pass realizes that stroke with exact v4 leg IK and
unchanged URDF limits; expression joints and motion outside the window are copied.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "stage2")))
from v4_kinematics import V4Kin, LEGS

URDF = os.path.abspath(os.path.join(
    HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
    "bingo_urdf_w_ear_joints_physics.urdf"))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--peak", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--push-x", type=float, default=-0.04,
                    help="metres; negative moves paws backward to push body forward")
    ap.add_argument("--push-z", type=float, default=-0.025,
                    help="metres; negative extends paws down to push body upward")
    ap.add_argument("--legs", default="fl,fr,bl,br",
                    help="comma-separated legs used for this takeoff stroke")
    a = ap.parse_args()
    if not a.start < a.peak < a.end:
        raise SystemExit("require start < peak < end")

    m = np.load(a.motion, allow_pickle=True)
    out = {k: m[k] for k in m.files}
    q_ref = m["dof_positions"].astype(float)
    q = q_ref.copy()
    names = [str(x) for x in m["dof_names"]]
    T = len(q)
    s, p, e = max(0, a.start), min(T - 1, a.peak), min(T - 1, a.end)
    w = np.zeros(T)
    w[s:p + 1] = smoothstep(np.linspace(0.0, 1.0, p - s + 1))
    w[p:e + 1] = smoothstep(np.linspace(1.0, 0.0, e - p + 1))
    offset = w[:, None] * np.array([a.push_x, 0.0, a.push_z])

    kin = V4Kin(URDF)
    achieved = []
    selected = [x.strip() for x in a.legs.split(",") if x.strip()]
    if any(x not in LEGS for x in selected):
        raise SystemExit(f"unknown leg in --legs {a.legs!r}")
    for leg in selected:
        ids = [names.index(f"{leg}_SY_J"), names.index(f"{leg}_SP_J"),
               names.index(f"{leg}_knee")]
        lim = kin.leg_limits(leg)
        lo, hi = lim[:, 0], lim[:, 1]
        ref_ankle = np.array([kin.leg_points(leg, q_ref[i, ids])[2]
                              for i in range(T)])
        prev_corr = np.zeros(3)
        for i in range(s, e + 1):
            qr = q_ref[i, ids]
            target = ref_ankle[i] + offset[i]

            def residual(ql):
                ankle = kin.leg_points(leg, ql)[2]
                return np.concatenate([
                    90.0 * (ankle - target),
                    0.45 * (ql - qr),
                    0.20 * ((ql - qr) - prev_corr),
                ])

            sol = least_squares(residual, np.clip(qr + prev_corr, lo, hi),
                                bounds=(lo, hi), max_nfev=60,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            q[i, ids] = sol.x
            prev_corr = sol.x - qr
            achieved.append(kin.leg_points(leg, sol.x)[2] - ref_ankle[i])

    dt = 1.0 / float(m["fps"])
    out["dof_positions"] = q.astype(np.float32)
    out["dof_velocities"] = np.gradient(q, dt, axis=0).astype(np.float32)
    out["stage4_takeoff_window"] = np.array([s, p, e], np.int32)
    out["stage4_takeoff_push"] = np.array([a.push_x, 0.0, a.push_z])
    np.savez(a.out, **out)

    dq = np.abs(q - q_ref)
    ach = np.asarray(achieved)
    print(f"[[ takeoff {s}-{p}-{e} on {','.join(selected)}: requested paw stroke "
          f"({a.push_x*1000:+.1f}, {a.push_z*1000:+.1f}) mm in body x/z")
    print(f"[[ achieved stroke range x {ach[:,0].min()*1000:+.1f}.."
          f"{ach[:,0].max()*1000:+.1f} mm | z {ach[:,2].min()*1000:+.1f}.."
          f"{ach[:,2].max()*1000:+.1f} mm")
    print(f"[[ joint correction mean/max {np.degrees(dq[:,:12].mean()):.2f}/"
          f"{np.degrees(dq[:,:12].max()):.2f} deg | max leg velocity "
          f"{np.abs(out['dof_velocities'][:,:12]).max():.2f} rad/s")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
