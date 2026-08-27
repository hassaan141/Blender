"""Small local body shift with leg IK that preserves authored paw trajectories.

This is Stage 4's second intervention, after local retiming.  It moves the root
laterally only inside a selected window, then solves each physical v4 leg so its
ankle trajectory remains authored and every near-ground paw keeps its original
world-space collision support.  Root orientation and expressive joints are not
changed.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "stage2")))
from contact_model import ContactModel, HULLS_NPZ
from v4_kinematics import V4Kin, LEGS, axis_rot, mat_to_quat, quat_to_mat

URDF = os.path.abspath(os.path.join(
    HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
    "bingo_urdf_w_ear_joints_physics.urdf"))
CONTACT_HEIGHT = 0.005


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--ramp", type=int, default=8)
    ap.add_argument("--shift-x", type=float, default=0.0, help="metres")
    ap.add_argument("--shift-y", type=float, required=True, help="metres")
    ap.add_argument("--shift-z", type=float, default=0.0, help="metres")
    ap.add_argument("--roll-deg", type=float, default=0.0,
                    help="small local body-roll correction inside the window")
    ap.add_argument("--pitch-deg", type=float, default=0.0,
                    help="small local body-pitch correction inside the window")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="small local body-yaw/steering correction inside the window")
    ap.add_argument("--hold-end", action="store_true",
                    help="keep the correction through the last selected frame")
    a = ap.parse_args()

    m = np.load(a.motion, allow_pickle=True)
    out = {k: m[k] for k in m.files}
    q_ref = m["dof_positions"].astype(float)
    q = q_ref.copy()
    names = [str(x) for x in m["dof_names"]]
    leg_idx = [[names.index(f"{leg}_SY_J"), names.index(f"{leg}_SP_J"),
                names.index(f"{leg}_knee")] for leg in LEGS]
    root_ref = (m["root_pos"] if "root_pos" in m.files
                else m["body_positions"][:, 0]).astype(float)
    root_quat_ref = (m["root_quat"] if "root_quat" in m.files
                     else m["body_rotations"][:, 0]).astype(float)
    T = len(q); Rref = np.array([quat_to_mat(x) for x in root_quat_ref])
    kin = V4Kin(URDF); cm = ContactModel()

    s = max(0, a.start); e = min(T - 1, a.end); ramp = max(1, a.ramp)
    weight = np.zeros(T)
    weight[s:e + 1] = 1.0
    weight[s:min(e + 1, s + ramp)] = smoothstep(
        np.linspace(0.0, 1.0, min(ramp, e - s + 1)))
    if not a.hold_end:
        weight[max(s, e - ramp + 1):e + 1] = smoothstep(
            np.linspace(1.0, 0.0, min(ramp, e - s + 1)))
    delta = weight[:, None] * np.array([a.shift_x, a.shift_y, a.shift_z])
    root = root_ref + delta
    roll = np.radians(a.roll_deg) * weight
    pitch = np.radians(a.pitch_deg) * weight
    yaw = np.radians(a.yaw_deg) * weight
    Rroot = np.array([
        Rref[i]
        @ axis_rot(np.array([1.0, 0.0, 0.0]), roll[i])
        @ axis_rot(np.array([0.0, 1.0, 0.0]), pitch[i])
        @ axis_rot(np.array([0.0, 0.0, 1.0]), yaw[i])
        for i in range(T)
    ])
    root_quat = np.array([mat_to_quat(R) for R in Rroot])

    hull = {leg: cm.hull[f"{leg}_knee"] for leg in LEGS}
    lim = {leg: kin.leg_limits(leg) for leg in LEGS}

    def geom(i, leg, qleg, root_pos, base_R):
        _, knee, ankle, patch, _ = kin.leg_points(
            leg, qleg, support_hull=hull[leg], world_R=base_R,
            support_softness=0.001)
        low = kin.leg_points(
            leg, qleg, support_hull=hull[leg], world_R=base_R)[3]
        return (root_pos + base_R @ knee,
                root_pos + base_R @ ankle,
                root_pos + base_R @ patch,
                root_pos + base_R @ low)

    ref = np.zeros((T, 4, 4, 3))
    for i in range(T):
        for k, leg in enumerate(LEGS):
            ref[i, k] = geom(i, leg, q_ref[i, leg_idx[k]], root_ref[i], Rref[i])
    physical = ref[:, :, 3, 2] <= CONTACT_HEIGHT
    schedule = (m["contacts"].astype(bool) if "contacts" in m.files
                and len(m["contacts"]) == T else np.zeros((T, 4), bool))
    anchored = physical | schedule

    for k, leg in enumerate(LEGS):
        prev_corr = np.zeros(3)
        ids = leg_idx[k]
        bounds = (lim[leg][:, 0], lim[leg][:, 1])
        for i in range(T):
            if weight[i] == 0.0:
                prev_corr = np.zeros(3)
                continue
            qr = q_ref[i, ids]

            def residual(ql):
                knee, ankle, patch, low = geom(i, leg, ql, root[i], Rroot[i])
                r = [55.0 * (ankle - ref[i, k, 1]),
                     0.30 * (ql - qr),
                     0.16 * ((ql - qr) - prev_corr)]
                if anchored[i, k]:
                    r.extend([110.0 * (patch[:2] - ref[i, k, 2, :2]),
                              np.array([110.0 * (low[2] - ref[i, k, 3, 2])])])
                return np.concatenate(r)

            sol = least_squares(residual, np.clip(qr + prev_corr, *bounds),
                                bounds=bounds, max_nfev=45,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            q[i, ids] = sol.x
            prev_corr = sol.x - qr

    # Measure the actual result with exact collision geometry.
    ankle_err = []; anchor_err = []; lowest = np.zeros((T, 4, 3))
    patches = np.zeros((T, 4, 3))
    for i in range(T):
        for k, leg in enumerate(LEGS):
            _, ankle, patch, low = geom(
                i, leg, q[i, leg_idx[k]], root[i], Rroot[i])
            patches[i, k] = patch
            lowest[i, k] = low
            ankle_err.append(np.linalg.norm(ankle - ref[i, k, 1]))
            if anchored[i, k]:
                anchor_err.append(np.linalg.norm(
                    np.r_[patch[:2] - ref[i, k, 2, :2],
                          low[2] - ref[i, k, 3, 2]]))

    fps = float(m["fps"]); dt = 1.0 / fps
    support = patches.copy(); support[:, :, 2] = lowest[:, :, 2]
    support_speed = np.linalg.norm(np.gradient(support, dt, axis=0), axis=2)
    speed_cap = float(m["contact_speed_tolerance"]) if "contact_speed_tolerance" in m.files else 0.014
    height_ok = lowest[:, :, 2] <= CONTACT_HEIGHT
    source_contacts = (m["source_contacts"].astype(bool)
                       if "source_contacts" in m.files else schedule)
    contacts = height_ok & (source_contacts | (support_speed <= speed_cap))
    out.update(
        root_pos=root.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
        dof_positions=q.astype(np.float32),
        dof_velocities=np.gradient(q, dt, axis=0).astype(np.float32),
        contacts=contacts,
        source_contacts=source_contacts,
        physical_height_contacts=height_ok,
        planted_points_world=support.astype(np.float32),
        support_patch_world=patches.astype(np.float32),
        tips_world=lowest.astype(np.float32),
        contacts_world=lowest.astype(np.float32),
        support_speed=support_speed.astype(np.float32),
        collision_hulls=np.array(HULLS_NPZ),
        stage4_balance_window=np.array([s, e, ramp], dtype=np.int32),
        stage4_root_shift=np.array([a.shift_x, a.shift_y, a.shift_z]),
    )
    if "body_positions" in out:
        bp = out["body_positions"].copy(); bp[:, 0] = root
        out["body_positions"] = bp
    if "body_rotations" in out:
        br = out["body_rotations"].copy(); br[:, 0] = root_quat
        out["body_rotations"] = br
    np.savez(a.out, **out)

    dq = np.abs(q - q_ref)
    print(f"[[ root shift window {s}-{e}, ramp {ramp}: "
          f"({a.shift_x*1000:+.1f}, {a.shift_y*1000:+.1f}, "
          f"{a.shift_z*1000:+.1f}) mm, "
          f"roll {a.roll_deg:+.1f} deg, pitch {a.pitch_deg:+.1f} deg, "
          f"yaw {a.yaw_deg:+.1f} deg")
    print(f"[[ leg correction mean/max {np.degrees(dq[:, :12].mean()):.2f}/"
          f"{np.degrees(dq[:, :12].max()):.2f} deg")
    print(f"[[ ankle error mean/max {np.mean(ankle_err)*1000:.2f}/"
          f"{np.max(ankle_err)*1000:.2f} mm | anchored support error mean/max "
          f"{np.mean(anchor_err)*1000:.2f}/{np.max(anchor_err)*1000:.2f} mm")
    print(f"[[ ground min {lowest[:,:,2].min()*1000:+.2f} mm | wrote {a.out}")
    print(f"[[ contacts {int(contacts.sum())} foot-frames | source agreement "
          f"{100*(contacts == source_contacts).mean():.1f}%")


if __name__ == "__main__":
    main()
