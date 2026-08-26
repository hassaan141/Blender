"""Stage 2 temporal refinement: lock physical v4 support points during stance.

This pass deliberately starts from an already-successful spatial retarget.  It
does not re-map morphology, change root orientation, touch expression joints, or
alter the robot.  For each source stance segment it records the orientation-aware
collision support at touchdown and then solves the 12 physical leg joints plus a
small root-translation residual so that support remains at that world anchor.

The collision-patch XY is a smooth 1 mm soft-min centroid of the real convex hull;
ground height is the exact lowest hull vertex.  This avoids the 10 cm numerical
jump produced by differentiating a raw argmin vertex while still using exactly the
geometry that PhysX collides against.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "stage4")))
from contact_model import ContactModel, HULLS_NPZ
from v4_kinematics import V4Kin, LEGS, quat_to_mat

ALEGS = ["aFL", "aFR", "aBL", "aBR"]
PATCH_SOFTNESS = 0.001
CONTACT_HEIGHT = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="successful Stage-2 spatial retarget")
    ap.add_argument("--contacts", required=True, help="source contact schedule")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--collision-hulls", default=HULLS_NPZ)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = np.load(a.motion, allow_pickle=True)
    src_c = np.load(a.contacts, allow_pickle=True)
    kin = V4Kin(a.urdf)
    cm = ContactModel(a.collision_hulls)
    hull = {l: cm.hull[f"{l}_knee"] for l in LEGS}

    q_ref = m["dof_positions"].astype(float)
    q = q_ref.copy()
    root_ref = (m["root_pos"] if "root_pos" in m.files
                else m["body_positions"][:, 0]).astype(float)
    root = root_ref.copy()
    root_quat = (m["root_quat"] if "root_quat" in m.files
                 else m["body_rotations"][:, 0]).astype(float)
    Rroot = np.array([quat_to_mat(x) for x in root_quat])
    fps = float(m["fps"]); dt = 1.0 / fps; T = len(q)
    frames = m["frames"] if "frames" in m.files else np.arange(T) + 1

    # Honour the spatial solver's actual bilateral mapping.
    rmap = {l: "a" + l.upper() for l in LEGS}
    if "leg_map" in m.files:
        rmap = {}
        for item in m["leg_map"]:
            al, rl = str(item).split("->")
            rmap[rl] = al
    source_ct = np.zeros((T, 4), bool)
    for k, l in enumerate(LEGS):
        source_ct[:, k] = src_c["contacts"][:, ALEGS.index(rmap[l])]

    lim = np.vstack([kin.leg_limits(l) for l in LEGS])
    knee_branch = np.sign([np.median(q_ref[:, 3*k+2]) for k in range(4)])
    for k, s in enumerate(knee_branch):
        if s >= 0:
            lim[3*k+2, 0] = max(lim[3*k+2, 0], 0.0)
        else:
            lim[3*k+2, 1] = min(lim[3*k+2, 1], 0.0)

    def leg_geom(i, k, qleg, root_pos):
        l = LEGS[k]
        SP, KP, AP, patch, _ = kin.leg_points(
            l, qleg, support_hull=hull[l], world_R=Rroot[i],
            support_softness=PATCH_SOFTNESS)
        low = kin.leg_points(l, qleg, support_hull=hull[l], world_R=Rroot[i])[3]
        return (root_pos + Rroot[i] @ KP,
                root_pos + Rroot[i] @ AP,
                root_pos + Rroot[i] @ patch,
                root_pos + Rroot[i] @ low)

    # Reference morphology landmarks and physical support tracks.
    knee_ref = np.zeros((T, 4, 3)); ankle_ref = np.zeros_like(knee_ref)
    patch_ref = np.zeros_like(knee_ref); low_ref = np.zeros_like(knee_ref)
    for i in range(T):
        for k in range(4):
            knee_ref[i, k], ankle_ref[i, k], patch_ref[i, k], low_ref[i, k] = \
                leg_geom(i, k, q_ref[i, 3*k:3*k+3], root_ref[i])

    # One XY anchor variable per contiguous source stance. Initial values come
    # from the authored touchdown, but the global solve may move them slightly so
    # overlapping stance segments remain compatible with v4 morphology.
    segment_id = np.full((T, 4), -1, dtype=int)
    anchor_ref = []
    for k in range(4):
        i = 0
        while i < T:
            if not source_ct[i, k]:
                i += 1
                continue
            j = i + 1
            while j < T and source_ct[j, k]:
                j += 1
            sid = len(anchor_ref)
            anchor_ref.append(patch_ref[i, k, :2].copy())
            segment_id[i:j, k] = sid
            i = j
    anchor_ref = np.asarray(anchor_ref)
    S = len(anchor_ref)

    max_q_step = 10.0 * dt
    max_root_step = 0.004

    # Global variables per frame: root residual xyz + 12 leg joints, followed by
    # 2D stance anchors. All contact frames and both neighbouring frames are
    # optimized together, so the swing trajectory can prepare for touchdown.
    NF = 15
    anchor_base = T * NF
    x0 = np.zeros(anchor_base + 2*S)
    for i in range(T):
        x0[i*NF + 3:i*NF + 15] = q_ref[i, :12]
    x0[anchor_base:] = anchor_ref.ravel()

    lo = np.full_like(x0, -np.inf); hi = np.full_like(x0, np.inf)
    for i in range(T):
        b = i * NF
        lo[b:b+3] = -0.035; hi[b:b+3] = 0.035
        lo[b+3:b+15] = lim[:, 0]; hi[b+3:b+15] = lim[:, 1]
    lo[anchor_base:] = (anchor_ref - 0.060).ravel()
    hi[anchor_base:] = (anchor_ref + 0.060).ravel()

    def unpack(x, i):
        b = i * NF
        return x[b:b+3], x[b+3:b+15]

    def residual(x, with_deps=False):
        out = []; deps = []
        def add(values, indices):
            v = np.atleast_1d(values)
            out.extend(v)
            if with_deps:
                deps.extend([np.asarray(indices, dtype=int)] * len(v))

        for i in range(T):
            b = i * NF
            d, qq = unpack(x, i); rp = root_ref[i] + d
            frame_idx = np.arange(b, b + NF)
            for k in range(4):
                knee, ankle, patch, low = leg_geom(i, k, qq[3*k:3*k+3], rp)
                if source_ct[i, k]:
                    sid = segment_id[i, k]
                    ab = anchor_base + 2*sid
                    anchor = x[ab:ab+2]
                    add(np.sqrt(100000.0) * np.r_[patch[:2] - anchor, low[2]],
                        np.r_[frame_idx, ab, ab+1])
                    add(np.sqrt(1.0) * (ankle - ankle_ref[i, k]), frame_idx)
                    add(np.sqrt(0.6) * (knee - knee_ref[i, k]), frame_idx)
                else:
                    add(np.sqrt(8.0) * (ankle - ankle_ref[i, k]), frame_idx)
                    add(np.sqrt(4.0) * (knee - knee_ref[i, k]), frame_idx)
                    add(np.sqrt(2.0) * (patch - patch_ref[i, k]), frame_idx)
                    add(np.sqrt(100000.0) * min(0.0, low[2]), frame_idx)
            add(0.035 * (qq - q_ref[i, :12]), frame_idx)
            add(np.sqrt(12.0) * d, frame_idx)

        # Smooth corrections rather than the authored performance itself.
        for i in range(1, T):
            b0 = (i-1)*NF; b1 = i*NF
            d0, q0_ = unpack(x, i-1); d1, q1_ = unpack(x, i)
            add(np.sqrt(50.0) * (d1 - d0), np.r_[np.arange(b0,b0+3), np.arange(b1,b1+3)])
            corr0 = q0_ - q_ref[i-1, :12]; corr1 = q1_ - q_ref[i, :12]
            add(0.04 * (corr1 - corr0), np.r_[np.arange(b0+3,b0+15), np.arange(b1+3,b1+15)])
            # Strong hinge penalties retain established physical rate bounds.
            qstep = q1_ - q0_
            add(np.sqrt(2000.0) * np.maximum(0.0, np.abs(qstep) - max_q_step),
                np.r_[np.arange(b0+3,b0+15), np.arange(b1+3,b1+15)])
            dstep = d1 - d0
            add(np.sqrt(5000.0) * np.maximum(0.0, np.abs(dstep) - max_root_step),
                np.r_[np.arange(b0,b0+3), np.arange(b1,b1+3)])
        for i in range(2, T):
            ids = np.r_[np.arange((i-2)*NF,(i-2)*NF+3),
                         np.arange((i-1)*NF,(i-1)*NF+3),
                         np.arange(i*NF,i*NF+3)]
            d0,_ = unpack(x,i-2); d1,_ = unpack(x,i-1); d2,_ = unpack(x,i)
            add(np.sqrt(20.0) * (d2 - 2*d1 + d0), ids)
        for sid in range(S):
            ab = anchor_base + 2*sid
            add(np.sqrt(25.0) * (x[ab:ab+2] - anchor_ref[sid]), np.arange(ab,ab+2))
        return (np.asarray(out), deps) if with_deps else np.asarray(out)

    r0, deps = residual(x0, with_deps=True)
    sparsity = lil_matrix((len(r0), len(x0)), dtype=int)
    for row, cols in enumerate(deps):
        sparsity[row, cols] = 1
    print(f"[[ global temporal solve: {T} frames, {S} stance anchors, "
          f"{len(x0)} variables, {len(r0)} residuals", flush=True)
    sol = least_squares(residual, x0, bounds=(lo, hi), jac_sparsity=sparsity.tocsr(),
                        method="trf", tr_solver="lsmr", x_scale="jac",
                        ftol=2e-7, xtol=2e-7, gtol=2e-7, max_nfev=90,
                        verbose=1)
    root_delta = np.zeros((T, 3))
    for i in range(T):
        root_delta[i], q[i, :12] = unpack(sol.x, i)
        root[i] = root_ref[i] + root_delta[i]
    anchor_xy = sol.x[anchor_base:].reshape(S, 2)
    anchors = np.full((T, 4, 3), np.nan)
    for i in range(T):
        for k in np.where(source_ct[i])[0]:
            anchors[i, k] = np.r_[anchor_xy[segment_id[i,k]], 0.0]

    # Honest final geometry, after every correction.
    knees = np.zeros((T, 4, 3)); ankles = np.zeros_like(knees)
    patch = np.zeros_like(knees); lowest = np.zeros_like(knees)
    support = np.zeros_like(knees)
    for i in range(T):
        for k in range(4):
            knees[i, k], ankles[i, k], patch[i, k], lowest[i, k] = \
                leg_geom(i, k, q[i, 3*k:3*k+3], root[i])
            support[i, k] = np.r_[patch[i, k, :2], lowest[i, k, 2]]

    support_speed = np.linalg.norm(np.gradient(support, dt, axis=0), axis=2)
    speed_cap = 0.06 * float(np.mean([kin.leg_reach(l) for l in LEGS]))
    height_ok = lowest[:, :, 2] <= CONTACT_HEIGHT
    contacts = height_ok & (source_ct | (support_speed <= speed_cap))

    out = {k: m[k] for k in m.files}
    out.update(
        frames=frames,
        root_pos=root.astype(np.float32),
        dof_positions=q.astype(np.float32),
        dof_velocities=np.gradient(q, dt, axis=0).astype(np.float32),
        tips_world=lowest.astype(np.float32),
        contacts_world=lowest.astype(np.float32),
        planted_points_world=support.astype(np.float32),
        support_patch_world=patch.astype(np.float32),
        ankles_world=ankles.astype(np.float32),
        knees_world=knees.astype(np.float32),
        contacts=contacts,
        source_contacts=source_ct,
        physical_height_contacts=height_ok,
        support_speed=support_speed.astype(np.float32),
        contact_anchor_world=anchors.astype(np.float32),
        root_contact_offset=root_delta.astype(np.float32),
        contact_height_tolerance=np.array(CONTACT_HEIGHT),
        contact_speed_tolerance=np.array(speed_cap),
        contact_patch_softness=np.array(PATCH_SOFTNESS),
        ground_z=np.array(0.0),
        collision_hulls=np.array(a.collision_hulls),
        contact_lock_base=np.array(a.motion),
    )
    np.savez(a.out, **out)

    persistent = source_ct[1:] & source_ct[:-1]
    step = np.linalg.norm(np.diff(support, axis=0), axis=2)[persistent]
    ae = np.linalg.norm(support[source_ct] - anchors[source_ct], axis=1)
    dq = np.abs(q[:, :12] - q_ref[:, :12])
    print(f"[[ stance support step mean/p95/max: {step.mean()*1000:.3f}/"
          f"{np.percentile(step,95)*1000:.3f}/{step.max()*1000:.3f} mm")
    print(f"[[ stance anchor error mean/p95/max: {ae.mean()*1000:.3f}/"
          f"{np.percentile(ae,95)*1000:.3f}/{ae.max()*1000:.3f} mm")
    print(f"[[ collision z min {lowest[:,:,2].min()*1000:+.2f} mm | stance mean "
          f"{lowest[source_ct][:,2].mean()*1000:+.2f} mm | schedule match "
          f"{100*(contacts == source_ct).mean():.1f}%")
    print(f"[[ visual preservation: leg delta mean/max {np.degrees(dq.mean()):.2f}/"
          f"{np.degrees(dq.max()):.2f} deg | root correction mean/max "
          f"{np.linalg.norm(root_delta,axis=1).mean()*1000:.1f}/"
          f"{np.linalg.norm(root_delta,axis=1).max()*1000:.1f} mm")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
