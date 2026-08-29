"""Stage 4 - derive a physically-grounded dynamic Cheeky from the trusted Stage 3 npz.

The Stage 3 reference is IMMUTABLE; this writes a separate file.

Fix #1 (proven contact bug): Stage 2's vertical ground placement resolved "planted
foot on z=0" using PAW_CONTACT_LOCAL, a single fixed point that is biased +9 mm
(front) to +67 mm (hind) against the true collision hull. Every planted frame was
therefore grounded to the wrong height. Re-solve the root height using the real
convex-hull lowest point, changing ONLY root z - joints, root xy and root
orientation are copied through untouched.
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0, "/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

U = ("/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/"
     "bingo_urdf_w_ear_joints_physics.urdf")


def gsmooth(x, sig):
    if sig <= 0: return x
    r = int(np.ceil(3*sig)); w = np.exp(-0.5*(np.arange(-r, r+1)/sig)**2); w /= w.sum()
    return np.convolve(np.pad(x, (r, r), mode="edge"), w, "valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--contacts", default=None, help="Stage 2 solver npz with the intended schedule")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--metadata-only", action="store_true",
                    help="refresh physical paw/contact arrays without changing the validated trajectory")
    a = ap.parse_args()

    kin = V4Kin(U); cm = ContactModel()
    m = np.load(a.motion, allow_pickle=True)
    d = {k: m[k] for k in m.files}
    q = m["dof_positions"].astype(float); names = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files
          else m["body_positions"][:, 0]).astype(float).copy()
    rq = (m["root_quat"] if "root_quat" in m.files
          else m["body_rotations"][:, 0]).astype(float)
    T = len(q)

    def paw_geom(i, dz=0.0):
        R0 = quat_to_mat(rq[i]); p0 = rp[i] + np.array([0, 0, dz])
        patch, low = [], []
        for leg in LEGS:
            qq = [q[i, names.index(f"{leg}_{s}")]
                  for s in ("SY_J", "SP_J", "knee")]
            soft = kin.leg_points(
                leg, qq, support_hull=cm.hull[f"{leg}_knee"],
                world_R=R0, support_softness=0.001)[3]
            exact = kin.leg_points(
                leg, qq, support_hull=cm.hull[f"{leg}_knee"], world_R=R0)[3]
            patch.append(p0 + R0 @ soft)
            low.append(p0 + R0 @ exact)
        return np.asarray(patch), np.asarray(low)

    def paw_z(i, dz=0.0):
        return paw_geom(i, dz)[1][:, 2]

    PZ0 = np.array([paw_z(i) for i in range(T)])

    # intended planted set: prefer the Stage 2 solver schedule, else "lowest paw"
    if a.contacts and os.path.exists(a.contacts):
        ct = np.load(a.contacts, allow_pickle=True)["contacts"]
        if len(ct) != T: ct = None
    else:
        ct = None
    if ct is None:
        ct = PZ0 < (PZ0.min(1)[:, None] + 0.010)   # feet within 10 mm of the lowest

    if a.metadata_only:
        # The reference has already passed the dynamic replay.  Even a tiny
        # root-height rewrite changes touchdown timing, so only refresh the
        # derived collision/contact data in this mode.
        dz = np.zeros(T)
    else:
        # per-frame dz putting the intended planted paws on the floor
        dz = np.full(T, np.nan)
        for i in range(T):
            s = ct[i]
            if s.any():
                dz[i] = -PZ0[i, s].min()
        ok = ~np.isnan(dz)
        idx = np.arange(T)
        dz = np.interp(idx, idx[ok], dz[ok]) if ok.any() else np.zeros(T)
        dz = gsmooth(dz, a.sigma)
        # never leave a paw through the floor
        pen = np.array([min(0.0, (PZ0[i] + dz[i]).min()) for i in range(T)])
        dz = dz - gsmooth(pen, 2.0)
        # Smoothing deliberately spreads the lift over neighbouring frames, but it
        # can under-correct the deepest frame.  Finish with the minimum exact lift
        # needed by each frame so the documented no-penetration guarantee is true.
        remaining = np.minimum(0.0, (PZ0 + dz[:, None]).min(axis=1))
        dz = dz - remaining

    rp_new = rp.copy(); rp_new[:, 2] += dz
    PZ1 = np.array([paw_z(i, dz[i]) for i in range(T)])
    if "body_positions" in d:
        bp = d["body_positions"].astype(float).copy(); bp[:, :, 2] += dz[:, None]
        d["body_positions"] = bp.astype(np.float32)
    d["root_pos"] = rp_new.astype(np.float32)
    d["root_quat"] = rq.astype(np.float32)
    d["stage4_dz"] = dz.astype(np.float32)

    # Recompute collision contact points from the final q/root, because Stage-4
    # takeoff strokes may have changed the legs after cached Stage-2 points were
    # created. Merely shifting old points would leave contact timing stale.
    geom = [paw_geom(i, dz[i]) for i in range(T)]
    patches = np.asarray([x[0] for x in geom])
    lows = np.asarray([x[1] for x in geom])
    support = patches.copy(); support[:, :, 2] = lows[:, :, 2]
    d["support_patch_world"] = patches.astype(np.float32)
    d["tips_world"] = lows.astype(np.float32)
    d["contacts_world"] = lows.astype(np.float32)
    d["planted_points_world"] = support.astype(np.float32)
    height_tol = (float(m["contact_height_tolerance"])
                  if "contact_height_tolerance" in m.files else 0.005)
    speed_tol = (float(m["contact_speed_tolerance"])
                 if "contact_speed_tolerance" in m.files else 0.014)
    physical = PZ1 <= height_tol
    source_contacts = (np.asarray(m["source_contacts"], bool)
                       if "source_contacts" in m.files else np.asarray(ct, bool))
    support_speed = np.linalg.norm(
        np.gradient(support, 1.0 / float(m["fps"]), axis=0), axis=2)
    contacts = physical & (source_contacts | (support_speed <= speed_tol))
    d["contacts"] = contacts
    d["source_contacts"] = source_contacts
    d["physical_height_contacts"] = physical
    d["support_speed"] = support_speed.astype(np.float32)
    d["contact_height_tolerance"] = np.array(height_tol)
    d["contact_speed_tolerance"] = np.array(speed_tol)
    np.savez(a.out, **d)

    s0 = (PZ0 < 0.005).sum(1); s1 = (PZ1 < 0.005).sum(1)
    print(f"[[ root z shift: mean {dz.mean()*1000:+.1f} mm  range {dz.min()*1000:+.1f}..{dz.max()*1000:+.1f}")
    print(f"[[ supports/frame : {s0.mean():.2f} -> {s1.mean():.2f} of 4")
    print(f"[[ zero-support frames: {(s0==0).sum()} -> {(s1==0).sum()} of {T}")
    print(f"[[ min paw z     : {PZ0.min()*1000:+.1f} -> {PZ1.min()*1000:+.1f} mm")
    print(f"[[ joints, root xy and root orientation copied through UNCHANGED")
    print(f"[[ wrote {a.out}")


main()
