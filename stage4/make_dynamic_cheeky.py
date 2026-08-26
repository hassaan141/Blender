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
    a = ap.parse_args()

    kin = V4Kin(U); cm = ContactModel()
    m = np.load(a.motion, allow_pickle=True)
    d = {k: m[k] for k in m.files}
    q = m["dof_positions"].astype(float); names = [str(x) for x in m["dof_names"]]
    rp = m["body_positions"][:, 0].astype(float).copy()
    rq = m["body_rotations"][:, 0].astype(float)
    T = len(q)

    def paw_z(i, dz=0.0):
        R0 = quat_to_mat(rq[i]); p0 = rp[i] + np.array([0, 0, dz]); out = []
        for leg in LEGS:
            R, p = R0, p0.copy()
            qq = [q[i, names.index(f"{leg}_{s}")] for s in ("SY_J", "SP_J", "knee")]
            for nm, qi in zip(kin.leg_chain(leg), qq):
                J = kin.j[nm]; p = p + R @ J["xyz"]; R = R @ J["R"] @ axis_rot(J["axis"], qi)
            w = cm.hull[f"{leg}_knee"] @ R.T + p
            out.append(w[:, 2].min())
        return np.array(out)

    PZ0 = np.array([paw_z(i) for i in range(T)])

    # intended planted set: prefer the Stage 2 solver schedule, else "lowest paw"
    if a.contacts and os.path.exists(a.contacts):
        ct = np.load(a.contacts, allow_pickle=True)["contacts"]
        if len(ct) != T: ct = None
    else:
        ct = None
    if ct is None:
        ct = PZ0 < (PZ0.min(1)[:, None] + 0.010)   # feet within 10 mm of the lowest

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

    rp_new = rp.copy(); rp_new[:, 2] += dz
    PZ1 = np.array([paw_z(i, dz[i]) for i in range(T)])
    bp = d["body_positions"].astype(float).copy(); bp[:, 0, 2] += dz
    d["body_positions"] = bp.astype(np.float32)
    d["root_pos"] = rp_new.astype(np.float32)
    d["root_quat"] = rq.astype(np.float32)
    d["stage4_dz"] = dz.astype(np.float32)
    np.savez(a.out, **d)

    s0 = (PZ0 < 0.005).sum(1); s1 = (PZ1 < 0.005).sum(1)
    print(f"[[ root z shift: mean {dz.mean()*1000:+.1f} mm  range {dz.min()*1000:+.1f}..{dz.max()*1000:+.1f}")
    print(f"[[ supports/frame : {s0.mean():.2f} -> {s1.mean():.2f} of 4")
    print(f"[[ zero-support frames: {(s0==0).sum()} -> {(s1==0).sum()} of {T}")
    print(f"[[ min paw z     : {PZ0.min()*1000:+.1f} -> {PZ1.min()*1000:+.1f} mm")
    print(f"[[ joints, root xy and root orientation copied through UNCHANGED")
    print(f"[[ wrote {a.out}")


main()
