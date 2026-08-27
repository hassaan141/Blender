"""Planted-paw slip, measured as the MATERIAL velocity of the contacting point.

The naive metric - "how far did the lowest hull vertex move between frames" -
overstates slip on a convex paw, because a paw that ROLLS without slipping still
moves its contact point along the floor. That is physically legal and costs no
friction. What is illegal is the material point in contact having a nonzero
horizontal velocity:

    v_contact = d/dt ( R(t) c + p(t) )  evaluated at the contact vertex c

computed from a central difference of the link pose. For rolling contact this is
~0; for skating it is the skate speed. Reported per leg and per clip in mm/frame
(at the clip fps) so it is directly comparable to Ashley's authored foot lock.

Ashley's own performance is the reference bar: she animates with square foot
controls parented to the world, so her planted toes move 0.03-0.11 mm/frame.

  python3 stage2/slip_audit.py motions/laidback_v4.npz --contacts stage2/out/laidback_contacts.npz
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "stage4"))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")
MAP = {"fl": "aFL", "fr": "aFR", "bl": "aBL", "br": "aBR"}


def link_poses(path, links):
    """World (R, p) per frame for the requested links, by URDF FK."""
    kin = V4Kin(URDF)
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)
    m = np.load(path, allow_pickle=True)
    q = m["dof_positions"].astype(float); nm = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q)
    R = {l: np.zeros((T, 3, 3)) for l in links}
    P = {l: np.zeros((T, 3)) for l in links}
    for i in range(T):
        d = {n: q[i, nm.index(n)] for n in nm}
        fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
                st.append(J["child"])
        for l in links:
            R[l][i], P[l][i] = fr[l]
    return R, P, float(m["fps"]), T


def slip(path, contacts=None, tol=0.005):
    links = [f"{l}_knee" for l in LEGS]
    R, P, fps, T = link_poses(path, links)
    cm = ContactModel()
    dt = 1.0 / fps
    # The motion's own source_contacts is authoritative: it is already in robot-leg
    # order and already follows any retiming the clip has been through, whereas the
    # Stage-2 contacts file is still at the ORIGINAL frame count and in Ashley-leg
    # order. Fall back to the file only when the motion does not carry the field.
    mm = np.load(path, allow_pickle=True)
    if "stage4_planted" in mm.files and len(mm["stage4_planted"]) == T:
        ct = np.asarray(mm["stage4_planted"], bool)   # the enforced schedule wins
    elif "source_contacts" in mm.files and len(mm["source_contacts"]) == T:
        ct = np.asarray(mm["source_contacts"], bool)
    elif contacts is not None:
        c = np.load(contacts, allow_pickle=True)
        order = [str(x) for x in c["aleg_order"]]
        ct = np.stack([c["contacts"][:, order.index(MAP[l])] for l in LEGS], 1)
        if len(ct) != T:
            idx = np.clip(np.round(np.linspace(0, len(ct) - 1, T)).astype(int), 0, len(ct) - 1)
            ct = ct[idx]
    else:
        ct = None
    out = {}
    for k, l in enumerate(LEGS):
        ln = f"{l}_knee"
        H = cm.hull[ln]
        Rl, Pl = R[ln], P[ln]
        dR = np.gradient(Rl, dt, axis=0); dP = np.gradient(Pl, dt, axis=0)
        v = np.zeros(T); geom = np.zeros(T); low = np.zeros(T)
        prev = None
        for i in range(T):
            w = H @ Rl[i].T + Pl[i]
            j = int(np.argmin(w[:, 2])); low[i] = w[j, 2]
            cvec = H[j]
            vc = dR[i] @ cvec + dP[i]                     # material velocity
            v[i] = np.linalg.norm(vc[:2])
            geom[i] = np.linalg.norm(w[j, :2] - prev) if prev is not None else 0.0
            prev = w[j, :2]
        phys = low < tol
        sel = (ct[:, k] & phys) if ct is not None else phys
        out[l] = dict(mask=sel, v=v, geom=geom, low=low)
    return out, fps, T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("motion"); ap.add_argument("--contacts", default=None)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    o, fps, T = slip(a.motion, a.contacts, a.tol)
    print(f"{os.path.basename(a.motion)}  T={T} @ {fps:g} fps  "
          f"[contact-point MATERIAL slip, mm per frame]")
    allv = []
    for l in LEGS:
        d = o[l]; s = d["mask"]
        if not s.any():
            print(f"  {l}: no stance frames"); continue
        vv = d["v"][s] / fps * 1000.0
        gg = d["geom"][s] * 1000.0
        allv.append(vv)
        print(f"  {l}: stance {int(s.sum()):4d} fr | material mean {vv.mean():6.2f} "
              f"p95 {np.percentile(vv,95):6.2f} max {vv.max():7.2f} | "
              f"geometric mean {gg.mean():6.2f} max {gg.max():7.2f}")
    if allv:
        A = np.concatenate(allv)
        print(f"  ALL: material mean {A.mean():.2f} mm/frame  p95 {np.percentile(A,95):.2f}  "
              f"max {A.max():.2f}  total {A.sum():.0f} mm")
