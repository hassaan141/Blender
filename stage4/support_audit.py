"""Whole-body support audit: static (CoM) and dynamic (ZMP) margin against the
support polygon formed by EVERY link touching the floor - not just the four paws.

Why this exists: stage4/balance_audit.py and stage4/dynamic_audit.py build the
support polygon from the four knee (paw) hulls only. That is correct for standing
clips, but Eccentric is an authored SIT - Ashley holds the body 35.9 mm above her
ground plane for all 370 frames, and the v4 retarget correctly puts the `origin`
torso hull on the floor with the hind paws 60-80 mm up. Scored paws-only, Eccentric
reads 0/370 statically stable purely because its real support surface is invisible
to the metric. Scored whole-body it is a large, very stable base.

Support set = every collision-hull vertex of every link within --tol of z=0.
Polygon    = 2D convex hull (monotone chain) of those vertices.
Margin     = signed distance from the point (CoM_xy, or ZMP_xy) to the polygon
             boundary; positive inside.

ZMP uses the cart-table approximation ZMP = C_xy - C_z * a_xy / (a_z + g), the same
form as dynamic_audit.py, so numbers are comparable between the two tools.

  python3 stage4/support_audit.py motions/eccentric_v4.npz [--frames 0-40]
"""
import argparse, itertools, os, sys
import numpy as np
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, HERE)
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")


def hull2d(P):
    """Monotone-chain convex hull of an (N,2) point set, CCW, no duplicates."""
    P = np.unique(np.round(P, 9), axis=0)
    if len(P) <= 2:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out
    lo = half(P); up = half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def margin(P, c):
    """Signed distance from c to the boundary of polygon P (positive = inside).
    Degenerate supports fall back to distance-to-segment / distance-to-point, and
    are always reported negative because a line or a point cannot be a stable base."""
    if len(P) >= 3:
        ins = True; dmin = 1e9
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]
            e = b - a; n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n) + 1e-12
            d = float(np.dot(c - a, n))
            if d < 0: ins = False
            dmin = min(dmin, abs(d))
        return dmin if ins else -dmin
    if len(P) == 2:
        a, b = P; e = b - a; L = np.linalg.norm(e) + 1e-12
        return -abs(float(e[0]/L*(c[1]-a[1]) - e[1]/L*(c[0]-a[0])))
    if len(P) == 1:
        return -float(np.linalg.norm(c - P[0]))
    return -9.99


def fk_frames(kin, ch, dof, names, root_pos, root_quat, i):
    d = {n: dof[i, names.index(n)] for n in names}
    fr = {"origin": (quat_to_mat(root_quat[i]), root_pos[i].copy())}
    st = ["origin"]
    while st:
        par = st.pop(); Rp, pp = fr[par]
        for jn in ch.get(par, []):
            J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
            fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
            st.append(J["child"])
    return fr


def runs(idx):
    out = []
    for _, grp in itertools.groupby(enumerate(idx), lambda t: t[1] - t[0]):
        g = [x[1] for x in grp]; out.append((g[0], g[-1]))
    return out


def analyse(path, tol=0.005, verbose_range=None):
    kin = V4Kin(URDF); cm = ContactModel()
    root = ET.parse(URDF).getroot(); LM = {}
    for ln in root.findall("link"):
        inr = ln.find("inertial"); o = inr.find("origin")
        LM[ln.get("name")] = (float(inr.find("mass").get("value")),
                              np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]))
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)

    m = np.load(path, allow_pickle=True)
    q = m["dof_positions"].astype(float); nm = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q); fps = float(m["fps"])

    COM = np.zeros((T, 3)); polys = []; touch = []; minz = np.zeros(T)
    for i in range(T):
        fr = fk_frames(kin, ch, q, nm, rp, rq, i)
        M = 0.0; C = np.zeros(3)
        for ln, (R, p) in fr.items():
            mm, cc = LM[ln]; M += mm; C += mm * (p + R @ cc)
        COM[i] = C / M
        pts = []; names_touch = []; lowest = 1e9
        for ln, (R, p) in fr.items():
            if ln not in cm.hull:
                continue
            w = cm.hull[ln] @ R.T + p
            lowest = min(lowest, float(w[:, 2].min()))
            sel = w[w[:, 2] < tol]
            if len(sel):
                pts.append(sel[:, :2]); names_touch.append(ln)
        minz[i] = lowest
        polys.append(hull2d(np.concatenate(pts, 0)) if pts else np.zeros((0, 2)))
        touch.append(names_touch)

    stat = np.array([margin(polys[i], COM[i, :2]) for i in range(T)])
    dt = 1.0 / fps
    V = np.gradient(COM, dt, axis=0); A = np.gradient(V, dt, axis=0)
    az = A[:, 2] + 9.81
    mu = np.linalg.norm(A[:, :2], axis=1) / np.maximum(az, 1e-6)
    zmp = COM[:, :2] - COM[:, 2:3] * A[:, :2] / np.maximum(az, 1e-6)[:, None]
    dyn = np.array([margin(polys[i], zmp[i]) for i in range(T)])
    nlink = np.array([len(t) for t in touch])
    area = np.array([polygon_area(p) for p in polys])

    name = os.path.basename(path)
    print(f"{name}: T={T} @ {fps:g} fps   [whole-body support, tol {tol*1000:.0f} mm]")
    print(f"  STATIC : CoM inside support {int((stat>0).sum())}/{T} = {100*(stat>0).mean():.0f}%"
          f" | median margin {np.median(stat)*1000:+.1f} mm | worst {stat.min()*1000:+.0f} mm")
    print(f"  DYNAMIC: ZMP inside support {int((dyn>0).sum())}/{T} = {100*(dyn>0).mean():.0f}%"
          f" | median margin {np.median(dyn)*1000:+.1f} mm | worst {dyn.min()*1000:+.0f} mm")
    print(f"  required mu: mean {mu.mean():.2f} p90 {np.percentile(mu,90):.2f} max {mu.max():.2f}"
          f" | frames mu>1: {int((mu>1).sum())} | free-fall frames (a_z+g<0): {int((az<0).sum())}")
    print(f"  touching links/frame: mean {nlink.mean():.2f} min {nlink.min()} | "
          f"support area mean {area.mean()*1e4:.1f} cm^2 | min body z {minz.min()*1000:+.1f} mm")
    from collections import Counter
    cnt = Counter(l for t in touch for l in t)
    print("  contact links: " + ", ".join(f"{k} {100*v/T:.0f}%" for k, v in cnt.most_common(8)))
    bad = np.where(dyn < 0)[0]
    if len(bad):
        r = runs(bad)
        print(f"  ZMP-infeasible windows ({len(r)}): " +
              ", ".join(f"{a}-{b}" for a, b in r[:16]) + (" ..." if len(r) > 16 else ""))
    if verbose_range:
        f0, f1 = (int(x) for x in verbose_range.split("-", 1))
        print("  frame nlink static_mm zmp_mm  com_xy_mm            zmp_xy_mm            links")
        for i in range(max(0, f0), min(T - 1, f1) + 1):
            print(f"  {i:5d} {nlink[i]:5d} {stat[i]*1000:+9.1f} {dyn[i]*1000:+8.1f} "
                  f"({COM[i,0]*1000:+7.1f},{COM[i,1]*1000:+7.1f}) "
                  f"({zmp[i,0]*1000:+7.1f},{zmp[i,1]*1000:+7.1f})  {','.join(touch[i])}")
    return dict(static=stat, dyn=dyn, mu=mu, com=COM, zmp=zmp, polys=polys, touch=touch)


def polygon_area(P):
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    analyse(a.motion, a.tol, a.frames)
