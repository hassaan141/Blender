"""Stage 4.2 - how much torque does each v4 joint actually need to hold gravity?

Full-tree URDF forward kinematics with the real link masses/COMs. For every frame
of a reference clip (and its authored base orientation) it computes, per actuated
joint, the static gravity torque about that joint's own axis:

    tau_j = sum over the subtree below j of  ( (com_i - p_j) x m_i g ) . axis_j

and the effective inertia about the same axis (parallel-axis over the subtree):

    I_j = sum over subtree of  ( I_i,axis + m_i * d_perp^2 )

From those it proposes an actuator model, not eyeballed numbers:

    effort  = SAFETY * max|tau|            (headroom for dynamics on top of statics)
    Kp      = max|tau| / SAG               (worst-case steady-state droop <= SAG rad)
    Kd      = 2 * zeta * sqrt(Kp * I_j)    (critical damping at that Kp)

    python3 stage4/actuator_analysis.py --urdf <urdf> --motion <clip.npz>
"""
import argparse, sys, os
import numpy as np
import xml.etree.ElementTree as ET

G = np.array([0.0, 0.0, -9.81])
SAFETY = 2.0       # effort headroom over the static requirement
SAG = 0.05         # rad of allowed steady-state droop under worst-case gravity
ZETA = 1.0         # critical damping


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])


def axis_rot(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def quat_to_mat(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


class Model:
    def __init__(self, path):
        r = ET.parse(path).getroot()
        self.J, self.L = {}, {}
        for ln in r.findall("link"):
            i = ln.find("inertial")
            m = float(i.find("mass").get("value")) if i is not None else 0.0
            o = i.find("origin") if i is not None else None
            com = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            it = i.find("inertia") if i is not None else None
            Im = np.zeros((3, 3))
            if it is not None:
                ixx, iyy, izz = float(it.get("ixx")), float(it.get("iyy")), float(it.get("izz"))
                ixy, ixz, iyz = float(it.get("ixy")), float(it.get("ixz")), float(it.get("iyz"))
                Im = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            self.L[ln.get("name")] = dict(m=m, com=com, I=Im)
        for j in r.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0])
            self.J[j.get("name")] = dict(xyz=xyz, R=rpy_to_mat(*rpy), axis=axis,
                                         parent=j.find("parent").get("link"),
                                         child=j.find("child").get("link"))
        self.children = {}
        for n, j in self.J.items():
            self.children.setdefault(j["parent"], []).append(n)

    def fk(self, q, R0, p0=np.zeros(3)):
        """World frame of every link + world position/axis of every joint."""
        frames = {"origin": (R0, p0)}
        jinfo = {}
        stack = ["origin"]
        while stack:
            par = stack.pop()
            Rp, pp = frames[par]
            for jn in self.children.get(par, []):
                J = self.J[jn]
                pj = pp + Rp @ J["xyz"]
                Rj = Rp @ J["R"]
                jinfo[jn] = dict(p=pj, axis=(Rj @ J["axis"]) / np.linalg.norm(Rj @ J["axis"]))
                Rc = Rj @ axis_rot(J["axis"], q.get(jn, 0.0))
                frames[J["child"]] = (Rc, pj)
                stack.append(J["child"])
        return frames, jinfo

    def subtree(self, jn):
        out, stack = [], [self.J[jn]["child"]]
        while stack:
            l = stack.pop(); out.append(l)
            for c in self.children.get(l, []):
                stack.append(self.J[c]["child"])
        return out

    def gravity_torque(self, q, R0):
        frames, jinfo = self.fk(q, R0)
        tau, Ieff = {}, {}
        for jn in self.J:
            p, a = jinfo[jn]["p"], jinfo[jn]["axis"]
            t = 0.0; Ia = 0.0
            for ln in self.subtree(jn):
                L = self.L[ln]
                if L["m"] <= 0:
                    continue
                Rl, pl = frames[ln]
                com_w = pl + Rl @ L["com"]
                t += np.dot(np.cross(com_w - p, L["m"] * G), a)
                d = np.linalg.norm(np.cross(com_w - p, a))     # perpendicular distance
                Ia += float(a @ (Rl @ L["I"] @ Rl.T) @ a) + L["m"] * d * d
            tau[jn], Ieff[jn] = t, Ia
        return tau, Ieff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    md = Model(a.urdf)
    m = np.load(a.motion, allow_pickle=True)
    dof = m["dof_positions"].astype(float)
    names = [str(n) for n in m["dof_names"]]
    quat = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = dof.shape[0]

    tmax = {n: 0.0 for n in md.J}
    imax = {n: 0.0 for n in md.J}
    for i in range(0, T, a.stride):
        q = {names[k]: dof[i, k] for k in range(len(names))}
        tau, Ie = md.gravity_torque(q, quat_to_mat(quat[i]))
        for n in md.J:
            tmax[n] = max(tmax[n], abs(tau[n]))
            imax[n] = max(imax[n], Ie[n])

    # current model, as configured today
    cur_eff = {}; cur_kp = {}; cur_kd = {}; cur_vel = {}
    for n in md.J:
        if n.endswith("_SY_J"):
            cur_eff[n], cur_kp[n], cur_kd[n], cur_vel[n] = 3.0, 1.15, 0.092, 10.0
        elif n.endswith("_SP_J"):
            cur_eff[n], cur_kp[n], cur_kd[n], cur_vel[n] = 3.0, 1.82, 0.146, 10.0
        elif n.endswith("_knee"):
            cur_eff[n], cur_kp[n], cur_kd[n], cur_vel[n] = 3.0, 2.10, 0.166, 10.0
        elif "ear" in n:
            cur_eff[n], cur_kp[n], cur_kd[n], cur_vel[n] = 1.0, 0.6, 0.05, 8.0
        else:
            cur_eff[n], cur_kp[n], cur_kd[n], cur_vel[n] = 1.5, 1.5, 0.12, 8.0

    order = [n for n in names]
    print(f"[[ gravity-torque analysis over {len(range(0,T,a.stride))} poses of "
          f"{os.path.basename(a.motion)}\n")
    hdr = (f"{'joint':18s} {'max|tau_g|':>10s} {'I_eff':>10s} | {'eff':>5s} {'Kp':>6s} "
           f"{'Kd':>6s} {'ok?':>4s} | {'->eff':>6s} {'->Kp':>7s} {'->Kd':>7s}")
    print(hdr); print("-" * len(hdr))
    prop = {}
    for n in order:
        t, Ie = tmax[n], imax[n]
        eff = max(SAFETY * t, 0.25)
        kp = max(t / SAG, 1.0)
        kd = 2.0 * ZETA * np.sqrt(kp * max(Ie, 1e-9))
        prop[n] = (eff, kp, kd)
        ok = "YES" if cur_eff[n] >= t * 1.05 else "NO"
        print(f"{n:18s} {t:10.3f} {Ie:10.2e} | {cur_eff[n]:5.2f} {cur_kp[n]:6.2f} "
              f"{cur_kd[n]:6.3f} {ok:>4s} | {eff:6.2f} {kp:7.2f} {kd:7.3f}")

    print(f"\n[[ insufficient today: "
          f"{[n for n in order if cur_eff[n] < tmax[n]*1.05]}")

    def grp(pred):
        ns = [n for n in order if pred(n)]
        e = max(prop[n][0] for n in ns); k = max(prop[n][1] for n in ns)
        d = max(prop[n][2] for n in ns)
        return ns, e, k, d
    print("\n[[ proposed grouped values (max over each group):")
    for label, pred in (("SY", lambda n: n.endswith("_SY_J")),
                        ("SP", lambda n: n.endswith("_SP_J")),
                        ("knee", lambda n: n.endswith("_knee")),
                        ("head", lambda n: n.startswith("head_")),
                        ("tail", lambda n: n.startswith("tail_")),
                        ("ears", lambda n: "ear" in n)):
        ns, e, k, d = grp(pred)
        print(f"     {label:5s} effort {e:6.2f} Nm  Kp {k:7.2f}  Kd {d:6.3f}")


main()
