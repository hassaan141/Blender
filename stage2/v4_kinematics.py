"""Exact v4 forward kinematics + joint limits, straight from the v4 URDF.

This is the single source of truth for the TARGET robot in Stage 2. It reuses the
same math as scripts/retarget.py (URDF axis vectors used directly, so joint angles
come out already in URDF sign convention) and extends it to all 21 v4 joints,
including the head, tail and the two ears.

Pure numpy, importable from both Blender and system python (no bpy, no scipy).

Frame convention (URDF/robot): +X forward, +Y left, +Z up. Origin = floating base.
The shank tip is an animation landmark.  Physical support is computed from the
orientation-dependent collision hull, never from a fixed paw offset.
"""
import numpy as np
import xml.etree.ElementTree as ET

SHANK_LEN = 0.120          # knee pivot -> animator/IK shank-tip reference
# Legacy visualization-only offset retained for old reports.  Stage 2 contact
# solving must use ``support_hull`` in ``leg_points`` below.
PAW_DROP = 0.0288
LEGS = ["fl", "fr", "bl", "br"]

# Output DOF order — identical to bake_conform.DOF_ORDER_21 so the baked .npz and
# the physical-bone bake stay byte-compatible with the Stage 1 tooling.
DOF_ORDER = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
             "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
             "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
             "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
HEAD_CHAIN = ["head_pitch_joint", "head_yaw", "head_roll"]
TAIL_CHAIN = ["tail_pitch", "tail_yaw"]
LEAR_CHAIN = ["l_ear_pitch", "l_ear_roll"]
REAR_CHAIN = ["r_ear_pitch", "r_ear_roll"]


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])


def axis_rot(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


class V4Kin:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.j = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0])
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else -np.pi
            hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else np.pi
            self.j[j.get("name")] = dict(xyz=xyz, R=rpy_to_mat(*rpy), axis=axis, lo=lo, hi=hi,
                                         parent=j.find("parent").get("link"),
                                         child=j.find("child").get("link"))

    # ---- legs (3-DOF) ---------------------------------------------------
    def leg_chain(self, leg):
        return [f"{leg}_SY_J", f"{leg}_SP_J", f"{leg}_knee"]

    def leg_points(self, leg, q, support_hull=None, world_R=None,
                   support_local=None, support_softness=0.0):
        """Physical leg points in the floating-base/origin frame.

        Returns ``(sp, knee, ankle, support, knee_R)``. ``ankle`` is the
        shank-tip animation reference. If ``support_hull`` is supplied, support
        is the collision-hull vertex that is lowest under ``world_R @ knee_R``;
        this is the physical support point for the current world orientation.
        ``support_softness`` (metres) returns a smooth centroid of the active
        collision patch. This avoids the non-physical XY jump produced when two
        equally-low hull vertices exchange ``argmin`` while retaining exact hull
        geometry; height checks still use softness zero.
        Without a hull, support aliases ankle for backward-compatible callers
        that only consume SP/knee/link orientation.
        """
        T_R, T_p = np.eye(3), np.zeros(3)
        points = {}
        for idx, (name, qi) in enumerate(zip(self.leg_chain(leg), q)):
            J = self.j[name]
            T_p = T_p + T_R @ J["xyz"]
            if idx == 1:
                points["sp"] = T_p.copy()
            elif idx == 2:
                points["knee"] = T_p.copy()
            T_R = T_R @ J["R"] @ axis_rot(J["axis"], qi)
        ankle = T_p + T_R @ np.array([0, 0, -SHANK_LEN])
        if support_local is not None:
            support = T_p + T_R @ np.asarray(support_local, dtype=float)
        elif support_hull is None:
            support = ankle.copy()
        else:
            hull = np.asarray(support_hull, dtype=float)
            Rw = T_R if world_R is None else np.asarray(world_R) @ T_R
            z = (hull @ Rw.T)[:, 2]
            if support_softness > 0:
                w = np.exp(-(z - z.min()) / support_softness)
                local = (w[:, None] * hull).sum(0) / w.sum()
            else:
                local = hull[int(np.argmin(z))]
            support = T_p + T_R @ local
        return points["sp"], points["knee"], ankle, support, T_R

    def lowest_support_local(self, leg, q, support_hull, world_R=None):
        """Return the active collision-support vertex in knee-link coordinates."""
        *_, knee_R = self.leg_points(leg, q)
        Rw = knee_R if world_R is None else np.asarray(world_R) @ knee_R
        hull = np.asarray(support_hull, dtype=float)
        return hull[int(np.argmin((hull @ Rw.T)[:, 2]))].copy()

    def leg_fk(self, leg, q):
        """Backward-compatible ``(shank_tip, knee_R, knee_p)`` tuple."""
        _, knee, ankle, _, knee_R = self.leg_points(leg, q)
        return ankle, knee_R, knee

    def sy_pos(self, leg):
        return self.j[f"{leg}_SY_J"]["xyz"]

    def sp_pos_zero(self, leg):
        """SP pivot at the zero pose; includes the real SY->SP offset once."""
        return self.leg_points(leg, np.zeros(3))[0]

    def upper_len(self, leg):
        return np.linalg.norm(self.j[f"{leg}_knee"]["xyz"])

    def leg_reach(self, leg):
        return np.linalg.norm(self.j[f"{leg}_knee"]["xyz"]) + SHANK_LEN

    def leg_limits(self, leg):
        return np.array([[self.j[n]["lo"], self.j[n]["hi"]] for n in self.leg_chain(leg)])

    # ---- serial rotation chains (head / tail / ears) --------------------
    def chain_rot(self, names, q):
        """Rotation of the terminal frame w.r.t the origin, joint xyz ignored
        (we only match orientation for these expressive chains)."""
        R = np.eye(3)
        for name, qi in zip(names, q):
            J = self.j[name]
            R = R @ J["R"] @ axis_rot(J["axis"], qi)
        return R

    def chain_limits(self, names):
        return np.array([[self.j[n]["lo"], self.j[n]["hi"]] for n in names])

    # ---- convenience ----------------------------------------------------
    def all_limits(self):
        return np.array([[self.j[n]["lo"], self.j[n]["hi"]] for n in DOF_ORDER])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def quat_to_mat(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rotvec_of(R):
    """Log map: rotation matrix -> axis-angle vector (for residuals)."""
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return v / (2*np.sin(ang)) * ang


if __name__ == "__main__":
    import sys
    k = V4Kin(sys.argv[1])
    print("joints:", len(k.j))
    for leg in LEGS:
        tip, _, kp = k.leg_fk(leg, np.zeros(3))
        print(f"  {leg}: sy={np.round(k.sy_pos(leg),4)} zero-pose tip={np.round(tip,4)} "
              f"reach={k.leg_reach(leg):.4f}")
    print("head zero R:\n", np.round(k.chain_rot(HEAD_CHAIN, np.zeros(3)), 3))
    print("limits (21):\n", np.round(k.all_limits(), 2))
