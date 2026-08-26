"""Orientation-aware contact/support model built from the real collision hulls.

Replaces the fixed PAW_CONTACT_LOCAL point. For a link with world pose (p, R) the
lowest point of its collision shape is

    min over hull vertices v of   (R @ v + p).z

which is exact for any orientation and is the same geometry PhysX collides against
(the URDF importer uses convex_decomp=False, so each collision shape is the convex
hull of its STL).

Pure numpy; importable from Isaac's python.
"""
import numpy as np

HULLS_NPZ = "/home/hassaan/Bingo/Blender/stage4/out/collision_hulls.npz"
LEGS = ["fl", "fr", "bl", "br"]
PAW_LINKS = {l: f"{l}_knee" for l in LEGS}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


class ContactModel:
    def __init__(self, path=HULLS_NPZ):
        d = np.load(path)
        self.hull = {k: d[k].astype(np.float64) for k in d.files}

    def links(self):
        return list(self.hull)

    def lowest(self, link, pos, quat):
        """(min world z, world position of that lowest vertex)."""
        if link not in self.hull:
            return np.inf, None
        w = self.hull[link] @ quat_to_R(quat).T + np.asarray(pos)
        i = int(np.argmin(w[:, 2]))
        return float(w[i, 2]), w[i]

    def lowest_z(self, link, pos, quat):
        return self.lowest(link, pos, quat)[0]

    def paw_heights(self, body_pos, body_quat, body_names):
        """Lowest collision-geometry z of each paw (the knee/shank link), metres."""
        out = np.zeros(4)
        for k, l in enumerate(LEGS):
            ln = PAW_LINKS[l]
            i = body_names.index(ln)
            out[k] = self.lowest_z(ln, body_pos[i], body_quat[i])
        return out

    def all_link_heights(self, body_pos, body_quat, body_names):
        """Lowest collision z for every link we have a hull for."""
        out = {}
        for ln in self.hull:
            if ln in body_names:
                i = body_names.index(ln)
                out[ln] = self.lowest_z(ln, body_pos[i], body_quat[i])
        return out

    def support_summary(self, body_pos, body_quat, body_names, thresh=0.005):
        """(paw z array, n paw contacts, dict of non-paw links touching/penetrating)."""
        pz = self.paw_heights(body_pos, body_quat, body_names)
        allz = self.all_link_heights(body_pos, body_quat, body_names)
        paw_links = set(PAW_LINKS.values())
        other = {k: v for k, v in allz.items() if k not in paw_links and v < thresh}
        return pz, int((pz < thresh).sum()), other
