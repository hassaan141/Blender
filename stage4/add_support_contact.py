"""Add one short physical paw contact without changing the v4 rig.

The selected paw keeps its authored world XY trajectory while its real collision
support is blended to a requested ground height.  Only that leg's three physical
joints are solved, within the unchanged URDF limits, and all other motion remains
untouched.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "stage2")))
from contact_model import ContactModel, HULLS_NPZ
from v4_kinematics import V4Kin, quat_to_mat

URDF = os.path.abspath(os.path.join(
    HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
    "bingo_urdf_w_ear_joints_physics.urdf"))
LEGS = ["fl", "fr", "bl", "br"]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leg", choices=LEGS, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--ramp", type=int, default=3)
    ap.add_argument("--target-z", type=float, default=0.0, help="metres")
    a = ap.parse_args()

    m = np.load(a.motion, allow_pickle=True)
    out = {k: m[k] for k in m.files}
    q_ref = m["dof_positions"].astype(float); q = q_ref.copy(); T = len(q)
    names = [str(x) for x in m["dof_names"]]
    ids = [names.index(f"{a.leg}_SY_J"), names.index(f"{a.leg}_SP_J"),
           names.index(f"{a.leg}_knee")]
    root = (m["root_pos"] if "root_pos" in m.files
            else m["body_positions"][:, 0]).astype(float)
    quat = (m["root_quat"] if "root_quat" in m.files
            else m["body_rotations"][:, 0]).astype(float)
    Rroot = np.array([quat_to_mat(x) for x in quat])
    kin = V4Kin(URDF); cm = ContactModel(); hull = cm.hull[f"{a.leg}_knee"]

    s=max(0,a.start); e=min(T-1,a.end); ramp=max(1,a.ramp)
    w=np.zeros(T); w[s:e+1]=1.0
    w[s:min(e+1,s+ramp)] = smoothstep(np.linspace(0,1,min(ramp,e-s+1)))
    w[max(s,e-ramp+1):e+1] = smoothstep(np.linspace(1,0,min(ramp,e-s+1)))

    def geom(i, qleg):
        _, _, ankle, patch, _ = kin.leg_points(
            a.leg, qleg, support_hull=hull, world_R=Rroot[i],
            support_softness=0.001)
        low = kin.leg_points(a.leg, qleg, support_hull=hull,
                             world_R=Rroot[i])[3]
        return (root[i] + Rroot[i] @ ankle,
                root[i] + Rroot[i] @ patch,
                root[i] + Rroot[i] @ low)

    ref=np.zeros((T,3,3))
    for i in range(T): ref[i]=geom(i,q_ref[i,ids])
    lim=kin.leg_limits(a.leg); prev=np.zeros(3)
    for i in range(T):
        if w[i] == 0:
            prev=np.zeros(3); continue
        qr=q_ref[i,ids]
        dz=w[i]*(a.target_z-ref[i,2,2])
        ankle_target=ref[i,0]+np.array([0.0,0.0,dz])
        low_target=ref[i,2,2]+dz
        def residual(ql):
            ankle,patch,low=geom(i,ql)
            return np.r_[70*(ankle-ankle_target),
                         180*(patch[:2]-ref[i,1,:2]),
                         240*(low[2]-low_target),
                         0.25*(ql-qr), 0.12*((ql-qr)-prev)]
        sol=least_squares(residual,np.clip(qr+prev,lim[:,0],lim[:,1]),
                          bounds=(lim[:,0],lim[:,1]),max_nfev=55,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q[i,ids]=sol.x; prev=sol.x-qr

    fps=float(m["fps"]); dt=1/fps
    out["dof_positions"]=q.astype(np.float32)
    out["dof_velocities"]=np.gradient(q,dt,axis=0).astype(np.float32)
    out["stage4_added_support"]=np.array(
        [a.leg,str(s),str(e),str(ramp),str(a.target_z)])
    np.savez(a.out,**out)

    low=np.array([geom(i,q[i,ids])[2] for i in range(T)])
    dq=np.abs(q[:,ids]-q_ref[:,ids])
    print(f"[[ {a.leg} support window {s}-{e}, ramp {ramp}, target z "
          f"{a.target_z*1000:+.1f} mm")
    print(f"[[ joint correction mean/max {np.degrees(dq.mean()):.2f}/"
          f"{np.degrees(dq.max()):.2f} deg | active paw z min/max "
          f"{low[w>0,2].min()*1000:+.2f}/{low[w>0,2].max()*1000:+.2f} mm")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
