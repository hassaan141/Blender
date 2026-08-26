"""Compare ground-contact coherence: Ashley source vs v4 retarget, per clip."""
import sys, numpy as np
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage2"); sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel
U="/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"
kin=V4Kin(U); cm=ContactModel()
ALEGS=["aFL","aFR","aBL","aBR"]
name=sys.argv[1]; src=sys.argv[2]; mot=sys.argv[3]
d=np.load(src,allow_pickle=True)
toe={l:d[f"toe_{l}"].astype(float) for l in ALEGS}
leglen=float(d["rest_lengths"].sum(1).mean()); scale=0.2036/leglen
Z=np.stack([toe[l][:,2] for l in ALEGS],1)
g=np.percentile(Z,3.0); band=0.18*leglen
lowS=(Z-g)<band; spS=(Z.max(1)-Z.min(1))*scale
m=np.load(mot,allow_pickle=True)
q=m["dof_positions"].astype(float); nm=[str(x) for x in m["dof_names"]]
rp=(m["root_pos"] if "root_pos" in m.files else m["body_positions"][:,0]).astype(float)
rq=(m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:,0]).astype(float)
T=len(q); PZ=[]
for i in range(T):
    R0=quat_to_mat(rq[i]); row=[]
    for leg in LEGS:
        R,p=R0,rp[i].copy()
        for j,qi in zip(kin.leg_chain(leg),[q[i,nm.index(f"{leg}_{s}")] for s in("SY_J","SP_J","knee")]):
            J=kin.j[j]; p=p+R@J["xyz"]; R=R@J["R"]@axis_rot(J["axis"],qi)
        row.append((cm.hull[f"{leg}_knee"]@R.T+p)[:,2].min())
    PZ.append(row)
PZ=np.array(PZ); spR=PZ.max(1)-PZ.min(1); lowR=PZ<0.005
best=[]
for i in range(T):
    z=np.sort(PZ[i]); best.append(int((z-z[0]<0.005).sum()))
best=np.array(best)
print(f"=== {name} ===")
print(f" SOURCE  paws in contact band: mean {lowS.sum(1).mean():.2f}/4 | 4-paw frames {int((lowS.sum(1)==4).sum())}/{len(Z)}"
      f" | spread mean {spS.mean()*1000:.1f} mm")
print(f" RETARGET paws on floor      : mean {lowR.sum(1).mean():.2f}/4 | 4-paw frames {int((lowR.sum(1)==4).sum())}/{T}"
      f" | spread mean {spR.mean()*1000:.1f} mm")
print(f" best-possible support at ANY root height: "+" ".join(f"{k}paw:{int((best==k).sum())}" for k in range(1,5)))
print(f" retarget min paw z {PZ.min()*1000:+.1f} mm | spread ratio retarget/source {spR.mean()/(spS.mean()*scale):.2f}x")
