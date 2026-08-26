"""Which LINK is actually lowest / touching, per frame - not just the paws."""
import sys, numpy as np
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage2"); sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, axis_rot, quat_to_mat
from contact_model import ContactModel
U="/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"
kin=V4Kin(U); cm=ContactModel()
ch={}
for n,j in kin.j.items(): ch.setdefault(j["parent"],[]).append(n)
m=np.load(sys.argv[1],allow_pickle=True)
q=m["dof_positions"].astype(float); nm=[str(x) for x in m["dof_names"]]
rp=(m["root_pos"] if "root_pos" in m.files else m["body_positions"][:,0]).astype(float)
rq=(m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:,0]).astype(float)
fr0,fr1=(int(sys.argv[2]),int(sys.argv[3])) if len(sys.argv)>3 else (0,len(q))
print(f"{sys.argv[1].split('/')[-1]}  frames {fr0}-{fr1}")
print(" f  rootz tilt |  lowest link           z(mm) | links within 5mm of floor")
for i in range(fr0,fr1):
    d={n:q[i,nm.index(n)] for n in nm}
    fr={"origin":(quat_to_mat(rq[i]),rp[i].copy())}; st=["origin"]
    while st:
        par=st.pop(); Rp,pp=fr[par]
        for jn in ch.get(par,[]):
            J=kin.j[jn]; pj=pp+Rp@J["xyz"]
            fr[J["child"]]=(Rp@J["R"]@axis_rot(J["axis"],d.get(jn,0.0)),pj); st.append(J["child"])
    zs={}
    for ln,(R,p) in fr.items():
        if ln in cm.hull: zs[ln]=float((cm.hull[ln]@R.T+p)[:,2].min())
    lo=min(zs,key=zs.get); touch=[k for k,v in sorted(zs.items(),key=lambda x:x[1]) if v<0.005]
    tl=np.degrees(np.arccos(np.clip(quat_to_mat(rq[i])[2,2],-1,1)))
    print(" %2d %6.3f %5.1f | %-20s %7.1f | %s"%(i,rp[i,2],tl,lo,zs[lo]*1000,", ".join(touch) if touch else "NOTHING"))
