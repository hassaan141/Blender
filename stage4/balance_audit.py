"""CoM vs support-polygon margin per frame, using true collision hulls."""
import sys, numpy as np, xml.etree.ElementTree as ET
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage2"); sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel
U="/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"
kin=V4Kin(U); cm=ContactModel()
r=ET.parse(U).getroot(); LM={}
for ln in r.findall("link"):
    i=ln.find("inertial"); o=i.find("origin")
    LM[ln.get("name")]=(float(i.find("mass").get("value")),
                        np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]))
ch={}
for n,j in kin.j.items(): ch.setdefault(j["parent"],[]).append(n)
m=np.load(sys.argv[1],allow_pickle=True)
q=m["dof_positions"].astype(float); nm=[str(x) for x in m["dof_names"]]
rp=(m["root_pos"] if "root_pos" in m.files else m["body_positions"][:,0]).astype(float)
rq=(m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:,0]).astype(float); T=len(q)
out=[]
for i in range(T):
    d={n:q[i,nm.index(n)] for n in nm}
    fr={"origin":(quat_to_mat(rq[i]),rp[i].copy())}; st=["origin"]
    while st:
        par=st.pop(); Rp,pp=fr[par]
        for jn in ch.get(par,[]):
            J=kin.j[jn]; pj=pp+Rp@J["xyz"]
            fr[J["child"]]=(Rp@J["R"]@axis_rot(J["axis"],d.get(jn,0.0)),pj); st.append(J["child"])
    M=0.0; C=np.zeros(3)
    for ln,(R,p) in fr.items():
        mm,cc=LM[ln]; M+=mm; C+=mm*(p+R@cc)
    C/=M
    pz=[];pxy=[]
    for leg in LEGS:
        R,p=fr[f"{leg}_knee"]; w=cm.hull[f"{leg}_knee"]@R.T+p
        k=int(np.argmin(w[:,2])); pz.append(w[k,2]); pxy.append(w[k,:2])
    pz=np.array(pz);pxy=np.array(pxy);sup=pz<0.005;n=int(sup.sum())
    if n>=3:
        P=pxy[sup];c=P.mean(0);P=P[np.argsort(np.arctan2(P[:,1]-c[1],P[:,0]-c[0]))]
        ins=True;dmin=1e9
        for k in range(len(P)):
            aa,bb=P[k],P[(k+1)%len(P)];e=bb-aa;nr=np.array([-e[1],e[0]]);nr/=np.linalg.norm(nr)+1e-12
            dd=float(np.dot(C[:2]-aa,nr))
            if dd<0: ins=False
            dmin=min(dmin,abs(dd))
        mg=dmin if ins else -dmin
    elif n==2:
        aa,bb=pxy[sup];e=bb-aa;L=np.linalg.norm(e)+1e-12
        mg=-abs(float(e[0]/L*(C[1]-aa[1])-e[1]/L*(C[0]-aa[0])))
    elif n==1: mg=-float(np.linalg.norm(C[:2]-pxy[sup][0]))
    else: mg=-9.99
    out.append((n,mg))
out=np.array(out); n=out[:,0]; mg=out[:,1]
print(f"{sys.argv[1].split('/')[-1]}: statically stable {int((mg>0).sum())}/{T} = {100*(mg>0).mean():.0f}%"
      f" | supports "+" ".join(f"{k}:{int((n==k).sum())}" for k in range(5))
      +f" | median margin {np.median(mg)*1000:+.1f} mm")
