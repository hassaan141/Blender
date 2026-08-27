"""CoM vs support-polygon margin per frame, using true collision hulls."""
import argparse, sys, numpy as np, xml.etree.ElementTree as ET
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
ap=argparse.ArgumentParser()
ap.add_argument("motion")
ap.add_argument("--frames", help="optional inclusive detail range, e.g. 12-36")
a=ap.parse_args()
m=np.load(a.motion,allow_pickle=True)
q=m["dof_positions"].astype(float); nm=[str(x) for x in m["dof_names"]]
rp=(m["root_pos"] if "root_pos" in m.files else m["body_positions"][:,0]).astype(float)
rq=(m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:,0]).astype(float); T=len(q)
out=[]; COM=[]; PXY=[]; SUP=[]; PZ=[]
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
    out.append((n,mg)); COM.append(C); PXY.append(pxy); SUP.append(sup); PZ.append(pz)
out=np.array(out); n=out[:,0]; mg=out[:,1]
print(f"{a.motion.split('/')[-1]}: statically stable {int((mg>0).sum())}/{T} = {100*(mg>0).mean():.0f}%"
      f" | supports "+" ".join(f"{k}:{int((n==k).sum())}" for k in range(5))
      +f" | median margin {np.median(mg)*1000:+.1f} mm")


# ---- dynamic feasibility: ZMP (cart-table) + required friction ratio ---------
COM=np.array(COM); PXY=np.array(PXY); SUP=np.array(SUP); PZ=np.array(PZ)
fps=float(m["fps"]); dt=1.0/fps
V=np.gradient(COM,dt,axis=0); A=np.gradient(V,dt,axis=0)
g=9.81; az=A[:,2]+g
mu_req=np.linalg.norm(A[:,:2],axis=1)/np.maximum(az,1e-6)
zmp=COM[:,:2]-COM[:,2:3]*A[:,:2]/az[:,None]
def margin(P,c):
    if len(P)>=3:
        cc=P.mean(0); P=P[np.argsort(np.arctan2(P[:,1]-cc[1],P[:,0]-cc[0]))]
        ins=True; dmin=1e9
        for k in range(len(P)):
            a,b=P[k],P[(k+1)%len(P)]; e=b-a; nr=np.array([-e[1],e[0]]); nr/=np.linalg.norm(nr)+1e-12
            d=float(np.dot(c-a,nr))
            if d<0: ins=False
            dmin=min(dmin,abs(d))
        return dmin if ins else -dmin
    if len(P)==2:
        a,b=P; e=b-a; L=np.linalg.norm(e)+1e-12
        return -abs(float(e[0]/L*(c[1]-a[1])-e[1]/L*(c[0]-a[0])))
    if len(P)==1: return -float(np.linalg.norm(c-P[0]))
    return -9.99
zm=np.array([margin(PXY[i][SUP[i]],zmp[i]) for i in range(T)])
print(f"  DYNAMIC: ZMP inside support {int((zm>0).sum())}/{T} = {100*(zm>0).mean():.0f}%"
      f" | median ZMP margin {np.median(zm)*1000:+.1f} mm | worst {zm.min()*1000:+.0f} mm")
print(f"  required mu (|a_xy|/(a_z+g)): mean {mu_req.mean():.2f} p90 {np.percentile(mu_req,90):.2f} max {mu_req.max():.2f}"
      f" | frames needing mu>1.0: {int((mu_req>1.0).sum())}")
print(f"  free-fall frames (a_z+g<0, i.e. base accelerating down faster than g): {int((az<0).sum())}")
bad=np.where(zm<0)[0]
if len(bad):
    import itertools
    runs=[]; 
    for k,grp in itertools.groupby(enumerate(bad),lambda t:t[1]-t[0]):
        gg=[x[1] for x in grp]; runs.append((gg[0],gg[-1]))
    print("  ZMP-infeasible windows:", ", ".join(f"{a}-{b}" for a,b in runs[:14]))
if a.frames:
    first,last=(int(x) for x in a.frames.split("-",1))
    print("  frame support mask paw_z_mm static_mm zmp_mm com_xy_mm zmp_xy_mm support_center_xy_mm")
    for i in range(max(0,first),min(T-1,last)+1):
        center=PXY[i][SUP[i]].mean(0) if SUP[i].any() else np.full(2,np.nan)
        mask="".join("1" if x else "0" for x in SUP[i])
        ztxt="("+",".join(f"{z*1000:+.1f}" for z in PZ[i])+")"
        print(f"  {i:4d} {int(n[i]):d} {mask} {ztxt:>25s} {mg[i]*1000:+8.1f} {zm[i]*1000:+8.1f} "
              f"({COM[i,0]*1000:+7.1f},{COM[i,1]*1000:+7.1f}) "
              f"({zmp[i,0]*1000:+7.1f},{zmp[i,1]*1000:+7.1f}) "
              f"({center[0]*1000:+7.1f},{center[1]*1000:+7.1f})")
