"""Solve a valid symmetric standing pose from the REAL collision hulls.
Each paw's lowest hull point is placed directly under its own SY pivot at stance
height h, with joints well inside their limits. No kinematics are changed."""
import numpy as np, sys
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage2")
sys.path.insert(0,"/home/hassaan/Bingo/Blender/stage4")
from v4_kinematics import V4Kin, LEGS, axis_rot
from contact_model import ContactModel
from scipy.optimize import least_squares
U="/home/hassaan/Bingo/Blender/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints_physics.urdf"
kin=V4Kin(U); cm=ContactModel()
def knee_frame(leg,q):
    R,p=np.eye(3),np.zeros(3)
    for name,qi in zip(kin.leg_chain(leg),q):
        J=kin.j[name]; p=p+R@J["xyz"]; R=R@J["R"]@axis_rot(J["axis"],qi)
    return R,p
def paw_low(leg,q):
    R,p=knee_frame(leg,q); w=cm.hull[f"{leg}_knee"]@R.T+p
    return w[np.argmin(w[:,2])]
H=float(sys.argv[1]) if len(sys.argv)>1 else 0.14
print(f"target stance height {H:.3f} m")
res={}
for leg in LEGS:
    hip=kin.sy_pos(leg); lim=kin.leg_limits(leg)
    # SY is held at 0: the SP pivot sits ~46 mm outboard of the SY pivot, so
    # demanding the paw sit under SY forces a permanent adducted (knock-kneed)
    # stance that needs continuous SY torque and topples the robot sideways.
    # A neutral-SY stance is what the leg geometry is actually designed for.
    def f(q):
        P=paw_low(leg,[0.0,q[0],q[1]])
        return [ (P[0]-hip[0])*2.0, (P[2]+H)*4.0 ]
    best=None
    for seed in ([-0.5,0.9],[0.5,-0.9],[-0.3,0.6],[0.3,-0.6],[0.85,0.94],[-0.85,-0.94]):
        r=least_squares(f,np.array(seed,float),bounds=(lim[1:,0]*0.9,lim[1:,1]*0.9),
                        xtol=1e-12,ftol=1e-12)
        c=np.linalg.norm(f(r.x))
        if best is None or c<best[0]: best=(c,r.x)
    q=np.array([0.0,best[1][0],best[1][1]]); P=paw_low(leg,q); res[leg]=q
    marg=np.minimum(q-lim[:,0],lim[:,1]-q)
    print(f"  {leg}: q={np.round(q,4)}  paw={np.round(P,4)}  resid={best[0]:.5f}  "
          f"limit-margin={np.round(marg,3)}")
print("\nSTAND = {")
for leg in LEGS:
    q=res[leg]
    print(f'    "{leg}_SY_J": {q[0]:+.4f}, "{leg}_SP_J": {q[1]:+.4f}, "{leg}_knee": {q[2]:+.4f},')
print("}")
lows=[paw_low(l,res[l])[2] for l in LEGS]
print(f"paw z: {np.round(lows,4)}  spread {1000*(max(lows)-min(lows)):.2f} mm  -> spawn {-min(lows)+0.002:.4f}")
