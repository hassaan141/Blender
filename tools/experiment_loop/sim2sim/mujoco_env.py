"""Python MuJoCo wrapper for BASELINE_1's verified 95 -> 12 runtime contract.

The MJCF and actuator/mesh assets are read from the browser simulator.  This is
an adapter for training/evaluation, not a second robot model.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
MJCF = ROOT / "bingo-simulator/app/public/robot/bingo_scene.xml"
BASE = ROOT / "BASELINE_1"
REFDIR = BASE / "references"
# Browser runtime feeds the policy these values for the 9 expressive joints (obs 12-20, 33-41);
# read from the app so there is one source. See EXPR_OBS_TRAINING_MEAN in constants.js.
import re as _re
EXPR_OBS_TRAINING_MEAN = np.array([float(x) for x in _re.search(
    r"EXPR_OBS_TRAINING_MEAN = new Float32Array\(\[(.*?)\]\)",
    (ROOT / "bingo-simulator/app/src/game/constants.js").read_text(), _re.S).group(1).replace("\n", "").split(",") if x.strip()],
    np.float32)
assert EXPR_OBS_TRAINING_MEAN.shape == (18,)
JOINTS = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
          "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
          "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
          "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
LEGS = JOINTS[:12]
PAWS = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]
STAND = np.array([0., .8100, .8932, 0., -.8109, -.8938, 0., .3932, .8913,
                  0., .3936, -.8913], np.float64)
CASES = [("stand", 0., 0.), ("forward", .2, 0.), ("backward", -.2, 0.),
         ("left", 0., .4), ("right", 0., -.4), ("forward_left", .15, .4),
         ("forward_right", .15, -.4), ("backward_left", -.15, .4),
         ("backward_right", -.15, -.4)]
DT = 1. / 24.
RESIDUAL_SCALE = .3
EMA_ALPHA = .3
CMD_ALPHA = .1
# Browser controllers/expression.js, Neutral personality, zero operator look.
EXPR_LIMITS = np.array([[-.65, .05], [-.60, .60], [-.78, .78], [-.60, .60], [-.60, .60],
                        [-3., 3.], [-1.5, 0.], [-3., 3.], [0., 1.5]])
NEUTRAL = {'head': (.05, .25, 0.), 'tail': (.10, .5, 0.), 'ear': (.05, .3, -.2)}  # amp, hz, bias


def neutral_expression(t):
    """ExpressionController.update() targets at controller time t (EXPR joint order)."""
    def osc(c, ph=0.):
        amp, hz, bias = NEUTRAL[c]; return bias + amp*math.sin(2*math.pi*hz*t + ph)
    v = np.array([osc('head'), osc('head', 1.1)*.6, osc('head', 2.2)*.5,
                  osc('tail'), osc('tail', 1.6),
                  osc('ear', .4), -abs(osc('ear', .9)), osc('ear', .4), abs(osc('ear', .9))])
    return np.clip(v, EXPR_LIMITS[:, 0], EXPR_LIMITS[:, 1]).astype(np.float32).astype(np.float64)


def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def qapply(q, v):
    qv=q[1:]; t=2*np.cross(qv, v)
    return v + q[0]*t + np.cross(qv,t)


class Ref:
    def __init__(self, path):
        z=np.load(path)
        self.fps=float(z['fps']); self.n=len(z['dof_positions']); self.duration=(self.n-1)/self.fps
        self.dof_names=list(z['dof_names']); self.body_names=list(z['body_names'])
        for k in ('dof_positions','dof_velocities','body_positions','body_rotations',
                  'body_linear_velocities','body_angular_velocities','contacts','ear_positions'):
            if k in z: setattr(self,k,z[k])
        self.expr=np.concatenate([self.dof_positions[:,self.dof_names.index(n):self.dof_names.index(n)+1]
                                  for n in JOINTS[12:]],axis=1)
    def sample(self, t):
        t=float(np.clip(t,0.,self.duration)); frame0=int(np.rint((t/self.duration)*(self.n-1)))
        frame1=min(frame0+1,self.n-1); u=np.round((t-frame0/self.fps)*self.fps,5)
        out={}
        # MotionLoader uses rounded-frame indexing and SLERP, including its
        # slightly negative blend fraction for samples in the lower half-frame.
        for k in ('dof_positions','dof_velocities','body_positions','body_linear_velocities','body_angular_velocities'):
            arr=getattr(self,k); out[k]=arr[frame0]*(1-u)+arr[frame1]*u
        a=self.body_rotations[frame0].copy(); b=self.body_rotations[frame1].copy()
        dot=np.sum(a*b,axis=-1,keepdims=True); b=np.where(dot<0,-b,b); dot=np.abs(dot).clip(0,1)
        theta=np.arccos(dot); sin=np.sqrt(np.maximum(1-dot*dot,0));
        ra=np.sin((1-u)*theta)/np.maximum(sin,1e-8); rb=np.sin(u*theta)/np.maximum(sin,1e-8)
        q=ra*a+rb*b; near=(sin<.001)|(dot>=1); q=np.where(near,(a+b)*.5,q); q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
        out['body_rotations']=q; out['contacts']=self.contacts[frame0]
        return out


class BingoMujoco:
    def __init__(self, seed=0, expression='neutral', expr_obs='training_mean'):
        self.expr_obs=expr_obs  # 'training_mean' = current browser runtime; 'live' = raw joint values
        if expression not in ('neutral','reference'): raise ValueError(f'unknown expression mode {expression}')
        self.expression=expression  # 'neutral' = browser runtime; 'reference' = attempts 01-03
        self.rng=np.random.default_rng(seed)
        self.model=mujoco.MjModel.from_xml_path(str(MJCF)); self.data=mujoco.MjData(self.model)
        if abs(self.model.opt.timestep-1/120)>1e-7: raise ValueError('MJCF physics timestep is not 120 Hz')
        self.jid={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINTS}
        self.qa={n:int(self.model.jnt_qposadr[j]) for n,j in self.jid.items()}
        self.da={n:int(self.model.jnt_dofadr[j]) for n,j in self.jid.items()}
        self.aid={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in JOINTS}
        self.bid={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['origin']+PAWS}
        if min(*self.jid.values(),*self.aid.values(),*self.bid.values())<0: raise ValueError('MJCF name lookup failed')
        self.refs={n:Ref(REFDIR/n/'reference.npz') for n in ('forward','backward','turning')}
        self.banks={n:np.load(REFDIR/f)['qdelta'] for n,f in [('turning','turning/turn_bank.npz'),('pivot','pivot_bank.npz'),('backward','backward_turn_bank.npz')]}
        self.norm=None; self.reset((0.,0.))

    def reset(self, command=(0.,0.)):
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        self.phase=0.; self.filtered=np.zeros(12); self.command=np.zeros(2); self.target=np.asarray(command,dtype=np.float64)
        self.prev_qvel=np.zeros(12); self.prev_rate=np.zeros(12); self.elapsed=0.; self.fallen=False; self.expr_t=0.
        mujoco.mj_forward(self.model,self.data)
        return self.observe()

    def _drive(self):
        vx,yaw=self.command; turn=np.clip(abs(yaw)/.4,0,1)
        positive=vx*(1.25+(.215/.15-1.25)*turn)
        negative=-min(abs(vx)*(1+turn/3),.20)
        drive=negative if vx<0 else positive
        pivot=np.clip(1-abs(vx)/.1,0,1)*turn
        drive=drive*(1-pivot)+(-.20 if vx<-.02 else .20)*pivot
        return float(drive),float(turn),float(pivot)

    def _ref(self,t):
        fwd=self.refs['forward'].sample(t); back=self.refs['backward'].sample(t); turnref=self.refs['turning'].sample(t)
        drive,turn,pivot=self._drive(); b=np.clip(-self.command[0]/.05,0,1)
        result={}
        for k in ('dof_positions','dof_velocities','body_positions','body_rotations','body_linear_velocities','body_angular_velocities'):
            x=fwd[k]*(1-b)+back[k]*b
            x=x*(1-turn*(1-b))+turnref[k]*turn*(1-b)
            result[k]=x
        # Match CommandEnv's yaw-indexed bank blend and its backward/pivot banks.
        signed_yaw=self.command[1]*(-.5 if drive<0 else 1.)
        y=np.clip((signed_yaw+.6)/.2,0,6); yl=min(int(y),5); yh=yl+1; yw=y-yl
        fr=(t % self.refs['forward'].duration)*self.refs['forward'].fps
        lo=min(int(fr),self.refs['forward'].n-2); tw=fr-lo
        def bank_at(bank):
            return (bank[yl,lo]*(1-tw)+bank[yl,lo+1]*tw)*(1-yw)+(bank[yh,lo]*(1-tw)+bank[yh,lo+1]*tw)*yw
        delta=bank_at(self.banks['turning'])*(1-b)+bank_at(self.banks['backward'])*b
        delta=delta*(1-pivot)+bank_at(self.banks['pivot'])*pivot
        result['dof_positions'][:12]+=delta
        return result

    def _ff(self):
        drive,_,_=self._drive(); t=(self.phase+DT*1.4*drive/.66) % self.refs['forward'].duration
        q=self._ref(t)['dof_positions'][:12]
        w=np.clip(abs(drive)/.198,0,1)
        return w*q+(1-w)*STAND

    def _proprio(self):
        # Browser buildCommandObs reads root from xpos('origin'), which after mj_step
        # is one substep behind qpos - the same staleness as the paw xpos below.
        d=self.data; q=d.qpos[3:7].copy(); root=d.xpos[self.bid['origin']].copy()
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
        qi=np.array([math.cos(yaw/2),0,0,-math.sin(yaw/2)])
        qh=qmul(qi,q)
        dof=np.array([d.qpos[self.qa[n]] for n in JOINTS]); vel=np.array([d.qvel[self.da[n]] for n in JOINTS])
        feet=[]
        for n in PAWS:
            bid=self.bid[n]; p=d.xpos[bid].copy(); fq=d.xquat[bid].copy()
            tip=p+qapply(fq,np.array([0.,0.,-.12])); feet.extend(qapply(qi,tip-root))
        tangent=qapply(qh,np.array([1.,0.,0.])); normal=qapply(qh,np.array([0.,0.,1.]))
        lin=qapply(qi,d.qvel[:3].copy()); ang=qapply(qi,d.qvel[3:6].copy())
        return np.concatenate([dof,vel,[root[2]],tangent,normal,lin,ang,np.asarray(feet)])

    def observe(self):
        phase=2*math.pi*self.phase/self.refs['forward'].duration
        obs=np.concatenate([self._proprio(),[math.sin(phase),math.cos(phase)],self.command/[.3,.6],self._ff(),self.filtered]).astype(np.float32)
        if self.expr_obs=='training_mean': obs[12:21]=EXPR_OBS_TRAINING_MEAN[:9]; obs[33:42]=EXPR_OBS_TRAINING_MEAN[9:]
        return obs

    def step(self, raw_action):
        raw=np.clip(np.asarray(raw_action,dtype=np.float64),-1,1)
        self.command += CMD_ALPHA*(self.target-self.command)
        prevf=self.filtered.copy(); self.filtered=EMA_ALPHA*raw+(1-EMA_ALPHA)*prevf
        activity=max(np.clip(abs(self.command[0])/.15,0,1),np.clip(abs(self.command[1])/.4,0,1))
        self.filtered*=activity
        drive,_,_=self._drive(); ff=self._ff(); phase_next=(self.phase+DT*1.4*drive/.66)%self.refs['forward'].duration
        # Browser: ExpressionController advances its clock by one control step, then
        # evaluates. Legacy mode replays the forward reference at the next phase.
        self.expr_t+=DT
        if self.expression=='neutral': expr=neutral_expression(self.expr_t)
        else:
            expr_ref=self.refs['forward']; frame=int(np.clip(np.rint((phase_next % expr_ref.duration)*expr_ref.fps),0,expr_ref.n-1))
            expr=expr_ref.expr[frame]
        target=np.concatenate([ff+RESIDUAL_SCALE*self.filtered,expr])
        for i,n in enumerate(JOINTS): self.data.ctrl[self.aid[n]]=target[i]
        for _ in range(5): mujoco.mj_step(self.model,self.data)
        self.phase=phase_next; self.elapsed+=DT
        obs=self.observe(); lin=self.data.qvel[:3].copy(); ang=self.data.qvel[3:6].copy(); q=self.data.qpos[3:7]
        vx=qapply(np.array([math.cos(math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))/2),0,0,-math.sin(math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))/2)]),lin)[0]
        yawrate=ang[2]; height=self.data.qpos[2]
        upright=max(0.,1-2*(q[1]**2+q[2]**2))
        torque=np.array([abs(self.data.actuator_force[self.aid[n]]) for n in LEGS])
        qvel=np.array([self.data.qvel[self.da[n]] for n in LEGS]); accel=(qvel-self.prev_qvel)/DT; self.prev_qvel=qvel
        rate=(self.filtered-prevf)/DT
        contacts=set()
        for ci in range(self.data.ncon):
            c=self.data.contact[ci]
            for gi in (c.geom1,c.geom2):
                bi=int(self.model.geom_bodyid[gi])
                for leg in PAWS:
                    if bi==self.bid[leg]: contacts.add(leg)
        slip=[]
        for leg in contacts:
            bid=self.bid[leg]; vel=self.data.cvel[bid]
            slip.append(float(np.linalg.norm(vel[3:5])))
        reward=(1.0*math.exp(-((vx-self.command[0])/.18)**2)+.45*math.exp(-((yawrate-self.command[1])/.35)**2)
                +.25*upright-.20*((height-.18)/.035)**2-.018*np.mean((torque/3.)**2)
                -.012*np.mean(rate**2)-.003*np.mean(self.filtered**2)-.08*np.mean((accel/50.)**2)
                -.05*(float(np.mean(slip)) if slip else 0.))
        fallen=height<.105 or upright<.35 or not np.isfinite(obs).all()
        self.fallen |= fallen
        info={'vx':float(vx),'yaw_rate':float(yawrate),'height':float(height),'upright':float(upright),
              'torque':torque,'accel':accel,'action_rate':rate,'slip':slip,'fallen':fallen,
              'distance':float(self.data.qpos[0]),'joint_pos':np.array([self.data.qpos[self.qa[n]] for n in LEGS])}
        return obs,float(reward),fallen,info
