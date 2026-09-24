import {DEFAULT_POSE, JOINT_NAMES, RESIDUAL_SCALE} from "../constants.js";

const legs=["fl","fr","bl","br"], legCount=12;
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const qapply=(q,v)=>{const [w,x,y,z]=q, t=[2*(y*v[2]-z*v[1]),2*(z*v[0]-x*v[2]),2*(x*v[1]-y*v[0])];return [v[0]+w*t[0]+y*t[2]-z*t[1],v[1]+w*t[1]+z*t[0]-x*t[2],v[2]+w*t[2]+x*t[1]-y*t[0]];};
const qmul=(a,b)=>[a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
const lerp=(a,b,t)=>a+(b-a)*t;
// Match float32 Torch operations, including MotionLoader's nearest-frame
// interpolation (which deliberately permits negative blend coefficients).
const f = Math.fround;
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const mix = (a,b,w) => f(f(a*f(1-w))+f(b*w));
const roundEven = x => { const n=Math.floor(x), d=x-n; return d===.5 ? n+(n%2) : Math.round(x); };
export const duration = refs => (refs.f.dof_positions.length-1)/refs.f.fps;
export async function loadCommandReferences(){
  const u=async p=>{const r=await fetch(`./policies/${p}.json`);if(!r.ok)throw Error(`reference ${p}: ${r.status}`);return r.json();};
  const [f,b,t,p,bt,turnBank]=await Promise.all(['reference_forward','reference_backward','reference_turning','pivot_bank','backward_turn_bank','turn_bank'].map(u));
  return {f,b,t,p,bt,turnBank};
}
export function drive(cmd){
  const [vx,yaw]=cmd.map(f), turn=clamp(f(Math.abs(yaw)/f(.4)),0,1);
  const positive=f(vx*f(1.25+f(f(.215/.15-1.25)*turn)));
  const negative=-Math.min(f(Math.abs(vx)*f(1+f(turn/3))),f(.2));
  const pivot=f(clamp(f(1-f(Math.abs(vx)/f(.1))),0,1)*turn);
  const d=f(f((vx<0?negative:positive)*f(1-pivot))+f(f(vx<-.02?-.2:.2)*pivot));
  return {d,turn,pivot};
}
export function nextTime(refs,cmd,clipTime){
  const increment=f(f(f((1/24)*1.4)*drive(cmd).d)/f(.66));
  const sum=f(clipTime+increment), dur=f(duration(refs));
  return f(((sum % dur)+dur)%dur);
}
function referenceLegs(ref,time){
  const n=ref.dof_positions.length-1, dt=1/ref.fps, dur=dt*n;
  // Isaac's CUDA float32 path lands a few ulps either side of half-frame
  // boundaries.  MotionLoader rounds the float32 phase to nearest frame;
  // this tiny guard keeps the JS index on the same side of that boundary.
  const i=clamp(roundEven(f(f(time/dur)*n)+1e-5),0,n), j=Math.min(i+1,n);
  const w=f(roundEven(((time-i*dt)/dt)*1e5)/1e5);
  return JOINT_NAMES.slice(0,12).map(name=>{const k=ref.dof_names.indexOf(name);if(k<0)throw Error(`reference missing ${name}`);return mix(ref.dof_positions[i][k],ref.dof_positions[j][k],w);});
}
function bankLegs(bank,time,yaw){
  const t=f(time*bank.fps), lo=clamp(Math.trunc(t),0,bank.qdelta[0].length-2), w=f(t-lo);
  const y=clamp(f(f(yaw+f(.6))/f(.2)),0,6), yl=clamp(Math.trunc(y),0,5), yw=f(y-yl);
  return Array.from({length:12},(_,k)=>mix(mix(bank.qdelta[yl][lo][k],bank.qdelta[yl][lo+1][k],w),mix(bank.qdelta[yl+1][lo][k],bank.qdelta[yl+1][lo+1][k],w),yw));
}
export function feedforwardAtTime(refs,cmd,clipTime){
  const {d,turn,pivot}=drive(cmd), time=nextTime(refs,cmd,clipTime);
  const back=clamp(f(-cmd[0]/f(.05)),0,1), wt=f(turn*f(1-back));
  const fq=referenceLegs(refs.f,time), bq=referenceLegs(refs.b,time), tq=referenceLegs(refs.t,time);
  const yaw=f(cmd[1]*(d<0?-.5:1));
  const delta=bankLegs(refs.turnBank,time,yaw), bd=bankLegs(refs.bt,time,yaw), pd=bankLegs(refs.p,time,yaw);
  const w=clamp(f(Math.abs(d)/f(.198)),0,1);
  return fq.map((v,k)=>mix(DEFAULT_POSE[k],f(mix(mix(v,bq[k],back),tq[k],wt)+mix(mix(delta[k],bd[k],back),pd[k],pivot)),w));
}
export function nextFeedforward(refs,cmd,phase){return feedforwardAtTime(refs,cmd,f(phase*duration(refs)));}
// This state machine is shared by the browser control step and Node verifier.
// Observation is read BEFORE command slew; targets use the slewed command and
// updated filtered residual. The clip clock advances only after target sampling.
export class CommandState {
  constructor(refs,{phase=0,command=[0,0],filtered=Array(12).fill(0)}={}){
    this.refs=refs;this.clipTime=f(phase*duration(refs));this.command=command.map(f);this.filtered=Float32Array.from(filtered);
  }
  get phase(){return f(this.clipTime/f(duration(this.refs)));}
  feedforward(){return feedforwardAtTime(this.refs,this.command,this.clipTime);}
  observation(proprio){
    const o=new Float32Array(95);o.set(proprio,0);
    const angle=f(f(this.phase*2)*f(Math.PI));
    o[67]=Math.sin(angle);o[68]=Math.cos(angle);
    o[69]=f(this.command[0]/f(.3));o[70]=f(this.command[1]/f(.6));
    o.set(this.feedforward(),71);o.set(this.filtered,83);return o;
  }
  advance(action,target){
    this.command=this.command.map((v,i)=>f(v+f(.1)*f(f(target[i])-v)));
    const activity=clamp(Math.max(f(Math.abs(this.command[0])/f(.15)),f(Math.abs(this.command[1])/f(.4))),0,1);
    this.filtered=Float32Array.from(action,(v,i)=>f(f(f(f(.3)*v)+f(f(.7)*this.filtered[i]))*activity));
    const ff=this.feedforward(), qTarget=ff.map((v,i)=>f(v+f(f(.3)*this.filtered[i])));
    this.clipTime=nextTime(this.refs,this.command,this.clipTime);
    return {feedforward:ff,qTarget};
  }
}
export function buildCommandObs(sim,cmd,phase,ff,filtered){const o=new Float32Array(95);let i=0,q=sim.baseQuat(),yaw=Math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3])),hq=[Math.cos(yaw/2),0,0,-Math.sin(yaw/2)],qh=qmul(hq,q),root=sim.bodyPos('origin'),d=sim.data;for(let j=0;j<21;j++)o[i++]=d.qpos[sim.qposAdr[JOINT_NAMES[j]]];for(let j=0;j<21;j++)o[i++]=d.qvel[sim.dofAdr[JOINT_NAMES[j]]];o[i++]=root[2];for(const v of [qapply(qh,[1,0,0]),qapply(qh,[0,0,1])])for(const x of v)o[i++]=x;for(const v of [qapply(hq,[d.qvel[0],d.qvel[1],d.qvel[2]]),qapply(hq,[d.qvel[3],d.qvel[4],d.qvel[5]])])for(const x of v)o[i++]=x;for(const leg of legs){const tip=add(sim.bodyPos(`${leg}_knee`),qapply(sim.bodyQuat(`${leg}_knee`),[0,0,-.12]));for(const x of qapply(hq,sub(tip,root)))o[i++]=x;}o[i++]=Math.sin(phase*2*Math.PI);o[i++]=Math.cos(phase*2*Math.PI);o[i++]=cmd[0]/.3;o[i++]=cmd[1]/.6;for(const x of ff.slice(0,12))o[i++]=x;for(const x of filtered)o[i++]=x;if(i!==95)throw Error(`command observation ${i}`);return o;}
export {RESIDUAL_SCALE};
