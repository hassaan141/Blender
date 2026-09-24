// Definitive live moving-obs check. Set the browser MuJoCo to a mid-gait Isaac state,
// then have the APP's own buildObs() construct the observation and compare its
// proprioception (indices 0..66) to the Isaac row. Standstill was already verified;
// this catches velocity/paw-frame bugs that only appear while moving.
import {chromium} from "playwright";
import fs from "node:fs";
const URL_ = process.argv[2] || "http://localhost:5173/";
const trace = JSON.parse(fs.readFileSync(process.argv[3]));
const CHROME = process.env.CHROME_PATH;
const rows = ["W","A"].flatMap(c => [50,100,150].map(s =>
  trace.rows.find(r => r.command_name===c && r.step===s)).filter(Boolean));

const browser = await chromium.launch({executablePath:CHROME||undefined,
  args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const page = await (await browser.newContext({viewport:{width:800,height:600}})).newPage();
await page.goto(URL_,{waitUntil:"networkidle",timeout:60000});
await page.waitForFunction(()=>window.__bingo?.runtime?.snapshot()?.hasPolicy,null,{timeout:60000});
await page.waitForTimeout(2000);

const res = await page.evaluate(({rows,names,paws})=>{
  const {runtime}=window.__bingo, sim=runtime.sim;
  const jid=n=>sim.qposAdr[n], did=n=>sim.dofAdr[n];
  const quatFromXZ=(x,z)=>{const nx=Math.hypot(...x),nz=Math.hypot(...z);x=x.map(v=>v/nx);z=z.map(v=>v/nz);
    const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
    const m00=x[0],m01=y[0],m02=z[0],m10=x[1],m11=y[1],m12=z[1],m20=x[2],m21=y[2],m22=z[2],t=m00+m11+m22;
    let q;if(t>0){let s=Math.sqrt(t+1)*2;q=[.25*s,(m21-m12)/s,(m02-m20)/s,(m10-m01)/s];}
    else if(m00>m11&&m00>m22){let s=Math.sqrt(1+m00-m11-m22)*2;q=[(m21-m12)/s,.25*s,(m01+m10)/s,(m02+m20)/s];}
    else if(m11>m22){let s=Math.sqrt(1+m11-m00-m22)*2;q=[(m02-m20)/s,(m01+m10)/s,.25*s,(m12+m21)/s];}
    else{let s=Math.sqrt(1+m22-m00-m11)*2;q=[(m10-m01)/s,(m02+m20)/s,(m12+m21)/s,.25*s];}return q;};
  const setState=o=>{const q=quatFromXZ(o.slice(43,46),o.slice(46,49));
    sim.data.qpos[0]=0;sim.data.qpos[1]=0;sim.data.qpos[2]=o[42];sim.data.qpos.set(q,3);
    for(let i=0;i<21;i++){sim.data.qpos[jid(names[i])]=o[i];sim.data.qvel[did(names[i])]=o[21+i];}
    sim.data.qvel.set(o.slice(49,55),0);sim.forward();};
  const segs=[["joint_pos",0,21],["joint_vel",21,42],["root_z",42,43],["tan_normal",43,49],
    ["lin_vel",49,52],["ang_vel",52,55],["paw_pos",55,67]];
  const out=[];
  for(const row of rows){
    setState(row.observation);
    runtime.cmd=[0,0]; runtime.phase=0; runtime.filteredResidual.fill(0);
    const obs=runtime.buildObs();
    const exp=row.observation, seg={};
    for(const [n,a,b] of segs){let m=0;for(let i=a;i<b;i++)m=Math.max(m,Math.abs(obs[i]-exp[i]));seg[n]=m;}
    out.push({cmd:row.command_name,step:row.step,seg});
  }
  return out;
},{rows,names:trace.obs_dof_indexes,paws:trace.paw_order});
await browser.close();

console.log("cmd/step  " + ["joint_pos","joint_vel","root_z","tan_normal","lin_vel","ang_vel","paw_pos"].map(s=>s.padStart(11)).join(""));
let worst=0;
for(const r of res){
  console.log(`${r.cmd}/${r.step}`.padEnd(10) + ["joint_pos","joint_vel","root_z","tan_normal","lin_vel","ang_vel","paw_pos"]
    .map(s=>r.seg[s].toExponential(2).padStart(11)).join(""));
  worst=Math.max(worst,...Object.values(r.seg));
}
console.log(`\nworst proprio error while moving: ${worst.toExponential(3)}  -> ${worst<1e-2?"LIVE OBS OK (dynamics gap is real)":"LIVE OBS BUG FOUND"}`);
