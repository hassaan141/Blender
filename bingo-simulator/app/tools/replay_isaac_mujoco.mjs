// Replay Isaac's recorded q_targets through browser MuJoCo and compare the
// resulting state with the next Isaac observation. Requires Vite preview and
// the same headless Chromium flags as check_rig_vs_physics.mjs.
import {chromium} from "playwright";
import fs from "node:fs";
import path from "node:path";

const URL_ = process.argv[2] || "http://localhost:4173/";
const tracePath = process.argv[3] || "../../scratch/isaac_trace.json";
const trace = JSON.parse(fs.readFileSync(tracePath));
const CHROME = process.env.CHROME_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const outPath = path.resolve("../../scratch/sim2sim_report.json");
const videoPath = path.resolve("../../scratch/sim2sim_divergence.png");
const browser = await chromium.launch({executablePath:CHROME,args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const page = await (await browser.newContext({viewport:{width:1000,height:700}})).newPage();
await page.goto(URL_,{waitUntil:"networkidle",timeout:60000});
await page.waitForFunction(()=>window.__bingo?.runtime?.sim,null,{timeout:60000});
const report=await page.evaluate(async ({trace})=>{
  const {runtime}=window.__bingo, sim=runtime.sim, O=sim.mujoco.mjtObj;
  const names=trace.obs_dof_indexes, paws=trace.paw_order;
  const jid=n=>sim.qposAdr[n], did=n=>sim.dofAdr[n], aid=n=>sim.actId[n];
  const body=n=>sim.mujoco.mj_name2id(sim.model,O.mjOBJ_BODY.value,n);
  const quatFromXZ=(x,z)=>{ // columns x/y/z of a yaw-zero root orientation
    const nx=Math.hypot(...x), nz=Math.hypot(...z);x=x.map(v=>v/nx);z=z.map(v=>v/nz);
    const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
    const m00=x[0],m01=y[0],m02=z[0],m10=x[1],m11=y[1],m12=z[1],m20=x[2],m21=y[2],m22=z[2],t=m00+m11+m22;
    let q;if(t>0){let s=Math.sqrt(t+1)*2;q=[.25*s,(m21-m12)/s,(m02-m20)/s,(m10-m01)/s];}
    else if(m00>m11&&m00>m22){let s=Math.sqrt(1+m00-m11-m22)*2;q=[(m21-m12)/s,.25*s,(m01+m10)/s,(m02+m20)/s];}
    else if(m11>m22){let s=Math.sqrt(1+m11-m00-m22)*2;q=[(m02-m20)/s,(m01+m10)/s,.25*s,(m12+m21)/s];}
    else {let s=Math.sqrt(1+m22-m00-m11)*2;q=[(m10-m01)/s,(m02+m20)/s,(m12+m21)/s,.25*s];}
    return q;
  };
  const setState=o=>{const q=quatFromXZ(o.slice(43,46),o.slice(46,49));sim.data.qpos[0]=0;sim.data.qpos[1]=0;sim.data.qpos[2]=o[42];sim.data.qpos.set(q,3);for(let i=0;i<21;i++){sim.data.qpos[jid(names[i])]=o[i];sim.data.qvel[did(names[i])]=o[21+i];}sim.data.qvel.set(o.slice(49,55),0);sim.forward();};
  const setTargets=(row)=>{for(let i=0;i<12;i++)sim.data.ctrl[aid(names[i])]=row.q_target_leg[i];for(let i=12;i<21;i++)sim.data.ctrl[aid(names[i])]=row.observation[i];};
  const obs=()=>{const d=sim.data,q=d.qpos,v=d.qvel,root=[q[0],q[1],q[2]], out=new Array(67);for(let i=0;i<21;i++)out[i]=q[jid(names[i])];for(let i=0;i<21;i++)out[21+i]=v[did(names[i])];out[42]=root[2];const bq=[q[3],q[4],q[5],q[6]],w=bq[0],x=bq[1],y=bq[2],z=bq[3],qa=(a,p)=>{const t=[2*(a[1]*p[2]-a[2]*p[1]),2*(a[2]*p[0]-a[0]*p[2]),2*(a[0]*p[1]-a[1]*p[0])];return [p[0]+a[0]*t[0]+a[1]*t[2]-a[2]*t[1],p[1]+a[0]*t[1]+a[2]*t[0]-a[1]*t[2],p[2]+a[0]*t[2]+a[1]*t[1]-a[2]*t[0]];};const yaw=Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),hi=[Math.cos(yaw/2),0,0,-Math.sin(yaw/2)],qh=[hi[0]*w-hi[3]*z,hi[0]*x+hi[3]*y,hi[0]*y-hi[3]*x,hi[0]*z+hi[3]*w];out.splice(43,0,...qa(qh,[1,0,0]),...qa(qh,[0,0,1]));out.splice(49,0,...qa(hi,[v[0],v[1],v[2]]),...qa(hi,[v[3],v[4],v[5]]));const pp=[];for(const n of paws){const bi=body(n),p=[d.xpos[3*bi],d.xpos[3*bi+1],d.xpos[3*bi+2]];pp.push(...qa(hi,[p[0]-root[0],p[1]-root[1],p[2]-root[2]]));}out.splice(55,0,...pp);return out.slice(0,67);};
  const maxes={};let first={};for(const row of trace.rows.filter(r=>r.command_name==='W')){
    if(row.step===0){setState(row.observation);continue;} // state was initialized from row 0
  }
  const byCommand={};
  const seg={joint_pos:[0,21],root_z:[42,43],orientation:[43,49],lin_vel:[49,52],ang_vel:[52,55],paw_pos:[55,67]};
  for(const cname of [...new Set(trace.rows.map(r=>r.command_name))]){const rows=trace.rows.filter(r=>r.command_name===cname);setState(rows[0].observation);const errs=[];for(let i=0;i<rows.length-1;i++){setTargets(rows[i]);for(let k=0;k<5;k++)sim.step();const got=obs(),exp=rows[i+1].observation;const e={};for(const [n,[a,b]] of Object.entries(seg))e[n]=Math.max(...got.slice(a,b).map((v,j)=>Math.abs(v-exp[a+j])));errs.push(e);for(const [n,v] of Object.entries(e)){maxes[n]=Math.max(maxes[n]||0,v);if(v>=.01&&!first[`${cname}:${n}:0.01`])first[`${cname}:${n}:0.01`]={step:i+1,error:v};if(v>=.1&&!first[`${cname}:${n}:0.1`])first[`${cname}:${n}:0.1`]={step:i+1,error:v};}}byCommand[cname]={max:{...Object.fromEntries(Object.keys(seg).map(n=>[n,Math.max(...errs.map(e=>e[n]))]))},first:errs.map((e,step)=>({step,...e})).find(e=>Object.values(e).some(v=>typeof v==='number'&&v>=.01))||null};}
  return {max:maxes,first,byCommand};
},{trace});
fs.writeFileSync(outPath,JSON.stringify(report,null,2));await page.screenshot({path:videoPath,fullPage:true});await browser.close();console.log(JSON.stringify({...report,reportPath:outPath,screenshotPath:videoPath},null,2));
