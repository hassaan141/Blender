// Closed-loop walk survival test. Loads the app fresh (picks up whatever physics the
// served MJCF currently has), forces a forward command, and reports how long the robot
// stays upright, how far it walks, and its min body height.
import {chromium} from "playwright";
const URL_ = process.argv[2] || "http://localhost:5173/";
const VX   = parseFloat(process.argv[3] ?? "0.20");
const YAW  = parseFloat(process.argv[4] ?? "0");
const SECS = parseFloat(process.argv[5] ?? "8");
const CHROME = process.env.CHROME_PATH;
const browser = await chromium.launch({executablePath:CHROME||undefined,
  args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const page = await (await browser.newContext({viewport:{width:900,height:600}})).newPage();
await page.goto(URL_,{waitUntil:"networkidle",timeout:60000});
await page.waitForFunction(()=>window.__bingo?.runtime?.snapshot()?.hasPolicy,null,{timeout:60000});
await page.waitForTimeout(2500);
const start = await page.evaluate(()=>window.__bingo.runtime.sim.bodyPos("origin"));
const RS = process.env.RES_SCALE!=null ? parseFloat(process.env.RES_SCALE) : null;
await page.evaluate(({vx,yaw,rs})=>{if(rs!=null)window.__resScale=rs;window.__walk=setInterval(()=>window.__bingo.runtime.setCommand(vx,yaw),16);},{vx:VX,yaw:YAW,rs:RS});

const N = Math.round(SECS/0.25);
const series=[];
for(let i=0;i<N;i++){
  await page.waitForTimeout(250);
  const s = await page.evaluate(()=>{const r=window.__bingo.runtime,sn=r.snapshot();
    return {z:r.sim.bodyPos("origin")[2], state:sn.skillState, contacts:sn.contactCount??null, cmd:sn.cmd};});
  series.push(s);
}
const end = await page.evaluate(()=>window.__bingo.runtime.sim.bodyPos("origin"));
await browser.close();

let fellAt=null, minZ=Infinity;
series.forEach((s,i)=>{minZ=Math.min(minZ,s.z); if(fellAt===null && (s.z<0.11||s.state==="FALLEN")) fellAt=(i+1)*0.25;});
const dist=Math.hypot(end[0]-start[0],end[1]-start[1]);
const finalState=series.at(-1);
const survived = fellAt===null && finalState.z>0.11 && finalState.state!=="FALLEN";
console.log(JSON.stringify({vx:VX, survived, fellAt, minZ:+minZ.toFixed(3),
  finalZ:+finalState.z.toFixed(3), finalState:finalState.state, dist:+dist.toFixed(3),
  cmdReached:finalState.cmd}));
