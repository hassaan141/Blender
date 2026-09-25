import {chromium} from "playwright";
const reps=+(process.argv[2]||5);
const b=await chromium.launch({executablePath:process.env.CHROME_PATH,args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const p=await(await b.newContext({viewport:{width:1000,height:700}})).newPage();
p.on("pageerror",e=>console.log("PAGEERROR:",e.message));
await p.goto("http://localhost:5173/",{waitUntil:"load",timeout:90000});
await p.waitForFunction(()=>window.__bingo?.runtime?.snapshot?.().hasPolicy,null,{timeout:90000});
await p.waitForTimeout(3000);
let ok=0;
for(let r=0;r<reps;r++){
  await p.evaluate(()=>window.__bingo.runtime.reset());
  await p.waitForTimeout(1500);
  const started=await p.evaluate(()=>window.__bingo.runtime.skills.request("Timid"));
  let st="",minZ=Infinity,maxT=0,t=0;
  for(let i=0;i<130;i++){
    await p.waitForTimeout(100); t+=0.1;
    const s=await p.evaluate(()=>{const rt=window.__bingo.runtime,sn=rt.snapshot();
      return {z:rt.sim.bodyPos("origin")[2],ti:sn.tiltDeg??0,st:sn.skillState};});
    minZ=Math.min(minZ,s.z); maxT=Math.max(maxT,s.ti||0); st=s.st;
    if(s.st==="FALLEN") break;
    if(i>15&&s.st==="STAND") break;
  }
  const good = started && st==="STAND";
  if(good) ok++;
  console.log(`run${r+1} started=${started} end=${st} t=${t.toFixed(1)}s minZ=${minZ.toFixed(3)} maxTilt=${maxT.toFixed(1)}`);
}
console.log(`\nTimid: ${ok}/${reps} completed and returned to STAND`);
await b.close();
