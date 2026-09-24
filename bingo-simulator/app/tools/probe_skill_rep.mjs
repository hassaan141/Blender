import {chromium} from "playwright";
const names=(process.argv[2]||"Yes,No,What").split(",");
const reps=+(process.argv[3]||3);
const b=await chromium.launch({executablePath:process.env.CHROME_PATH,args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const p=await(await b.newContext({viewport:{width:900,height:650}})).newPage();
await p.goto("http://localhost:5173/",{waitUntil:"load",timeout:60000});
await p.waitForFunction(()=>window.__bingo?.runtime?.snapshot?.().hasPolicy,null,{timeout:60000});
await p.waitForTimeout(2500);
const idx=await p.evaluate(async()=>(await (await fetch("motions/index.json")).json()).ready);
for(const nm of names){
  const c=idx.find(e=>e.name===nm); const out=[];
  for(let r=0;r<reps;r++){
    await p.evaluate(()=>window.__bingo.runtime.reset());
    await p.waitForTimeout(1500);
    await p.evaluate((n)=>window.__bingo.runtime.skills.request(n),nm);
    let minZ=Infinity,maxT=0,fell=false;
    const steps=Math.ceil((c.duration_s+1.5)/0.1);
    for(let i=0;i<steps;i++){
      await p.waitForTimeout(100);
      const s=await p.evaluate(()=>{const rt=window.__bingo.runtime,sn=rt.snapshot();
        return {z:rt.sim.bodyPos("origin")[2],t:sn.tiltDeg??0,st:sn.skillState};});
      minZ=Math.min(minZ,s.z); maxT=Math.max(maxT,s.t||0);
      if(s.st==="FALLEN"||s.z<0.091) fell=true;
    }
    out.push({pass:!fell&&maxT<70,minZ:+minZ.toFixed(3),tilt:+maxT.toFixed(1)});
  }
  const ok=out.filter(o=>o.pass).length;
  console.log(`${nm.padEnd(6)} ${ok}/${reps} pass  ` + out.map(o=>`[${o.pass?"ok":"FALL"} z${o.minZ} t${o.tilt}]`).join(" "));
}
await b.close();
