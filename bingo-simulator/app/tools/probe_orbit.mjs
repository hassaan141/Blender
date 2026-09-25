import {chromium} from "playwright";
const S="/private/tmp/claude-501/-Users-hassaan-Projects-Blender/15f9b2e8-d83b-40d4-9727-047f34f7fb19/scratchpad/";
const b=await chromium.launch({executablePath:process.env.CHROME_PATH,args:["--no-sandbox","--use-gl=angle","--use-angle=swiftshader"]});
const p=await(await b.newContext({viewport:{width:1000,height:700}})).newPage();
p.on("pageerror",e=>console.log("PAGEERROR:",e.message));
await p.goto("http://localhost:5173/",{waitUntil:"load",timeout:90000});
await p.waitForFunction(()=>window.__bingo?.runtime,null,{timeout:90000});
await p.waitForTimeout(4000);
const cam=()=>p.evaluate(()=>{const c=window.__bingo.rig?[0,0,0]:null;return null;});
const pos=async()=>await p.evaluate(()=>{const t=window.__bingo.THREE;return null;});
await p.screenshot({path:S+"orbit_0.png"});
// ctrl+drag right
await p.mouse.move(500,350);
await p.keyboard.down("Control");
await p.mouse.down(); await p.mouse.move(760,330,{steps:12}); await p.mouse.up();
await p.keyboard.up("Control");
await p.waitForTimeout(1200);
await p.screenshot({path:S+"orbit_1.png"});
// ctrl+wheel zoom out
await p.keyboard.down("Control"); await p.mouse.wheel(0,600); await p.keyboard.up("Control");
await p.waitForTimeout(1200);
await p.screenshot({path:S+"orbit_2.png"});
console.log("ok"); await b.close();
