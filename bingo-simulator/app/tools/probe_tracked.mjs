// Generic gate for tracked (residual-RL) skills. Unlike probe_timid.mjs this
// sizes the watch window from the clip itself, so a long clip is not cut off
// mid-motion and scored as a failure.
//
// Usage: node tools/probe_tracked.mjs <SkillName> [reps]
import {chromium} from "playwright";

const name = process.argv[2] || "Timid";
const reps = +(process.argv[3] || 5);

const b = await chromium.launch({
  executablePath: process.env.CHROME_PATH,
  args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
});
const p = await (await b.newContext({viewport: {width: 1000, height: 700}})).newPage();
p.on("pageerror", (e) => console.log("PAGEERROR:", e.message));
await p.goto("http://localhost:5173/", {waitUntil: "load", timeout: 90000});
await p.waitForFunction(() => window.__bingo?.runtime?.snapshot?.().hasPolicy, null, {timeout: 90000});
await p.waitForTimeout(3000);

const entry = await p.evaluate((n) => {
  const s = window.__bingo.runtime.skills.byName?.(n) ??
            window.__bingo.runtime.skills.all?.find?.((x) => x.name === n);
  return s ? {found: true, dur: s.tracker?.clip?.frames / (s.tracker?.clip?.fps || 24)} : {found: false};
}, name).catch(() => ({found: false}));

// Fall back to the clip file when the runtime does not expose the skill list.
let secs = entry.dur;
if (!secs) {
  secs = await p.evaluate(async (n) => {
    const idx = await (await fetch("/motions/index.json")).json();
    const e = idx.tracked.find((x) => x.name === n);
    if (!e) return null;
    const c = await (await fetch(e.reference.replace("./", "/"))).json();
    return c.frames / c.fps;
  }, name);
}
if (!secs) { console.log(`no tracked entry for "${name}"`); await b.close(); process.exit(2); }
const steps = Math.ceil((secs + 4) * 10);   // clip + settle, at 100 ms per step
console.log(`${name}: clip ${secs.toFixed(1)}s -> watching up to ${(steps / 10).toFixed(1)}s\n`);

let ok = 0;
for (let r = 0; r < reps; r++) {
  await p.evaluate(() => window.__bingo.runtime.reset());
  await p.waitForTimeout(1500);
  const started = await p.evaluate((n) => window.__bingo.runtime.skills.request(n), name);
  let st = "", minZ = Infinity, maxT = 0, t = 0, ran = 0;
  for (let i = 0; i < steps; i++) {
    await p.waitForTimeout(100); t += 0.1;
    const s = await p.evaluate(() => {
      const rt = window.__bingo.runtime, sn = rt.snapshot();
      return {z: rt.sim.bodyPos("origin")[2], ti: sn.tiltDeg ?? 0, st: sn.skillState};
    });
    minZ = Math.min(minZ, s.z); maxT = Math.max(maxT, s.ti || 0); st = s.st;
    if (s.st === "GESTURE") ran = t;
    if (s.st === "FALLEN") break;
    if (ran > 0 && s.st === "STAND") break;   // finished and handed back to stand
  }
  const good = started && st === "STAND" && ran > 0.5 * secs;
  if (good) ok++;
  console.log(`run${r + 1} started=${started} end=${st} ran=${ran.toFixed(1)}s/${secs.toFixed(1)}s ` +
              `minZ=${minZ.toFixed(3)} maxTilt=${maxT.toFixed(1)}`);
}
console.log(`\n${name}: ${ok}/${reps} completed and returned to STAND`);
await b.close();
