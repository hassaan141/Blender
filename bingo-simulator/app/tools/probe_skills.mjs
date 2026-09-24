// Gate the exported gesture clips in the ACTUAL simulator.
//
// Isaac world-space drift is not the question the browser asks: the browser never
// teleports the root, it tracks the clip's joint angles as PD targets under gravity
// and contact. So the only honest test is to play each clip here and measure whether
// Bingo holds it. A clip is promoted only on these numbers.
//
//   node tools/probe_skills.mjs [url]
import {chromium} from "playwright";
import fs from "node:fs";

const URL_ = process.argv[2] || "http://localhost:5173/";
const OUT = process.argv[3]
  || "/private/tmp/claude-501/-Users-hassaan-Projects-Blender/15f9b2e8-d83b-40d4-9727-047f34f7fb19/scratchpad/skill_gate.json";
const FALL_HEIGHT = 0.091;   // 0.5 * STAND_BASE_HEIGHT
const FALL_TILT = 70;

const browser = await chromium.launch({executablePath: process.env.CHROME_PATH,
  args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"]});
const page = await (await browser.newContext({viewport: {width: 1000, height: 700}})).newPage();
await page.goto(URL_, {waitUntil: "load", timeout: 60000});
await page.waitForFunction(() => window.__bingo?.runtime?.snapshot?.().hasPolicy,
                           null, {timeout: 60000});
await page.waitForTimeout(2500);

const clips = await page.evaluate(async () => {
  const r = await fetch("motions/index.json");
  return (await r.json()).ready.map((e) => ({name: e.name, dur: e.duration_s,
                                             candidate: !!e.candidate}));
});

const rows = [];
for (const c of clips) {
  // fresh start for every clip
  await page.evaluate(() => window.__bingo.runtime.reset());
  await page.waitForTimeout(1200);
  const started = await page.evaluate((n) => window.__bingo.runtime.skills.request(n), c.name);

  let minZ = Infinity, maxTilt = 0, fell = false, completed = false, lastState = "";
  const steps = Math.ceil((c.dur + 2.0) / 0.1);
  for (let i = 0; i < steps; i++) {
    await page.waitForTimeout(100);
    const s = await page.evaluate(() => {
      const rt = window.__bingo.runtime, sn = rt.snapshot();
      return {z: rt.sim.bodyPos("origin")[2], tilt: sn.tiltDeg ?? 0,
              state: sn.skillState, contacts: sn.contactCount ?? null};
    });
    minZ = Math.min(minZ, s.z);
    maxTilt = Math.max(maxTilt, s.tilt || 0);
    lastState = s.state;
    if (s.state === "FALLEN" || s.z < FALL_HEIGHT) fell = true;
    if (i > 3 && s.state === "STAND" && !fell) completed = true;
  }
  const end = await page.evaluate(() => {
    const rt = window.__bingo.runtime, sn = rt.snapshot();
    return {contacts: sn.contactCount ?? null, state: sn.skillState};
  });
  const pass = started && !fell && maxTilt < FALL_TILT;
  rows.push({name: c.name, candidate: c.candidate, started, pass, completed,
             minZ: +minZ.toFixed(3), maxTiltDeg: +maxTilt.toFixed(1),
             endContacts: end.contacts, endState: end.state});
  console.log(`${c.name.padEnd(13)} start=${String(started).padEnd(5)} ` +
              `pass=${String(pass).padEnd(5)} minZ=${minZ.toFixed(3)} ` +
              `tilt=${maxTilt.toFixed(1)}deg contacts=${end.contacts}/4 end=${end.state}`);
}
fs.writeFileSync(OUT, JSON.stringify({fall_height: FALL_HEIGHT, fall_tilt_deg: FALL_TILT, rows}, null, 1));
console.log("\nwrote " + OUT);
await browser.close();
