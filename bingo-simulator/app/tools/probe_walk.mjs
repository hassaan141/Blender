// Headless: load the sim, confirm the policy loaded, press W, measure whether the
// robot's root actually translates forward under physics.
import {chromium} from "playwright";

const URL_ = process.argv[2] || "http://localhost:5173/";
const SHOT = process.argv[3] || "/private/tmp/claude-501/-Users-hassaan-Projects-Blender/15f9b2e8-d83b-40d4-9727-047f34f7fb19/scratchpad/walk.png";

const browser = await chromium.launch({
  args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
});
const page = await (await browser.newContext({viewport: {width: 1100, height: 760}})).newPage();
page.on("pageerror", e => console.log("PAGEERROR:", e.message));
await page.goto(URL_, {waitUntil: "networkidle", timeout: 60000});
await page.waitForFunction(() => window.__bingo?.runtime, null, {timeout: 60000});
await page.waitForTimeout(3000);

const before = await page.evaluate(() => {
  const r = window.__bingo.runtime, s = r.snapshot();
  return {pos: r.sim.bodyPos("origin"), hasPolicy: s.hasPolicy, policyName: s.policyName,
          policyError: s.policyError, skillState: s.skillState, cmd: s.cmd};
});

// Drive the command straight into the runtime. Headless Chromium throttles rAF, so
// the App's input-sampling loop (which reads the keyboard) may not run; the physics
// loop is on a wall-clock accumulator and keeps stepping. Force a forward command.
await page.evaluate(() => {
  window.__walk = setInterval(() => window.__bingo.runtime.setCommand(0.20, 0), 16);
});
await page.waitForTimeout(6000);
const during = await page.evaluate(() => {
  const r = window.__bingo.runtime, s = r.snapshot();
  return {pos: r.sim.bodyPos("origin"), skillState: s.skillState, cmd: s.cmd,
          contacts: s.contactCount ?? s.contacts ?? null};
});
await page.screenshot({path: SHOT});
await page.keyboard.up("w");

const dx = during.pos[0] - before.pos[0], dy = during.pos[1] - before.pos[1];
const dist = Math.hypot(dx, dy);
console.log("BEFORE:", JSON.stringify(before));
console.log("DURING:", JSON.stringify(during));
console.log(`ROOT MOVED: dx=${dx.toFixed(3)} dy=${dy.toFixed(3)} dist=${dist.toFixed(3)} m over ~5s`);
console.log("screenshot:", SHOT);
const upright = during.pos[2] > 0.10;
console.log(before.hasPolicy && !before.policyError && dist > 0.05 && upright
  ? `PASS — policy '${before.policyName}' loaded, robot walked ${dist.toFixed(3)} m, still upright (z=${during.pos[2].toFixed(3)})`
  : `CHECK — hasPolicy=${before.hasPolicy} err=${before.policyError} dist=${dist.toFixed(3)} z=${during.pos[2].toFixed(3)}`);
await browser.close();
