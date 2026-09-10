// Verify the RENDERED robot agrees with MuJoCo.
//
// This is the one property the whole design rests on (brief §5, §16): "Robot motion
// visible on screen must originate from MuJoCo physics state." A rig that has quietly
// drifted into being a decorative animation looks fine in a screenshot, so it has to
// be measured. Every paw's world position is computed from the Three.js scene graph
// and compared with MuJoCo's own body position for the same link.
//
//   node tools/check_rig_vs_physics.mjs [url]
//
// Requires the preview server running (npm run build && npx vite preview).
import {chromium} from "playwright";

const URL_ = process.argv[2] || "http://localhost:4173/";
const CHROME = process.env.CHROME_PATH
  || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const TOL_MM = 1.0;

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
});
const page = await (await browser.newContext({viewport: {width: 1000, height: 700}}))
  .newPage();
await page.goto(URL_, {waitUntil: "networkidle", timeout: 60000});
await page.waitForFunction(() => window.__bingo?.rig && window.__bingo?.runtime,
                           null, {timeout: 60000});
await page.waitForTimeout(2500);

const res = await page.evaluate(() => {
  const {rig, runtime, THREE} = window.__bingo;
  const sim = runtime.sim;
  const OBJ = sim.mujoco.mjtObj;
  const out = [];
  for (const body of ["fl_knee", "fr_knee", "bl_knee", "br_knee",
                      "head_roll", "tail_yaw", "origin"]) {
    const g = rig.bodies.get(body);
    if (!g) { out.push({body, err: "not in rig"}); continue; }
    const w = new THREE.Vector3();
    g.getWorldPosition(w);
    // three (x, y, z) maps back to MJCF (x, -z, y)
    const three2mjcf = [w.x, -w.z, w.y];
    const bid = sim.mujoco.mj_name2id(sim.model, OBJ.mjOBJ_BODY.value, body);
    const mj = [sim.data.xpos[bid * 3], sim.data.xpos[bid * 3 + 1],
                sim.data.xpos[bid * 3 + 2]];
    const err = Math.hypot(three2mjcf[0] - mj[0], three2mjcf[1] - mj[1],
                           three2mjcf[2] - mj[2]);
    out.push({body, rig: three2mjcf, mujoco: mj, err_mm: err * 1000});
  }
  return {rows: out, state: runtime.snapshot().skillState,
          contacts: runtime.snapshot().contactCount};
});

console.log("RENDER RIG vs MUJOCO — world position per link\n");
console.log("body".padEnd(12) + "rig (m)".padEnd(30) + "mujoco (m)".padEnd(30) + "err");
let worst = 0;
for (const r of res.rows) {
  if (r.err) { console.log(`${r.body.padEnd(12)}${r.err}`); continue; }
  const f = (v) => "[" + v.map((x) => x.toFixed(4)).join(", ") + "]";
  console.log(r.body.padEnd(12) + f(r.rig).padEnd(30) + f(r.mujoco).padEnd(30)
              + `${r.err_mm.toFixed(4)} mm`);
  worst = Math.max(worst, r.err_mm);
}
console.log(`\nstate ${res.state}, ${res.contacts}/4 paws in contact`);
console.log(worst <= TOL_MM
  ? `PASS — worst disagreement ${worst.toFixed(4)} mm (tol ${TOL_MM} mm)`
  : `FAIL — worst disagreement ${worst.toFixed(4)} mm exceeds ${TOL_MM} mm`);
await browser.close();
process.exit(worst <= TOL_MM ? 0 : 1);
