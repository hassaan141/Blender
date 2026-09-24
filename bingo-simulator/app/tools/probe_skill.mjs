// Run a tracked skill in the real browser runtime (MuJoCo WASM + onnxruntime-web), exactly
// as the skill button does: stand, request(skill), run until done, then stand again.
// The skill ONNX, reference JSON and index entry are served in memory, so nothing in
// public/ changes before validation. Records a per-step trace for the Python parity test.
//
// usage: node probe_skill.mjs <skill.onnx> <reference.json> <out.json> [name] [runs]
//        node probe_skill.mjs deployed - <out.json> [name] [runs]   (uses public/ assets as shipped)
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const deployed = process.argv[2] === 'deployed';
const [onnx, refFile, output] = process.argv.slice(2, 5).map((p) => path.resolve(p));
const name = process.argv[5] || 'Timid', runs = Number(process.argv[6] || 1);
const rscale = process.env.PROBE_RESIDUAL_SCALE ? JSON.parse(process.env.PROBE_RESIDUAL_SCALE) : 0.25;
const server = await createServer({root: app, server: {host: '127.0.0.1', port: 0}, logLevel: 'error'});
let browser;
try {
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  browser = await chromium.launch({headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader']});
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.error('page:', m.text()); });
  const index = JSON.parse(fs.readFileSync(path.join(app, 'public/motions/index.json')));
  index.tracked = [{name, reference: './motions/probe_skill_ref.json', policy: './policies/probe_skill.onnx',
    obs_dim: 82, residual_scale: rscale, evidence: 'probe'}];
  index.not_ready = (index.not_ready || []).filter((n) => n.name !== name);
  await page.route('**/probe.html', (r) => r.fulfill({contentType: 'text/html', body: '<html><body>skill probe</body></html>'}));
  if (!deployed) {
  await page.route('**/motions/index.json', (r) => r.fulfill({contentType: 'application/json', body: JSON.stringify(index)}));
  await page.route('**/motions/probe_skill_ref.json', (r) => r.fulfill({contentType: 'application/json', body: fs.readFileSync(refFile)}));
  await page.route('**/policies/probe_skill.onnx', (r) => r.fulfill({contentType: 'application/octet-stream', body: fs.readFileSync(onnx)}));
  }
  await page.route('**/vendor/**', (route) => {
    const m = new URL(route.request().url()).pathname.match(/^\/vendor\/(mujoco|ort)\/([\w.\-]+)$/);
    if (!m) return route.abort();
    const file = path.join(app, m[1] === 'mujoco' ? 'node_modules/@mujoco/mujoco' : 'node_modules/onnxruntime-web/dist', m[2]);
    if (!fs.existsSync(file)) return route.abort();
    return route.fulfill({contentType: file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript', body: fs.readFileSync(file)});
  });
  await page.goto(new URL('/probe.html', origin).href);
  const result = await page.evaluate(async ({name, runs}) => {
    const {BingoRuntime} = await import('/src/game/runtime/sim.js');
    const {loadSkills} = await import('/src/game/skills/load.js');
    const rt = await new BingoRuntime().init({skills: await loadSkills()});
    const skill = rt.skills.skills.get(name);
    if (!skill?.ready) return {error: `skill not ready: ${skill?.reason}`};
    const out = [];
    for (let run = 0; run < runs; run++) {
      rt.reset();
      for (let i = 0; i < 24 + run * 7; i++) await rt.controlStep();   // stand (vary expression phase)
      const d = rt.sim.data, snap = () => ({qpos: Array.from(d.qpos), qvel: Array.from(d.qvel), ctrl: Array.from(d.ctrl)});
      const start = snap();
      if (!rt.skills.request(name)) return {error: rt.skills.lastRefusal};
      const tr = skill.tracker, steps = [];
      let origObserve = tr.observe.bind(tr), origStep = tr.step.bind(tr), cur = null;
      tr.observe = (sim) => { const o = origObserve(sim); cur = {obs: Array.from(o)}; return o; };
      tr.step = (a) => { cur.act = Array.from(a); return origStep(a); };
      let state = rt.skills.state, n = 0;
      while (rt.skills.state === 'GESTURE' && n < 400) {
        await rt.controlStep(); n++;
        if (run === 0) steps.push({...cur, ...snap(), k: tr.k});
      }
      const endState = rt.skills.state, completed = endState === 'STAND' && !tr.fallen;
      if (run === 0) out.push({expr0: tr.expr0, anchor: tr.anchor});
      for (let i = 0; i < 48; i++) await rt.controlStep();            // back to standing
      const q = rt.sim.data.qpos, tilt = Math.acos(Math.min(1, 1 - 2*(q[4]*q[4] + q[5]*q[5]))) * 180 / Math.PI;
      out.push({run, completed, steps_run: n, end_state: endState, final_k: tr.k, fallen: tr.fallen,
        after_stand: {z: q[2], tilt_deg: tilt, state: rt.skills.state}, ...(run === 0 ? {start, steps} : {})});
      tr.observe = origObserve; tr.step = origStep;
    }
    return {name, runs: out};
  }, {name, runs});
  fs.writeFileSync(output, JSON.stringify(result));
  if (result.error) { console.error(result.error); process.exit(2); }
  for (const r of result.runs.filter((r) => 'run' in r))
    console.log(`run ${r.run}: completed=${r.completed} steps=${r.steps_run} end=${r.end_state} after_stand z=${r.after_stand.z.toFixed(3)} tilt=${r.after_stand.tilt_deg.toFixed(1)} state=${r.after_stand.state}`);
  console.log(`RESULT=${output}`);
} finally {
  await browser?.close(); await server.close();
}
