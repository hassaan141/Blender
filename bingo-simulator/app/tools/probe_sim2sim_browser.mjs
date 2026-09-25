// Run the real browser MuJoCo/WASM runtime with a candidate ONNX served in memory.
// No installed policy, model, or browser source files are changed.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidate = path.resolve(process.argv[2] || path.join(app,'../../scratch/sim2sim_browser_probe/candidate.onnx'));
const output = path.resolve(process.argv[3] || path.join(app,'../../scratch/sim2sim_browser_probe/browser_results.json'));
const server = await createServer({root: app, server: {host: '127.0.0.1', port: 0}});
let browser;
try {
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  browser = await chromium.launch({...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {}),
    headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader']});
  const page = await browser.newPage();
  await page.route('**/probe.html', route => route.fulfill({contentType: 'text/html', body: '<html><body>MuJoCo probe</body></html>'}));
  await page.route('**/policies/policy.onnx', route => route.fulfill({contentType: 'application/octet-stream', body: fs.readFileSync(candidate)}));
  await page.route('**/vendor/**', route => {
    const u = new URL(route.request().url());
    const match = u.pathname.match(/^\/vendor\/(mujoco|ort)\/([\w.\-]+)$/);
    if (!match) return route.abort();
    const base = match[1] === 'mujoco' ? 'node_modules/@mujoco/mujoco' : 'node_modules/onnxruntime-web/dist';
    const file = path.join(app, base, match[2]);
    if (!fs.existsSync(file)) return route.abort();
    return route.fulfill({contentType: file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript', body: fs.readFileSync(file)});
  });
  await page.goto(new URL('/probe.html', origin).href);
  const results = await page.evaluate(async ({matchPythonExpression,pythonStart,traceCase,exprObs}) => {
    const {BingoRuntime} = await import('/src/game/runtime/sim.js');
    const rt = await new BingoRuntime().init();
    if (!rt.hasPolicy) throw Error(rt.policyError || 'policy unavailable');
    if (matchPythonExpression) {
      const {nextTime} = await import('/src/game/runtime/command_runtime.js');
      const names = ['head_pitch_joint','head_yaw','head_roll','tail_pitch','tail_yaw',
        'l_ear_pitch','l_ear_roll','r_ear_pitch','r_ear_roll'];
      rt.expr.update = () => {
        const c=rt.commandState.command.map((v,i)=>v+.1*(rt.cmdTarget[i]-v));
        const t=nextTime(rt.refs,c,rt.commandState.clipTime);
        const n=rt.refs.f.dof_positions.length;
        const frame=Math.max(0,Math.min(n-1,Math.round(t*rt.refs.f.fps)));
        return names.map(name=>rt.refs.f.dof_positions[frame][rt.refs.f.dof_names.indexOf(name)]);
      };
    }
    // PROBE_EXPR_OBS='[...18 values]': diagnostic only - hold the 9 expressive joint pos (obs 12-20)
    // and vel (33-41) inputs at the given values; physics still runs the real expression.
    if (exprObs) { const build=rt.buildObs.bind(rt); rt.buildObs=()=>{const o=build(); for(let k=0;k<9;k++){o[12+k]=exprObs[k];o[33+k]=exprObs[9+k];} rt.lastObservation.set(o); return o;}; }
    const cases = [['stand',0,0],['forward',.2,0],['backward',-.2,0],['left',0,.4],['right',0,-.4],
      ['forward_left',.15,.4],['forward_right',.15,-.4],['backward_left',-.15,.4],['backward_right',-.15,-.4]];
    const out = [];
    // PROBE_TRACE=<case>: record ctrl applied and state after every control step (settle included).
    let trace=null; const rec=()=>{if(trace){const d=rt.sim.data;trace.ctrl.push(Array.from(d.ctrl));trace.qpos.push(Array.from(d.qpos));trace.qvel.push(Array.from(d.qvel));trace.obs.push(Array.from(rt.lastObservation));trace.act.push(Array.from(rt.lastAction));}};
    for (const [name,vx,yaw] of cases) {
      rt.reset(); trace = name===traceCase ? {ctrl:[],qpos:[],qvel:[],obs:[],act:[]} : null;
      if (trace) {trace.qpos0=Array.from(rt.sim.data.qpos); trace.qvel0=Array.from(rt.sim.data.qvel);}
      if (pythonStart && (vx || yaw)) {rt.setCommand(vx,yaw);rt.skills.requestWalk(true);}
      for (let i=0;i<24;i++) {await rt.controlStep(); rec();}
      if (!pythonStart && (vx || yaw)) {rt.setCommand(vx,yaw); rt.skills.requestWalk(true);}
      let fallen=false, firstFall=null, minZ=Infinity, distance=0, sumVx=0, maxTilt=0, steps=0;
      for (let i=0;i<192;i++) {
        await rt.controlStep(); rec();
        const s=rt.stats; steps++;
        minZ=Math.min(minZ,s.baseZ); maxTilt=Math.max(maxTilt,s.tiltDeg);
        distance=s.basePos[0]; sumVx+=s.vx;
        if (!fallen && (s.baseZ<.105 || s.tiltDeg>70 || s.state==='FALLEN')) {
          fallen=true; firstFall=(i+1)/24;
        }
      }
      out.push({command:name, vx_cmd:vx, yaw_cmd:yaw, survived:!fallen,
        first_fall_seconds:firstFall, min_root_z:minZ, max_tilt_deg:maxTilt,
        distance_x:distance, mean_vx:sumVx/steps, final_state:rt.stats.state, ...(trace?{trace}:{})});
    }
    return {policy:rt.policy.name, model_timestep:rt.sim.model.opt.timestep,
      expression:matchPythonExpression?'training_reference':'browser_neutral',
      start:pythonStart?'command_before_settle':'stand_then_command', expr_obs_override:exprObs, cases:out};
  }, {matchPythonExpression:process.env.PROBE_EXPRESSION==='python',pythonStart:process.env.PROBE_START==='python',traceCase:process.env.PROBE_TRACE||null,exprObs:process.env.PROBE_EXPR_OBS?JSON.parse(process.env.PROBE_EXPR_OBS):null});
  results.mjcf_sha256=createHash('sha256').update(await (await page.request.get(new URL('/robot/bingo_scene.xml',origin).href)).body()).digest('hex');
  results.onnx_sha256=createHash('sha256').update(fs.readFileSync(candidate)).digest('hex');
  fs.mkdirSync(path.dirname(output), {recursive:true});
  fs.writeFileSync(output, JSON.stringify(results,null,2)+'\n');
  console.log(JSON.stringify({...results, cases:results.cases.map(({trace,...c})=>c)}));
  console.log(`RESULT=${output}`);
} finally {
  await browser?.close();
  await server.close();
}
