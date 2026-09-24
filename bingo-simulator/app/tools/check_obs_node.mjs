// Node-only observation comparison. No browser, WASM, React, or MuJoCo imports.
import fs from "node:fs";
import {buildCommandObs} from "../src/game/runtime/command_runtime.js";

const expected = JSON.parse(fs.readFileSync("../../scratch/expected_obs.json", "utf8"));
const names = expected.obs_dof_indexes;
const sim = (state) => {
  const qposAdr = Object.fromEntries(names.map((n,i) => [n,7+i]));
  const dofAdr = Object.fromEntries(names.map((n,i) => [n,6+i]));
  const data = {qpos: [state.root_pos[0],state.root_pos[1],state.root_pos[2],...state.root_quat,...state.qpos],
               qvel: [...state.root_lin_vel,...state.root_ang_vel,...state.qvel]};
  return {
    data, qposAdr, dofAdr,
    baseQuat: () => state.root_quat,
    bodyPos: (n) => n === "origin" ? state.root_pos : state.knee_body_pos[expected.paw_order.indexOf(n)],
    bodyQuat: (n) => state.knee_body_quat[expected.paw_order.indexOf(n)],
  };
};
const bounds = Object.fromEntries(expected.segments.map(([n,k],i) => {
  const start = expected.segments.slice(0,i).reduce((a,[,x]) => a+x,0); return [n,[start,start+k]];
}));
let worst = 0;
for (const c of expected.cases) {
  const got = Array.from(buildCommandObs(sim(c.state), c.command, c.phase, c.feedforward, c.filtered_action));
  const err = got.map((x,i) => Math.abs(x-c.observation[i]));
  worst = Math.max(worst, ...err);
  console.log(`case ${c.name}: max ${Math.max(...err).toExponential(6)}`);
  for (const [name,[a,b]] of Object.entries(bounds)) console.log(`  ${name}: ${Math.max(...err.slice(a,b)).toExponential(6)}`);
  if (got.length !== 95) throw new Error(`${c.name}: JS length ${got.length}`);
}
console.log(`MAX_ABS_ERROR ${worst.toExponential(9)}`);
process.exit(worst < 1e-4 ? 0 : 1);
