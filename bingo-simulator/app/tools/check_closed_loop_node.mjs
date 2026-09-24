// Open-loop robot-state replay; controller state and ONNX inference run freely.
// No browser, server, MuJoCo, Isaac or GPU. ONNX uses the CPU WASM backend.
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import * as ort from 'onnxruntime-web';
import {CommandState} from '../src/game/runtime/command_runtime.js';
import {JOINT_NAMES} from '../src/game/constants.js';

const repo=new URL('../../../',import.meta.url);
const tracePath=process.argv[2] || fileURLToPath(new URL('scratch/isaac_trace.json',repo));
const trace=JSON.parse(fs.readFileSync(tracePath));
const policies=new URL('../public/policies/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(`${name}.json`,policies)));
const refs={f:read('reference_forward'),b:read('reference_backward'),t:read('reference_turning'),p:read('pivot_bank'),bt:read('backward_turn_bank'),turnBank:read('turn_bank')};
if(JSON.stringify(trace.obs_dof_indexes)!==JSON.stringify(JOINT_NAMES))throw Error('DOF ordering mismatch');
if(trace.rows.length!==1400)throw Error(`Expected 1400 rows, got ${trace.rows.length}`);
ort.env.wasm.numThreads=1;
const session=await ort.InferenceSession.create(fs.readFileSync(new URL('policy.onnx',policies)),{executionProviders:['wasm']});
const fields=['phase','feedforward','filtered_action','observation','raw_policy_action','q_target_leg'];
const empty=()=>Object.fromEntries(fields.map(k=>[k,0]));
const report={threshold:1e-4,trace:tracePath,per_command:{},overall:empty(),first_divergence:null,first_by_field:{},target_logging_formula_max_error:0};
let state,commandName;
for(const row of trace.rows){
  if(row.command_name!==commandName){
    commandName=row.command_name;
    if(row.step!==0)throw Error('Missing initial state');
    // Trace starts after settling. Seed smoothed command once too; subsequent
    // rows supply only requested command and identical 67D proprioception.
    state=new CommandState(refs,{phase:row.phase,command:[Math.fround(row.observation[69]*Math.fround(.3)),Math.fround(row.observation[70]*Math.fround(.6))],filtered:row.filtered_action});
    report.per_command[commandName]=empty();
  }
  const observation=state.observation(row.observation.slice(0,67));
  const output=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',observation,[1,95])});
  const action=Array.from(output[session.outputNames[0]].data);
  const actual={phase:[state.phase],feedforward:state.feedforward(),filtered_action:Array.from(state.filtered),observation:Array.from(observation),raw_policy_action:action};
  actual.q_target_leg=state.advance(action,row.command).qTarget;
  for(const field of fields){
    const expected=field==='phase'?[row.phase]:row[field];
    if(actual[field].length!==expected.length)throw Error(`Length mismatch: ${field}`);
    for(let i=0;i<expected.length;i++){
      const error=Math.abs(actual[field][i]-expected[i]);
      if(!Number.isFinite(error))throw Error(`Nonfinite ${field} ${i}`);
      report.per_command[commandName][field]=Math.max(report.per_command[commandName][field],error);
      report.overall[field]=Math.max(report.overall[field],error);
      if(error>=report.threshold){
        const failure={command:commandName,step:row.step,field,index:i,error,expected:expected[i],actual:actual[field][i]};
        report.first_divergence??=failure;report.first_by_field[field]??=failure;
      }
    }
  }
  // Audit the supplied target field without replacing it or weakening the gate.
  row.q_target_leg.forEach((v,i)=>{const formula=Math.fround(row.feedforward[i]+Math.fround(Math.fround(.3)*row.raw_policy_action[i]));report.target_logging_formula_max_error=Math.max(report.target_logging_formula_max_error,Math.abs(v-formula));});
}
report.pass=fields.every(k=>report.overall[k]<report.threshold);
const out=new URL('scratch/closed_loop_diff.json',repo);
fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');
console.log(['command',...fields].join('\t'));
for(const [name,stats] of Object.entries({...report.per_command,OVERALL:report.overall}))console.log([name,...fields.map(k=>stats[k].toExponential(6))].join('\t'));
console.log('First divergence:',JSON.stringify(report.first_divergence));
console.log('First per field:',JSON.stringify(report.first_by_field));
console.log('Trace target == ff + 0.3 * RAW action; max error:',report.target_logging_formula_max_error);
console.log(report.pass?'PASS':'FAIL',fileURLToPath(out));
process.exitCode=report.pass?0:1;
