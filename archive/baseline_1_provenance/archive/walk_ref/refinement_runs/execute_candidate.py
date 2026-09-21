"""Execute one predeclared candidate; does not select hypotheses or launch more runs."""
import json,os,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[3]
os.chdir(ROOT)
name=sys.argv[1]; out=ROOT/'docs/walk_ref/refinement_runs'/name
spec=json.loads((out/'experiment.json').read_text())
assert name in [f'run_{i:02d}' for i in range(1,9)]
assert 1 <= len(spec['changes']) <= 2
assert not (out/'metrics.json').exists(), 'Candidate already evaluated'
python='/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
def execute(kind,cmd):
 (out/f'{kind}_command.sh').write_text('CUDA_VISIBLE_DEVICES=1 '+shlex.join(cmd)+'\n')
 with (out/f'{kind}.log').open('w') as f:
  subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for f in ('bingo_walk_ref_env.py','bingo_walk_ref_env_cfg.py'):
 (out/f).write_bytes((ROOT/'rl/bingo_rl/bingo_rl/walk_ref'/f).read_bytes())
if '--eval-only' not in sys.argv:
 execute('train',[python,'rl/tools/train_walk_ref.py','--task','Bingo-WalkRef-v4-C-v0','--headless','--num_envs','512','--seed','42','--max_iterations','100','--checkpoint',spec['parent_checkpoint'],*spec['train_overrides'],f'agent.agent.experiment.directory=walk_refinement/{name}','--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window'])
ckpts=list((ROOT/f'logs/skrl/walk_refinement/{name}').glob('*/checkpoints/agent_2400.pt'))
if len(ckpts)!=1:raise RuntimeError(f'Expected one final checkpoint, found {ckpts}')
(out/'policy.pt').write_bytes(ckpts[0].read_bytes())
spec['trained_checkpoint']=str(ckpts[0]);spec['status']='evaluating';(out/'experiment.json').write_text(json.dumps(spec,indent=2))
execute('eval',[python,'rl/tools/eval_walk_loop.py','--headless','--no_video','--checkpoint',str(out/'policy.pt'),'--out_dir',str(out),'--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window'])
subprocess.run([sys.executable,str(out.parent/'record_result.py'),name],check=True)
