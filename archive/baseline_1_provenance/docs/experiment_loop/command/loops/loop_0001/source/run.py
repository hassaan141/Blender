"""Bounded, resumable train/evaluate execution. Agent writes one hypothesis per loop.

A completed candidate is never automatically promoted. Review metrics AND video,
write a decision, then start the next targeted loop (maximum eight).
"""
import argparse,hashlib,json,os,subprocess,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HOME=ROOT/'docs/experiment_loop/command';CODE=Path(__file__).parent
PYTHON='/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python'
p=argparse.ArgumentParser();p.add_argument('--loop',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--hypothesis',required=True);p.add_argument('--iterations',type=int,default=400);p.add_argument('--yaw-weight',type=float,default=.3);p.add_argument('--curriculum-steps',type=int,default=1920);a=p.parse_args()
assert 1<=a.loop<=8
out=HOME/'loops'/f'loop_{a.loop:04d}';out.mkdir(parents=True,exist_ok=True)
assert not (out/'decision.json').exists(),'Completed experiments are immutable'
def check():
 expected=json.loads((HOME/'PROTECTED.json').read_text());bad=[]
 for name,h in expected.items():
  f=ROOT/name
  if not f.is_file() or hashlib.file_digest(f.open('rb'),'sha256').hexdigest()!=h:bad.append(name)
 if bad:raise RuntimeError('Protected files changed: '+str(bad))
 return {'files':len(expected),'unchanged':True}
(out/'protected_before.json').write_text(json.dumps(check()))
(out/'hypothesis.md').write_text(a.hypothesis+'\n');(out/'config.json').write_text(json.dumps(vars(a),indent=2))
if not (out/'source').exists():shutil.copytree(CODE,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',COMMAND_CURRICULUM_STEPS=str(a.curriculum_steps),COMMAND_YAW_WEIGHT=str(a.yaw_weight))
base=[PYTHON];app=['--headless','--device','cuda:0','--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window']
def run(name,cmd):
 (out/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
 with (out/(name+'.log')).open('w') as log:
  child_env=dict(env)
  if name=='evaluate':child_env['CUDA_VISIBLE_DEVICES']='1'
  process=subprocess.Popen(cmd,cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
  start=time.monotonic()
  while process.poll() is None:
   time.sleep(3)
   tail=(out/(name+'.log')).read_text(errors='replace')[-10000:]
   if 'Traceback (most recent call last)' in tail or time.monotonic()-start>5400:
    process.kill();process.wait();raise RuntimeError(name+' failed or timed out; inspect log')
  result=process
 (out/'protected_after.json').write_text(json.dumps(check()))
 if result.returncode:raise RuntimeError(name+' failed; inspect '+str(out/(name+'.log')))
if not (out/'policy.pt').exists():
 run('train',base+[str(CODE/'train.py'),'--task','Bingo-Command-Train-v0','--num_envs','512','--seed','42','--max_iterations',str(a.iterations),'--checkpoint',str(Path(a.checkpoint).resolve())]+app+['agent.agent.experiment.directory='+str(out/'training'),'agent.agent.experiment.experiment_name=ppo','hydra.run.dir='+str(out/'hydra')])
 checkpoints=list((out/'training').rglob('agent_*.pt'));assert checkpoints,'No training checkpoint'
 last=max(checkpoints,key=lambda f:int(f.stem.split('_')[-1]));shutil.copy2(last,out/'policy.pt')
if not (out/'evaluation/metrics.json').exists():run('evaluate',base+[str(CODE/'evaluate.py'),'--checkpoint',str(out/'policy.pt'),'--out',str(out/'evaluation'),'--video']+app)
assert (out/'evaluation/metrics.json').is_file(), 'Evaluation exited without metrics'
assert all((out/'evaluation'/(name+'.mp4')).is_file() for name in ['stand','forward','backward','left','right','forward_left','forward_right','backward_left','backward_right','transitions']), 'Missing videos'
print('READY_FOR_VISUAL_REVIEW',out,flush=True)
