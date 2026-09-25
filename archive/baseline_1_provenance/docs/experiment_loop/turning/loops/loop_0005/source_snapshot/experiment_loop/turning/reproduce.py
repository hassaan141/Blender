"""Replay a sealed turning candidate into a fresh directory; never train or overwrite it."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--yaw',type=float,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--no-video',action='store_true');a=p.parse_args()
if not -.6<=a.yaw<=.6 or not 0<a.seconds<=120:raise ValueError('Replay bounds: yaw ±.6, duration <=120s')
src=a.candidate.resolve();out=a.out.resolve();c=json.loads((src/'config.json').read_text())
for name in ['policy.pt','reference.npz','turn_bank.npz']:
 if not (src/name).is_file():raise FileNotFoundError(src/name)
out.mkdir(parents=True,exist_ok=False)
for name in ['reference.npz','turn_bank.npz']:(out/name).symlink_to(src/name)
c['arm_dir']=str(out);(out/'config.json').write_text(json.dumps(c,indent=2))
cmd=[c['python'],str(Path(__file__).with_name('worker.py')),'evaluate','--context',str(out/'config.json'),'--task','Bingo-Turning-Play-v0','--checkpoint',str(src/'policy.pt'),'--out_dir',str(out/'evaluation'),'--duration_s',str(a.seconds),'--cmd_vx',str(c['settings'].get('turn_command_vx',.175)),'--cmd_yaw',str(a.yaw),'--diagnostics','--headless','--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window']
if a.no_video:cmd+=['--no_video']
env={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu']),'PYTHONDONTWRITEBYTECODE':'1','BINGO_NATURAL_REFERENCE':str(out/'reference.npz')}
(out/'command.json').write_text(json.dumps({'argv':cmd,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','BINGO_NATURAL_REFERENCE']}},indent=2))
subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=900)
subprocess.run([c['python'],str(Path(__file__).with_name('analyze.py')),str(out/'evaluation')],cwd=ROOT,env=env,check=True,timeout=60)
print(out/'evaluation/metrics.json')
