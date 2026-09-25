"""One backward hypothesis using the existing train/eval stack and bounded loop hooks."""
import argparse,os,subprocess,sys,time,shutil
from pathlib import Path
from core import ROOT,read,save,digest
from adapter import command,comparison
p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','compare']);p.add_argument('--context',type=Path,required=True);a=p.parse_args();c=read(a.context);arm=Path(c['arm_dir'])
env={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu']),'PYTHONDONTWRITEBYTECODE':'1',
     'BINGO_NATURAL_REFERENCE':str(arm/'reference.npz'),'BINGO_NATURAL_CONTACT_FIX':'1'}

def run(cmd):
    print('EXEC',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)

def native(stage):
    cmd=command(stage,c);cmd[1]=str(ROOT/'rl/bingo_rl/experiment_loop/backward_worker.py')
    if stage=='evaluate':cmd[cmd.index('--cmd_vx')+1]='-0.20'
    return cmd

if a.stage=='train':
    run([c['python'],str(ROOT/'rl/bingo_rl/experiment_loop/backward_analyze.py'),'reference',str(arm)])
    save(arm/'changed_files.json',{'historical_files_changed':[], 'new_reference':'reference.npz',
       'reference_change':'Only stored velocity channels multiplied by negative signed phase rate. Original pose/contact samples unchanged; existing negative phase reverses their traversal.',
       'runtime_change':'fixed negative command range; no reward/actuator changes','code':['backward_worker.py','backward_adapter.py','backward_analyze.py','backward_kinematic.py']})
    run([c['python'],str(ROOT/'rl/bingo_rl/experiment_loop/backward_kinematic.py'),'--reference',str(arm/'reference.npz'),'--out',str(arm/'kinematic.mp4'),'--headless','--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window'])
    run(['/usr/bin/ffmpeg','-v','error','-i',str(arm/'kinematic.mp4'),'-vf','fps=4,scale=640:-1,tile=4x4','-frames:v','1',str(arm/'kinematic_sheet.jpg')])
    print('KINEMATIC_REVIEW_READY',flush=True)
    deadline=time.monotonic()+300
    while not (arm/'KINEMATIC_GATE.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Kinematic visual review timed out; training not started')
        time.sleep(1)
    if not read(arm/'KINEMATIC_GATE.json')['allow_training']:raise RuntimeError('Kinematic reference rejected')
    pre=native('evaluate');pre[pre.index('--checkpoint')+1]=c['training_parent']['checkpoint'];pre[pre.index('--out_dir')+1]=str(arm/'preflight');pre[pre.index('--duration_s')+1]='10'
    save(arm/'preflight.command.json',{'argv':pre});run(pre)
    cmd=native('train');save(arm/'train.command.json',{'argv':cmd,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','BINGO_NATURAL_REFERENCE','BINGO_NATURAL_CONTACT_FIX']}});run(cmd)
    expected=c['iterations']*24;cp=list((arm/'training').glob(f'*/checkpoints/agent_{expected}.pt'))
    if len(cp)!=1:raise RuntimeError('Final checkpoint missing')
    shutil.copy2(cp[0],arm/'policy.pt');save(arm/'training_completion.json',{'iterations':c['iterations'],'steps':expected,'sha256':digest(arm/'policy.pt'),'source':str(cp[0])})
elif a.stage=='evaluate':
    cmd=native('evaluate');save(arm/'evaluate.command.json',{'argv':cmd});run(cmd)
    run([c['python'],str(ROOT/'rl/bingo_rl/experiment_loop/backward_analyze.py'),'evaluation',str(arm)])
else:
    comparison(arm,Path(c['parent']['video']),c['duration_s'])
    run(['/usr/bin/ffmpeg','-v','error','-i',str(arm/'comparison.mp4'),'-vf','fps=1/5,scale=960:-1,tile=2x6','-frames:v','1',str(arm/'comparison_overview.jpg')])
    run(['/usr/bin/ffmpeg','-v','error','-ss','8','-i',str(arm/'comparison.mp4'),'-vf','fps=8,scale=960:-1,tile=2x6','-frames:v','1',str(arm/'comparison_stride.jpg')])
