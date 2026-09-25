"""Bingo hooks. No simulator imports until the explicitly requested worker starts."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
from core import ROOT, read, save, digest

def comparison(arm, champion_video, seconds=60):
    filt="[0:v]crop=720:520:280:180,drawtext=text='CHAMPION':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[a];[1:v]crop=720:520:280:180,drawtext=text='CANDIDATE':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[b];[a][b]hstack=inputs=2:shortest=1[v]"
    subprocess.run(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',str(champion_video),'-i',str(arm/'evaluation/eval.mp4'),
                    '-filter_complex',filt,'-map','[v]','-t',str(seconds),'-c:v','libx264','-crf','20',str(arm/'comparison.mp4')],check=True,timeout=300)

def command(stage,c):
    arm=Path(c['arm_dir']);parent=c['training_parent'];kit='--/rtx/verifyDriverVersion/enabled=false --no-window'
    script=ROOT/'rl/bingo_rl/experiment_loop/worker.py'
    common=[c['python'],str(script),stage,'--context',str(arm/'config.json'),'--headless']
    if stage=='train':
        return common+['--task','Bingo-NaturalWalk-Train-v0','--num_envs',str(c['num_envs']),'--seed',str(c['seed']),
                       '--max_iterations',str(c['iterations']),'--checkpoint',parent['checkpoint'],
                       'agent.agent.experiment.directory='+str(arm/'training'),'hydra.run.dir='+str(arm/'hydra'),
                       '--kit_args',kit]
    return common+['--task','Bingo-NaturalWalk-Play-v0','--diagnostics','--checkpoint',str(arm/'policy.pt'),
                   '--out_dir',str(arm/'evaluation'),'--duration_s',str(c['duration_s']),
                   '--settle_seconds','1','--cmd_vx','0.25','--kit_args',kit]

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','compare']);p.add_argument('--context',type=Path,required=True);a=p.parse_args()
    c=read(a.context);arm=Path(c['arm_dir'])
    if not arm.resolve().is_relative_to(ROOT/'docs/experiment_loop'):raise ValueError('Output outside experiment_loop')
    if c['smoke']:raise ValueError('Real hooks cannot run in smoke mode')
    if a.stage=='compare':comparison(arm,Path(c['parent']['video']),c['duration_s']);return
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu']),'PYTHONDONTWRITEBYTECODE':'1',
         'BINGO_NATURAL_REFERENCE':str(arm/'reference.npz'),
         'BINGO_NATURAL_CONTACT_FIX':'1' if c['training_parent']['contact_fix'] else '0'}
    cmd=command(a.stage,c);save(arm/f'{a.stage}.command.json',{'argv':cmd,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','BINGO_NATURAL_REFERENCE','BINGO_NATURAL_CONTACT_FIX']}})
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if a.stage=='train':
        # Original skrl saves final timestep, independent of intermediate checkpoint interval.
        expected=c['iterations']*24
        checkpoints=list((arm/'training').glob(f'*/checkpoints/agent_{expected}.pt'))
        if len(checkpoints)!=1:raise RuntimeError('Expected final checkpoint missing/ambiguous; exit 0 is insufficient')
        shutil.copy2(checkpoints[0],arm/'policy.pt')
        save(arm/'training_completion.json',{'expected_iterations':c['iterations'],'expected_steps':expected,
             'checkpoint':str(checkpoints[0]),'sha256':digest(arm/'policy.pt'),
             'limitation':'Final checkpoint filename and completed trainer call verified; optimizer update counts not independently audited'})
    else:
        m=read(arm/'evaluation/metrics.json')
        if m['checkpoint']!=str(arm/'policy.pt'):raise RuntimeError('Evaluator returned wrong checkpoint')
        probe=subprocess.run(['/usr/bin/ffprobe','-v','error','-show_entries','format=duration','-of','json',str(arm/'evaluation/eval.mp4')],capture_output=True,text=True,check=True)
        if float(json.loads(probe.stdout)['format']['duration'])<c['duration_s']-.1:raise RuntimeError('Evaluation video truncated')

if __name__=='__main__':main()
