import argparse,sys,subprocess,os,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read,save,digest
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--context',required=True);a=p.parse_args();c=read(a.context);arm=Path(c['arm_dir']);HERE=Path(__file__).parent
E={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu']),'PYTHONDONTWRITEBYTECODE':'1','BINGO_NATURAL_REFERENCE':str(arm/'reference.npz')}
KIT='--/rtx/verifyDriverVersion/enabled=false --no-window'
def run(cmd):
 print('EXEC',cmd,flush=True);subprocess.run(cmd,cwd=ROOT,env=E,check=True)
def evaluator(name,yaw,duration,video=True,checkpoint=None):
 out=arm/name
 cmd=[c['python'],str(HERE/'worker.py'),'evaluate','--context',str(arm/'config.json'),'--task','Bingo-Turning-Play-v0','--checkpoint',str(checkpoint or arm/'policy.pt'),'--out_dir',str(out),'--duration_s',str(duration),'--cmd_vx',str(c['settings'].get('turn_command_vx',.175)),'--cmd_yaw',str(yaw),'--diagnostics','--headless','--kit_args',KIT]
 if not video:cmd+=['--no_video']
 save(arm/(name+'_command.json'),{'argv':cmd,'environment':E and {k:E[k] for k in ['CUDA_VISIBLE_DEVICES','BINGO_NATURAL_REFERENCE']}});run(cmd);run([c['python'],str(HERE/'analyze.py'),str(out)])
if a.stage=='train':
 run([c['python'],str(HERE/'geometry.py'),str(arm)])
 run([c['python'],str(HERE/'migrate.py'),c['training_parent']['checkpoint'],str(arm/'warmstart.pt')])
 # Two short physics previews, no optimizer updates; inspect before training.
 evaluator('preflight_left',.4,6,True,arm/'warmstart.pt');evaluator('preflight_right',-.4,6,True,arm/'warmstart.pt')
 for name in ['preflight_left','preflight_right']:
  run(['/usr/bin/ffmpeg','-v','error','-n','-i',str(arm/name/'eval.mp4'),'-vf','fps=4,scale=480:-1,tile=4x4','-frames:v','1',str(arm/(name+'.jpg'))])
 print('REFERENCE_REVIEW_READY',flush=True);limit=time.monotonic()+1800
 while not (arm/'KINEMATIC_GATE.json').exists():
  if time.monotonic()>limit:raise RuntimeError('Visual review timeout')
  time.sleep(1)
 if not read(arm/'KINEMATIC_GATE.json')['allow_training']:raise RuntimeError('Reference rejected before training')
 cmd=[c['python'],str(HERE/'worker.py'),'train','--context',str(arm/'config.json'),'--task','Bingo-Turning-Train-v0','--num_envs',str(c['num_envs']),'--seed',str(c['seed']),'--max_iterations',str(c['iterations']),'--checkpoint',str(arm/'warmstart.pt'),'agent.agent.experiment.directory='+str(arm/'training'),'hydra.run.dir='+str(arm/'hydra'),'--headless','--kit_args',KIT]
 save(arm/'train.command.json',{'argv':cmd,'environment':{k:E[k] for k in ['CUDA_VISIBLE_DEVICES','BINGO_NATURAL_REFERENCE']}});run(cmd)
 checkpoints=list((arm/'training').glob(f'*/checkpoints/agent_{c["iterations"]*24}.pt'));assert len(checkpoints)==1
 shutil.copyfile(checkpoints[0],arm/'policy.pt');save(arm/'training_completion.json',{'iterations':c['iterations'],'checkpoint_sha256':digest(arm/'policy.pt')})
elif a.stage=='evaluate':
 for name,yaw,seconds,video in [('left',.4,60,True),('right',-.4,60,True),('left_max',.6,30,False),('right_max',-.6,30,False),('straight',0.,30,False)]:evaluator(name,yaw,seconds,video)
 metrics={name:read(arm/name/'metrics.json') for name in ['left','right','left_max','right_max','straight']};fail=[]
 for name,m in metrics.items():
  n=read(arm/name/'natural_metrics.json');q=read(arm/name/'quality_metrics.json')
  for foot,values in n['feet'].items():
   if values['duty']>.85 or values['clearance_p95_mm']<5:fail.append(name+' '+foot+' dragging')
  if m['survival']!=1 or q['warmup_falls']:fail.append(name+' survival')
  if n['joint_limit_violation_max_rad']>1e-5:fail.append(name+' joint limits')
  if m['joint_accel_rms']>13.20023:fail.append(name+' acceleration')
  if m['mean_slip_m_s']>.24:fail.append(name+' slip')
  if max(m['torque_saturation_pct_per_joint'].values())>10:fail.append(name+' saturation')
  if not .12<=m['vx_mean']<=.23 or m['vx_std']>.075:fail.append(name+' forward smoothness')
  cmd=m['command']['yaw']
  if abs(m['yaw_mean']-cmd)>(.15 if cmd else .08):fail.append(name+' yaw tracking')
  if m['yaw_std']>.3:fail.append(name+' yaw oscillation')
 out=arm/'evaluation';out.mkdir();shutil.copyfile(arm/'left/eval.mp4',out/'eval.mp4')
 summary=dict(metrics['left']);summary.update(survival=min(m['survival'] for m in metrics.values()),any_fell=any(m['any_fell'] for m in metrics.values()),joint_accel_rms=max(m['joint_accel_rms'] for m in metrics.values()),vx_std=max(metrics[n]['vx_std'] for n in ['left','right']),turning_failures=fail)
 save(out/'metrics.json',summary);save(arm/'turning_metrics.json',metrics);save(arm/'turning_gates.json',{'failures':fail})
 natural=read(arm/'left/natural_metrics.json');natural['joint_limit_violation_max_rad']=max(read(arm/n/'natural_metrics.json')['joint_limit_violation_max_rad'] for n in metrics);save(out/'natural_metrics.json',natural);shutil.copyfile(arm/'left/quality_metrics.json',out/'quality_metrics.json')
else:
 f="[0:v]crop=720:520:280:180,drawtext=text='LEFT +0.4':x=15:y=15:fontsize=24:fontcolor=white[a];[1:v]crop=720:520:280:180,drawtext=text='RIGHT -0.4':x=15:y=15:fontsize=24:fontcolor=white[b];[a][b]hstack[v]"
 run(['/usr/bin/ffmpeg','-v','error','-n','-i',str(arm/'left/eval.mp4'),'-i',str(arm/'right/eval.mp4'),'-filter_complex',f,'-map','[v]','-t','60','-c:v','libx264','-crf','20',str(arm/'left_right.mp4')])
 from adapter import comparison
 comparison(arm,Path(c['parent']['video']),60)
 run(['/usr/bin/ffmpeg','-v','error','-n','-ss','8','-i',str(arm/'left_right.mp4'),'-vf','fps=8,scale=960:-1,tile=2x6','-frames:v','1',str(arm/'turn_stride.jpg')])
 run(['/usr/bin/ffmpeg','-v','error','-n','-i',str(arm/'left_right.mp4'),'-vf','fps=1/5,scale=960:-1,tile=2x6','-frames:v','1',str(arm/'turn_overview.jpg')])
