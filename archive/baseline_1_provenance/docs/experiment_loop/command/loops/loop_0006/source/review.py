"""Metrics gates and review artifacts. Visual approval is a separate explicit input."""
import argparse,json,csv,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HOME=ROOT/'docs/experiment_loop/command'
p=argparse.ArgumentParser();p.add_argument('--loop',type=int,required=True);p.add_argument('--decision',choices=['KEEP','REJECT']);p.add_argument('--reason');p.add_argument('--visual-pass',action='store_true');a=p.parse_args()
out=HOME/'loops'/f'loop_{a.loop:04d}'
assert not (out/'decision.json').exists(), 'Sealed review is immutable; consult corrections separately'
metric_path=out/'evaluation/metrics_extended.json'
if not metric_path.exists():metric_path=out/'evaluation/metrics.json'
metrics=json.loads(metric_path.read_text());gates={}
for name,m in metrics.items():
 if name=='transitions':
  gates[name]={'survival':m['survival'],'joint_limits':m['joint_limit_violation_rad']<=.001,'vx_tracking':m.get('settled_vx_tracking_rmse',1)<.07,'yaw_tracking':m.get('settled_yaw_tracking_rmse',1)<.20};continue
 vx,yaw=m['command'];gates[name]={'survival':m['survival'],'vx':abs(m['vx_mean']-vx)<(.04 if vx==0 else .065),'yaw':abs(m['yaw_mean']-yaw)<(.10 if yaw==0 else .15),'acceleration':m['joint_acceleration_rms']<(9.6 if name=='backward' else 13.85),'torque':m['torque_saturation_max_pct']<(10 if name=='backward' else 18),'joint_limits':m['joint_limit_violation_rad']<=.001,'slip':m['slip_mean'] is not None and m['slip_mean']<.25,'rocking':m['roll_rms_deg']<8.,'vx_std':m['vx_std']<(.03 if name=='stand' else .08)}
(out/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
# Nine synchronized command panels, clearly labeled.
names=[n for n in metrics if n!='transitions'];video=out/'commands.mp4'
if not video.exists():
 cmd=['/usr/bin/ffmpeg','-y','-loglevel','error','-filter_complex_threads','1'];filters=[]
 for i,n in enumerate(names):
  cmd+=['-i',str(out/'evaluation'/(n+'.mp4'))];filters.append(f"[{i}:v]scale=320:180,drawtext=text='{n}':fontsize=17:fontcolor=white:box=1:boxcolor=black@0.6:x=8:y=8[v{i}]")
 layout='|'.join(f'{i%3*320}_{i//3*180}' for i in range(9));filters.append(''.join(f'[v{i}]' for i in range(9))+f'xstack=inputs=9:layout={layout}[v]')
 subprocess.run(cmd+['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','21','-pix_fmt','yuv420p',str(video)],check=True)
for name in ['left','backward_left','forward','backward','transitions']:
 image=out/(name+'_frames.jpg')
 if not image.exists():subprocess.run(['/usr/bin/ffmpeg','-y','-loglevel','error','-i',str(out/'evaluation'/(name+'.mp4')),'-vf','fps=1,scale=480:270,tile=4x3','-frames:v','1',str(image)],check=True)
print(json.dumps({'passing_cases':sum(all(g.values()) for g in gates.values()),'failed':{n:[k for k,v in g.items() if not v] for n,g in gates.items() if not all(g.values())}},indent=2))
if a.decision:
 assert a.reason and not (out/'decision.json').exists()
 final=all(all(g.values()) for g in gates.values()) and a.visual_pass
 decision={'decision':a.decision,'reason':a.reason,'visual_pass':a.visual_pass,'goal_met':final,'passing_cases':sum(all(g.values()) for g in gates.values())}
 (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 (out/'report.md').write_text('# '+out.name+'\n\n'+(out/'hypothesis.md').read_text()+'\n'+a.decision+': '+a.reason+'\n\nFull command gate passed: '+str(final)+'\n')
 if a.decision=='KEEP':(HOME/'CURRENT_CHAMPION.json').write_text(json.dumps({'loop':out.name,'checkpoint':str(out/'policy.pt'),'video':str(video),**decision},indent=2)+'\n')
 rows=[]
 for loop in sorted((HOME/'loops').glob('loop_*')):
  if (loop/'decision.json').exists():rows.append({'loop':loop.name,**json.loads((loop/'decision.json').read_text())})
 with (HOME/'LOOP_INDEX.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 manifest={str(f.relative_to(out)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='MANIFEST.json'}
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
