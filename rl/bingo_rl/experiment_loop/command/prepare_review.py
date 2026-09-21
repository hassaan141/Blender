"""Build review artifacts without deciding or promoting a candidate."""
import argparse,difflib,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
CODE=Path(__file__).parent
HOME=ROOT/'docs/experiment_loop/command'
p=argparse.ArgumentParser()
p.add_argument('--loop',type=int,required=True)
p.add_argument('--champion',type=int,required=True)
a=p.parse_args()
out=HOME/'loops'/f'loop_{a.loop:04d}'
old=HOME/'loops'/f'loop_{a.champion:04d}'
assert not (out/'MANIFEST.json').exists(), 'Sealed experiments are read-only'
def run(cmd):subprocess.run([str(x) for x in cmd],check=True,cwd=ROOT)
run(['/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python',CODE/'analyze.py',out/'evaluation'])
run(['python3',CODE/'review.py','--loop',a.loop])
run(['python3',CODE/'compare.py',out])
run(['/usr/bin/ffmpeg','-y','-loglevel','error','-filter_complex_threads','1',
     '-i',old/'commands.mp4','-i',out/'commands.mp4',
     '-filter_complex',f"[0:v]drawtext=text='Champion {a.champion}':fontsize=20:fontcolor=yellow:x=5:y=30[a];[1:v]drawtext=text='Candidate {a.loop}':fontsize=20:fontcolor=yellow:x=5:y=30[b];[a][b]hstack[v]",
     '-map','[v]','-c:v','libx264','-threads','2','-crf','21',out/'comparison.mp4'])
for name in ['left','right','backward_left','backward_right']:
 run(['/usr/bin/ffmpeg','-y','-loglevel','error','-i',out/'evaluation'/f'{name}.mp4',
      '-vf','fps=4,scale=480:270,tile=4x4','-frames:v','1',out/f'{name}_detail.jpg'])
diff=[]
for f in sorted((out/'source').glob('*.py')):
 previous=old/'source'/f.name
 diff.extend(difflib.unified_diff(previous.read_text().splitlines(True) if previous.exists() else [],f.read_text().splitlines(True),fromfile=str(previous),tofile=str(f)))
(out/'changes.diff').write_text(''.join(diff))
print('Artifacts ready; inspect metrics and video before review.py --decision')
