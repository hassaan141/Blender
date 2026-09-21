"""Pixel-only teacher/student comparison; never edits or replays teacher artifacts."""
import argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser();p.add_argument('loop',type=Path);a=p.parse_args()
files=[ROOT/'docs/natural_walk/attempt_03/evaluation/eval.mp4',ROOT/'docs/experiment_loop/loops/loop_0007/A/evaluation/eval.mp4',ROOT/'docs/experiment_loop/turning/loops/loop_0005/A/left/eval.mp4',ROOT/'docs/experiment_loop/turning/loops/loop_0005/A/right/eval.mp4']
names=['forward','backward','forward_left','forward_right'];files += [a.loop/'evaluation'/(n+'.mp4') for n in names]
cmd=['/usr/bin/ffmpeg','-y','-loglevel','error','-filter_complex_threads','1'];filters=[]
for i,f in enumerate(files):
 cmd+=['-i',str(f)];label=('TEACHER ' if i<4 else 'STUDENT ')+names[i%4]
 filters.append(f"[{i}:v]scale=400:300:force_original_aspect_ratio=decrease,pad=400:300:(ow-iw)/2:(oh-ih)/2,setsar=1,drawtext=text='{label}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:x=8:y=8[v{i}]")
layout='|'.join(f'{i%4*400}_{i//4*300}' for i in range(8));filters.append(''.join(f'[v{i}]' for i in range(8))+f'xstack=inputs=8:layout={layout}[v]')
subprocess.run(cmd+['-filter_complex',';'.join(filters),'-map','[v]','-t','12','-r','24','-c:v','libx264','-threads','2','-crf','21','-pix_fmt','yuv420p',str(a.loop/'teacher_comparison.mp4')],check=True)
