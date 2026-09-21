"""Matched-time close-up video comparison, with truthful labels and no speed changes."""
from pathlib import Path
import argparse,subprocess
R=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();d=R/'docs/natural_walk'/a.attempt
base=R/'docs/walk_ref/quality_runs/best_render/eval.mp4';candidate=d/'evaluation/eval.mp4';out=d/'comparison.mp4'
# Same inherited camera and playback rate; crop identically for readable foot motion.
f="[0:v]crop=720:520:280:180,scale=720:520,drawtext=text='Locomotion 1 - preserved baseline':x=15:y=15:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[a];[1:v]crop=720:520:280:180,scale=720:520,drawtext=text='Natural walk "+a.attempt+" - candidate':x=15:y=15:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[b];[a][b]hstack=inputs=2[v]"
subprocess.run(['/usr/bin/ffmpeg','-v','error','-i',str(base),'-i',str(candidate),'-filter_complex',f,'-map','[v]','-t','60','-c:v','libx264','-crf','20','-y',str(out)],check=True)
subprocess.run(['/usr/bin/ffmpeg','-v','error','-ss','8','-i',str(out),'-vf','fps=8,scale=960:-1,tile=2x6','-frames:v','1','-y',str(d/'comparison_stride.jpg')],check=True)
subprocess.run(['/usr/bin/ffmpeg','-v','error','-i',str(out),'-vf','fps=1/5,scale=960:-1,tile=2x6','-frames:v','1','-y',str(d/'comparison_overview.jpg')],check=True)
print(out)
