#!/usr/bin/env python3
"""Compare two evaluation bundles and render compact command/comparison videos."""
import argparse,csv,json,subprocess
from pathlib import Path

FIELDS=['survival','survival_seconds','vx_mean','vx_std','vx_max_abs','yaw_rate_mean','yaw_error_mean_abs','slip_mean_mps','joint_acceleration_rms','action_rate_rms','torque_saturation_pct','joint_limit_violations']
NAMES=['stand','forward','backward','left','right','forward_left','forward_right','backward_left','backward_right']
def ffmpeg(args):
 subprocess.run(['/usr/bin/ffmpeg','-y','-loglevel','error',*args],check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 b=Path(a.baseline);c=Path(a.candidate);o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
 bm=json.loads((b/'metrics.json').read_text());cm=json.loads((c/'metrics.json').read_text())
 rows=[]
 with (o/'comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['command','vx_cmd','yaw_cmd']+[f'baseline_{k}' for k in FIELDS]+[f'candidate_{k}' for k in FIELDS])
  for n in NAMES:
   bv=bm[n];cv=cm[n];row=[n,*bv['command'],*[bv[k] for k in FIELDS],*[cv[k] for k in FIELDS]];w.writerow(row);rows.append(row)
 all_survive=all(cm[n]['survival'] for n in NAMES);stand_ok=cm['stand']['survival'] and cm['stand']['vx_max_abs']<.05
 btrack=sum(bm[n]['yaw_error_mean_abs']+abs(bm[n]['vx_mean']-bm[n]['command'][0]) for n in NAMES[1:])/8
 ctrack=sum(cm[n]['yaw_error_mean_abs']+abs(cm[n]['vx_mean']-cm[n]['command'][0]) for n in NAMES[1:])/8
 keep=all_survive and stand_ok and ctrack < btrack
 (o/'decision.md').write_text(f"# Sim-to-sim attempt decision\n\nDecision: **{'KEEP' if keep else 'REJECT'}**\n\nAll-nine survival: {all_survive}. Stand stable: {stand_ok}. Mean normalized command-tracking error (baseline/candidate): {btrack:.4f} / {ctrack:.4f}. Candidate survival is mandatory; a tracking gain cannot offset a fall.\n\nSee `comparison.csv` for per-command survival, speed, yaw, slip, acceleration, action-rate, saturation and joint-limit metrics.\n")
 # Concatenate all candidate command clips into one review video, holding the
 # final frame when a falling run ends before the 9-second evaluation window.
 ins=[];fc=[]
 for i,n in enumerate(NAMES):
  ins += ['-i',str(c/f'{n}.mp4')];fc.append(f'[{i}:v]tpad=stop_mode=clone:stop_duration=10,trim=duration=9,setpts=PTS-STARTPTS[v{i}]')
 fc.append(''.join(f'[v{i}]' for i in range(len(NAMES)))+f'concat=n={len(NAMES)}:v=1:a=0[out]')
 ffmpeg([*ins,'-filter_complex',';'.join(fc),'-map','[out]','-an','-c:v','libx264','-preset','veryfast','-crf','24','-pix_fmt','yuv420p',str(o/'all_nine_commands.mp4')])
 # Three-by-three table of side-by-side baseline and candidate clips.
 ins=[];fc=[]
 for i,n in enumerate(NAMES):
  ins += ['-i',str(b/f'{n}.mp4'),'-i',str(c/f'{n}.mp4')]
  fc += [f'[{2*i}:v]scale=320:240,setsar=1,tpad=stop_mode=clone:stop_duration=10,trim=duration=9,setpts=PTS-STARTPTS[b{i}]',
         f'[{2*i+1}:v]scale=320:240,setsar=1,tpad=stop_mode=clone:stop_duration=10,trim=duration=9,setpts=PTS-STARTPTS[c{i}]',
         f'[b{i}][c{i}]hstack=inputs=2[p{i}]']
 layout='|'.join(f'{(i%3)*640}_{(i//3)*240}' for i in range(9))
 fc.append(''.join(f'[p{i}]' for i in range(9))+f'xstack=inputs=9:layout={layout}:fill=0x202020[out]')
 ffmpeg([*ins,'-filter_complex',';'.join(fc),'-map','[out]','-an','-c:v','libx264','-preset','veryfast','-crf','25','-pix_fmt','yuv420p',str(o/'baseline_vs_candidate_montage.mp4')])
 print(f"decision={'KEEP' if keep else 'REJECT'} baseline_track={btrack:.4f} candidate_track={ctrack:.4f}")
if __name__=='__main__':main()
