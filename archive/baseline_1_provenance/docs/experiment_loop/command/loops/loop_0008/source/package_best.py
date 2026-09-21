"""Copy a reviewed experiment into a self-contained reference/runtime bundle.

The repository's validated Isaac environment remains a dependency. Teacher
policies are neither copied nor loaded. Packaging never edits sealed runs.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HOME=ROOT/'docs/experiment_loop/command'
p=argparse.ArgumentParser();p.add_argument('--loop',type=int,required=True);a=p.parse_args()
run=HOME/'loops'/f'loop_{a.loop:04d}';out=HOME/'best'
assert (run/'MANIFEST.json').exists(), 'Review and seal the run first'
assert not out.exists(), 'Never overwrite an existing delivery bundle'
out.mkdir()
shutil.copy2(run/'policy.pt',out/'policy.pt')
shutil.copytree(run/'references',out/'references')
refs={'forward':ROOT/'docs/natural_walk/reference_02b/reference.npz',
      'backward':ROOT/'docs/experiment_loop/loops/loop_0007/A/reference.npz',
      'turning':ROOT/'docs/experiment_loop/turning/loops/loop_0005/A/reference.npz'}
for name,src in refs.items():
 dest=out/'references'/name;dest.mkdir();shutil.copy2(src,dest/'reference.npz')
shutil.copy2(refs['turning'].with_name('turn_bank.npz'),out/'references/turning/turn_bank.npz')
for name in ['common.py','env.py','runtime.py','play.py']:
 shutil.copy2(run/'source'/name,out/name)
# References are now bundle-local. Teacher checkpoints are not used by inference.
common=out/'common.py';text=common.read_text();start=text.index('TEACHERS=');end=text.index('\nCASES=',start)
text=text[:start]+"TEACHERS={name:(None,Path(__file__).parent/'references'/name/'reference.npz') for name in ['forward','backward','turning']}"+text[end:];common.write_text(text)
shutil.copy2(run/'evaluation/metrics_extended.json',out/'metrics.json')
shutil.copy2(run/'decision.json',out/'decision.json')
links={name:str((run/'evaluation'/f'{name}.mp4').relative_to(ROOT)) for name in ['stand','forward','backward','left','right','forward_left','forward_right','backward_left','backward_right','transitions']}
links['all_commands']=str((run/'commands.mp4').relative_to(ROOT));links['teacher_comparison']=str((run/'teacher_comparison.mp4').relative_to(ROOT))
(out/'videos.json').write_text(json.dumps(links,indent=2)+'\n')
command=f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
exec /pub0/muhammadf/miniconda3/envs/isaaclab/bin/python docs/experiment_loop/command/best/play.py --checkpoint docs/experiment_loop/command/best/policy.pt --references docs/experiment_loop/command/best/references --headless --device cuda:0 --kit_args '--/rtx/verifyDriverVersion/enabled=false --no-window' "$@"
'''
(out/'reproduce.sh').write_text(command);(out/'reproduce.sh').chmod(0o755)
(out/'provenance.json').write_text(json.dumps({'source_loop':str(run.relative_to(ROOT)),'teacher_policies_required':False,'repository_runtime_required':True,'checkpoint_sha256':hashlib.file_digest((out/'policy.pt').open('rb'),'sha256').hexdigest(),'observation_width':95,'action_width':12,'command_units':['m/s','rad/s'],'command_range':[[-.2,.3],[-.6,.6]],'ema_alpha':.3,'residual_scale':.3,'command_alpha':.1},indent=2)+'\n')
print(out)
