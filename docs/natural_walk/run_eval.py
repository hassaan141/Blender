from pathlib import Path
import subprocess,os,sys,json,argparse
R=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--reference',default='docs/natural_walk/reference_01/reference.npz');p.add_argument('--checkpoint',default='docs/walk_ref/quality_runs/run_05/policy.pt');p.add_argument('--seconds',type=float,default=60);p.add_argument('--no-video',action='store_true');p.add_argument('--pd-only',action='store_true');p.add_argument("--contact-fix",action="store_true");a=p.parse_args();out=R/'docs/natural_walk'/a.out;assert out.resolve().is_relative_to((R/'docs/natural_walk').resolve()),'Output must stay inside docs/natural_walk';out.mkdir(parents=True,exist_ok=False)
cmd=['/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python',str(R/'rl/bingo_rl/bingo_rl/natural_walk/evaluate.py'),'--headless','--diagnostics','--checkpoint',str(R/a.checkpoint),'--out_dir',str(out),'--duration_s',str(a.seconds),'--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window']
if a.no_video:cmd+=['--no_video']
if a.pd_only:cmd+=['--pd_only']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONDONTWRITEBYTECODE':'1','BINGO_NATURAL_CONTACT_FIX':'1' if a.contact_fix else '0','BINGO_NATURAL_REFERENCE':str(R/a.reference)};(out/'command.json').write_text(json.dumps({'argv':cmd,'reference':env['BINGO_NATURAL_REFERENCE'],'pd_only':a.pd_only,'contact_fix':a.contact_fix},indent=2))
with (out/'eval.log').open('w') as f:ret=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
print(out,'exit',ret.returncode,flush=True);sys.exit(ret.returncode)
