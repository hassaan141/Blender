from pathlib import Path
import subprocess,os,json,sys,argparse,shutil
R=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--reference',required=True);p.add_argument("--contact-fix",action="store_true");a=p.parse_args();out=R/'docs/natural_walk'/a.run
assert out.resolve().is_relative_to((R/'docs/natural_walk').resolve());assert not out.exists();assert len(list((R/'docs/natural_walk').glob('attempt_*/train_command.json')))<3
ref=R/a.reference;assert (ref.parent/'visual_gate.json').exists(),'Kinematic/physics visual gate required before training'
gate=json.loads((ref.parent/'visual_gate.json').read_text());assert gate['allow_training']
out.mkdir();cmd=['/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python',str(R/'rl/bingo_rl/bingo_rl/natural_walk/train.py'),'--task','Bingo-NaturalWalk-Train-v0','--headless','--num_envs','512','--seed','42','--max_iterations','100','--checkpoint',str(R/'docs/walk_ref/quality_runs/run_05/policy.pt'),'agent.agent.experiment.directory='+str(out/'training'),'hydra.run.dir='+str(out/'hydra'),'--kit_args','--/rtx/verifyDriverVersion/enabled=false --no-window']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONDONTWRITEBYTECODE':'1','BINGO_NATURAL_CONTACT_FIX':'1' if a.contact_fix else '0','BINGO_NATURAL_REFERENCE':str(ref)};(out/'train_command.json').write_text(json.dumps({'argv':cmd,'reference':str(ref),'contact_fix':a.contact_fix,'seed':42,'max_iterations':100,'parent':'Locomotion 1 quality Run 05'},indent=2))
with (out/'train.log').open('w') as f:ret=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
if ret.returncode:sys.exit(ret.returncode)
cp=list((out/'training').glob('*/checkpoints/agent_2400.pt'));assert len(cp)==1,cp;shutil.copy2(cp[0],out/'policy.pt');print('Saved',out/'policy.pt',flush=True)
