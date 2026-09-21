"""Apply one explicitly declared hypothesis to the retained source snapshot."""
import json,pathlib,re,sys,subprocess
R=pathlib.Path(__file__).resolve().parent;ROOT=R.parents[2];SRC=ROOT/'rl/bingo_rl/bingo_rl/walk_ref'
name,parent,hypothesis=sys.argv[1:4];after=json.loads(sys.argv[4])
assert name in [f'run_{i:02d}' for i in range(1,9)] and 1<=len(after)<=2
p=R/name;p.mkdir(exist_ok=False)
for f in ['bingo_walk_ref_env.py','bingo_walk_ref_env_cfg.py']:(SRC/f).write_bytes((R/parent/f).read_bytes())
patterns={'ema_alpha':('bingo_walk_ref_env_cfg.py',r'(RESIDUAL_EMA_ALPHA = )([0-9.]+)'), 'residual_scale':('bingo_walk_ref_env_cfg.py',r'(    residual_scale = )([0-9.]+)'), 'phase_rate_multiplier':('bingo_walk_ref_env.py',r'(speed_ratio = )([0-9.]+)( \* self._cmd_vx / self._nominal_vx)'), 'action_rate_penalty':('bingo_walk_ref_env.py',r'(- )([0-9.]+)( \* act_rate_pen)'), 'joint_acceleration_penalty':('bingo_walk_ref_env.py',r'(- )([0-9.]+)( \* accel_pen)')}
changes={}
for key,val in after.items():
 if key=='reference_timing_strength':
  parent_spec=R/parent/'experiment.json'
  before=0.0
  if parent_spec.exists():
   before=json.loads(parent_spec.read_text()).get('effective_reference_timing_strength',0.0)
  subprocess.run(['/pub0/muhammadf/miniconda3/envs/isaaclab/bin/python',str(R/'retime_reference.py'),'--source',str(ROOT/'motions/bingo_walk_v4_upright_grounded_loopsmooth.npz'),'--out',str(p/'reference.npz'),'--strength',str(val)],check=True)
  f=SRC/'bingo_walk_ref_env_cfg.py';s=f.read_text();s=re.sub(r'^WALK_V4 = .*$',f'WALK_V4 = str(BLENDER_ROOT / "docs/walk_ref/refinement_runs/{name}/reference.npz")',s,flags=re.M);f.write_text(s);changes[key]={'before':before,'after':val};continue
 filename,pattern=patterns[key];f=SRC/filename;s=f.read_text();matches=list(re.finditer(pattern,s));assert len(matches)==1
 match=matches[0];before=float(match.group(2));assert before!=val
 s=s[:match.start(2)]+str(val)+s[match.end(2):];f.write_text(s);changes[key]={'before':before,'after':val}
m=json.loads((R/parent/'metrics.json').read_text())
e={'parent_run':parent,'parent_checkpoint':m['checkpoint'],'hypothesis':hypothesis,'changes':changes,'train_overrides':['env.vx_range=[0.2,0.3]','env.stand_prob=0.0'],'training':{'iterations':100,'num_envs':512,'seed':42},'status':'training'}
e['effective_reference_timing_strength']=after.get('reference_timing_strength',json.loads((R/parent/'experiment.json').read_text()).get('effective_reference_timing_strength',0.0) if (R/parent/'experiment.json').exists() else 0.0)
(p/'experiment.json').write_text(json.dumps(e,indent=2))
