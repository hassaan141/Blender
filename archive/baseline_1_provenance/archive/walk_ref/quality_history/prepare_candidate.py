import pathlib,json,re,sys
R=pathlib.Path(__file__).resolve().parent;ROOT=R.parents[2];SRC=ROOT/'rl/bingo_rl/bingo_rl/walk_ref'
name,parent,hypothesis=sys.argv[1:4];values=json.loads(sys.argv[4]);assert name in [f'run_{i:02d}' for i in range(1,7)] and 1<=len(values)<=2
p=R/name;p.mkdir(exist_ok=False)
for f in ['bingo_walk_ref_env.py','bingo_walk_ref_env_cfg.py']:(SRC/f).write_bytes((R/parent/f).read_bytes())
f=SRC/'bingo_walk_ref_env_cfg.py';s=f.read_text()
if 'fl_torque_weight =' not in s:s=s.replace('    residual_scale = 0.3','    fl_torque_weight = 0.0\n    vertical_velocity_weight = 0.0\n\n    residual_scale = 0.3')
if 'body_height_weight =' not in s:
 s=s.replace('    fl_torque_weight =', '    body_height_weight = 0.0\n    body_height_target = '+str(json.loads((R/'baseline/quality_metrics.json').read_text())['body_height_mean_m'])+'\n    fl_torque_weight =')
f.write_text(s)
f=SRC/'bingo_walk_ref_env.py';s=f.read_text()
if 'self.cfg.fl_torque_weight' not in s:
 s=s.replace('            - 0.04 * torque_pen','            - self.cfg.fl_torque_weight * torch.mean((torque[:, [1, 2]] / 3.0) ** 2, dim=1)\n            - self.cfg.vertical_velocity_weight * (root_lin_w[:, 2] / 0.1) ** 2\n            - 0.04 * torque_pen')
if 'self.cfg.body_height_weight' not in s:
 s=s.replace('            - 0.04 * torque_pen','            - self.cfg.body_height_weight * ((self.robot.data.body_pos_w[:, self.ref_body_index, 2] - self.scene.env_origins[:, 2] - self.cfg.body_height_target) / 0.01) ** 2\n            - 0.04 * torque_pen')
f.write_text(s)
patterns={'body_height_weight':('bingo_walk_ref_env_cfg.py',r'(    body_height_weight = )([0-9.]+)'), 'ema_alpha':('bingo_walk_ref_env_cfg.py',r'(RESIDUAL_EMA_ALPHA = )([0-9.]+)'), 'fl_torque_weight':('bingo_walk_ref_env_cfg.py',r'(    fl_torque_weight = )([0-9.]+)'), 'vertical_velocity_weight':('bingo_walk_ref_env_cfg.py',r'(    vertical_velocity_weight = )([0-9.]+)'), 'action_rate_penalty':('bingo_walk_ref_env.py',r'(- )([0-9.]+)( \* act_rate_pen)'), 'joint_acceleration_penalty':('bingo_walk_ref_env.py',r'(- )([0-9.]+)( \* accel_pen)')}
changes={}
for key,val in values.items():
 file,pattern=patterns[key];f=SRC/file;s=f.read_text();matches=list(re.finditer(pattern,s));assert len(matches)==1;mt=matches[0];before=float(mt.group(2));assert before!=val
 f.write_text(s[:mt.start(2)]+str(val)+s[mt.end(2):]);changes[key]={'before':before,'after':val}
m=json.loads((R/parent/'metrics.json').read_text())
e={'parent_run':parent,'parent_checkpoint':m['checkpoint'],'hypothesis':hypothesis,'changes':changes,'train_overrides':['env.vx_range=[0.2,0.3]','env.stand_prob=0.0'],'training':{'iterations':100,'num_envs':512,'seed':42},'status':'training'}
(p/'experiment.json').write_text(json.dumps(e,indent=2))
