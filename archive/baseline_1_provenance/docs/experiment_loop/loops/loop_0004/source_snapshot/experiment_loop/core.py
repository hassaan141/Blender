"""Finite experiment orchestration. Standard library only; never imports Isaac Lab."""
from __future__ import annotations
import argparse, csv, difflib, fcntl, hashlib, io, json, math, os, re, shutil
import signal, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOME = ROOT / 'docs/experiment_loop'
ALLOWED = {'residual_ema_alpha': (0.05, 1.0), 'residual_scale': (0.05, 0.4),
           'fl_torque_weight': (0., .1), 'body_height_weight': (0., .2),
           'vertical_velocity_weight': (0., .1),
           'backward_command_vx': (-.2, -.2), 'signed_reference_velocity_scale': (-1.4*.2/.66, -1.4*.2/.66)}
ALLOWED.update(lateral_scale=(.3,1.),front_stride_scale=(.8,1.5),rear_stride_scale=(.8,1.5),
               fl_lift_mm=(0.,12.),fr_lift_mm=(0.,12.),bl_lift_mm=(0.,12.),br_lift_mm=(0.,12.),phase_gain=(.8,1.5),
               all_contacts_corrected=(0,1),stance_shape_blend=(0.,.8))
DEFAULTS = dict(residual_ema_alpha=.3, residual_scale=.3, fl_torque_weight=.03,
                body_height_weight=.05, vertical_velocity_weight=.01)

def read(p):
    return json.loads(Path(p).read_text())

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def inventory(paths):
    """Follow historical compatibility symlinks, detect contents AND link targets."""
    out = {}; visited = set(); file_hashes = {}
    def walk(p):
        key = str(p.absolute())
        if p.is_symlink(): out[key + ':symlink'] = os.readlink(p)
        if not p.exists():
            out[key] = 'MISSING'; return
        if p.is_file():
            resolved=p.resolve()
            if resolved not in file_hashes:file_hashes[resolved]=digest(p)
            out[key]=file_hashes[resolved];return
        resolved = p.resolve()
        if resolved in visited: return
        visited.add(resolved)
        for child in sorted(p.iterdir()):
            if child.name not in {'__pycache__', '.git'}: walk(child)
    for p in paths: walk(Path(p))
    return out

@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another runner/job holds ' + str(path))
        yield f.fileno()

def execute(argv, cwd, log, timeout, env=None, lock_fd=None):
    if not isinstance(argv, list) or not argv or not all(isinstance(x,str) and x for x in argv):
        raise ValueError('Hook must be a nonempty argv array')
    if not math.isfinite(timeout) or not 0 < timeout <= 86400: raise ValueError('Invalid timeout')
    with Path(log).open('xb') as f:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            code = p.wait(timeout=timeout)
            if code: raise RuntimeError(f'Job exited {code}: {argv[0]}')
        except BaseException:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise
    return {'exit_code': code, 'argv': argv, 'completed_at': time.time()}

def validate_plan(p):
    for key in ('defect', 'evidence', 'hypothesis', 'expected_effect'):
        if not isinstance(p.get(key), str) or not p[key].strip(): raise ValueError('Missing '+key)
    arms = p.get('arms', [])
    if not 1 <= len(arms) <= 2: raise ValueError('One candidate or two hypothesis-driven alternatives only')
    if len(arms)==2 and not p.get('ab_rationale'): raise ValueError('A/B requires a rationale')
    names = set()
    for a in arms:
        name = a.get('name','')
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,30}',name) or name in names: raise ValueError('Invalid arm name')
        names.add(name)
        changes = a.get('changes',{})
        if not 1 <= len(changes) <= 2: raise ValueError('Change only 1–2 related parameters')
        for k,v in changes.items():
            if k not in ALLOWED or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not ALLOWED[k][0] <= v <= ALLOWED[k][1]:
                raise ValueError('Unsupported/out-of-bounds change: '+k)
    return p

def metric_gate(m, baseline, cfg):
    keys = ('survival','vx_mean','vx_std','vx_max','joint_accel_rms','residual_action_rate_rms')
    if any(not isinstance(m.get(k),(float,int)) or isinstance(m[k],bool) or not math.isfinite(m[k]) for k in keys):
        return ['Missing or nonfinite metrics']
    errors=[]
    if m['survival'] != 1 or m.get('any_fell') is not False: errors.append('survival')
    if not cfg['vx_min'] <= m['vx_mean'] <= cfg['vx_max_mean']: errors.append('mean vx')
    if not 0 <= m['vx_std'] < .06: errors.append('vx std')
    if cfg.get('backward',False):
        if not math.isfinite(m.get('vx_abs_max',float('nan'))) or m['vx_abs_max']>=.40:errors.append('max absolute vx')
        if m.get('command',{}).get('vx') != -.2:errors.append('wrong backward command')
    elif not m['vx_mean'] <= m['vx_max'] < .40: errors.append('max vx')
    if not 0 <= m['joint_accel_rms'] <= 1.2 * baseline['joint_accel_rms']: errors.append('fixed-baseline acceleration cap')
    if m['residual_action_rate_rms'] < 0: errors.append('action rate')
    if m.get('duration_s') != cfg['duration_s'] or m.get('steps_recorded') != round(cfg['duration_s']*24): errors.append('incomplete evaluation')
    torques=m.get('torque_saturation_pct_per_joint',{})
    if len(torques)!=12 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=100 for v in torques.values()): errors.append('torque metrics')
    return errors

def rank(m):
    # Speed is a gate, never a reward for going faster.
    return (m['vx_std'],m['joint_accel_rms'],m['residual_action_rate_rms'],
            sum(m['torque_saturation_pct_per_joint'].values())/12)

def seal(d):
    save(d/'MANIFEST.json', {str(p.relative_to(d)): digest(p) for p in sorted(d.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'})
    for p in d.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    for p in sorted((p for p in d.rglob('*') if p.is_dir()), reverse=True): p.chmod(0o555)
    d.chmod(0o555)

def verify_seal(d):
    expected=read(d/'MANIFEST.json')
    actual={str(p.relative_to(d)): digest(p) for p in d.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    if actual!=expected: raise RuntimeError('Immutable loop artifact mismatch: '+str(d))

def initial_champion():
    return {'id':'locomotion1_quality_run05', 'checkpoint':str(ROOT/'docs/walk_ref/quality_runs/run_05/policy.pt'),
            'reference':str(ROOT/'docs/walk_ref/refinement_runs/run_05/reference.npz'),
            'metrics':str(ROOT/'docs/walk_ref/quality_runs/run_05/metrics.json'),
            'video':str(ROOT/'docs/walk_ref/quality_runs/best_render/eval.mp4'),
            'report':str(ROOT/'docs/natural_walk/RESULTS.md'), 'contact_fix':False,
            'settings':DEFAULTS, 'status':'historical retained champion'}

def initialize(home):
    home.mkdir(parents=True,exist_ok=True)
    if not (home/'CURRENT_CHAMPION.json').exists():
        c=initial_champion(); c['hashes']={k:digest(c[k]) for k in ('checkpoint','reference','metrics','video')}
        save(home/'CURRENT_CHAMPION.json',c)
    index(home)

def index(home):
    rows=[]
    for d in sorted((home/'loops').glob('loop_*')):
        if not (d/'decision.json').exists(): continue
        decision=read(d/'decision.json')
        external=home/'reviews'/f'{d.name}.json'
        rows.append(dict(loop=d.name, decision=read(external)['decision'] if external.exists() else decision['decision'],
                         selected=decision.get('selected',''), report=str(d/'report.md')))
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=['loop','decision','selected','report']);w.writeheader();w.writerows(rows)
    p=home/'LOOP_INDEX.csv';t=p.with_suffix('.tmp');t.write_text(s.getvalue());t.replace(p)

class Runner:
    def __init__(self, home, cfg, smoke=False):
        self.home=home;self.cfg=cfg;self.smoke=smoke;self.lock_fd=None
    def protected(self):
        paths=[ROOT/p for p in self.cfg['protected']]
        for manifest in self.cfg.get('protected_manifest_sources',[]):paths.extend(ROOT/p for p in read(ROOT/manifest))
        return inventory(paths)
    def check_protected(self, expected):
        actual=self.protected()
        if actual!=expected: raise RuntimeError('Protected files changed; stop and investigate; champion NOT promoted')
        return actual
    def plan(self,d,champion):
        context={'champion':champion,'current_candidate':self.cfg['candidate'],
                 'previous_report':str(next(iter(sorted((self.home/'loops').glob('loop_*/report.md'),reverse=True)),Path(champion['report']))),
                 'review_history':[read(p) for p in sorted((self.home/'reviews').glob('loop_*.json'))],
                 'feedback':(self.home/'FEEDBACK.md').read_text() if (self.home/'FEEDBACK.md').exists() else '',
                 'allowed_changes':ALLOWED,'max_related_changes':2,'output':str(d/'proposal.json')}
        save(d/'context.json',context)
        if self.smoke:
            p={'defect':'SMOKE ONLY: replay existing artifacts','evidence':'No simulator or training invoked',
               'hypothesis':'Exercise bookkeeping, not locomotion','expected_effect':'No physical change',
               'arms':[{'name':'A','changes':{'residual_ema_alpha':.29}}]}
        elif self.cfg.get('planner_hook'):
            argv=[x.format(context=str(d/'context.json'),output=str(d/'proposal.json')) for x in self.cfg['planner_hook']]
            execute(argv,ROOT,d/'planner.log',self.cfg['planner_timeout_s'],lock_fd=self.lock_fd)
            p=read(d/'proposal.json')
        else:
            plans=self.cfg.get('plans',[]);i=int(d.name.split('_')[1])-1
            if i>=len(plans): raise ValueError('No proposal supplied. Configure plans or a planner_hook before training.')
            p=read(ROOT/plans[i])
        validate_plan(p)
        backward_keys={'backward_command_vx','signed_reference_velocity_scale'}
        if any(backward_keys.intersection(a['changes']) for a in p['arms']) and not self.cfg.get('backward',False):
            raise ValueError('Backward settings require the dedicated backward configuration/hooks')
        save(d/'hypothesis.json',p);self.check_protected(read(d/'protected_before.json'));return p
    def hooks(self, arm, parent):
        entry=str(ROOT/'rl/bingo_rl/experiment_loop/adapter.py')
        context=str(arm/'config.json')
        result={}
        for stage in ('train','evaluate','compare'):
            template=self.cfg.get('hooks',{}).get(stage,[sys.executable,entry,stage,'--context','{context}'])
            result[stage]=[x.format(context=context,arm=str(arm),root=str(ROOT)) for x in template]
        return result
    def arm(self,d,a,parent):
        arm=d/a['name'];arm.mkdir()
        training_parent = self.cfg['candidate'] if parent['id']=='locomotion1_quality_run05' and self.cfg.get('start_from_candidate',True) else parent
        settings={**training_parent['settings'],**a['changes']}
        config={**self.cfg,'parent':parent,'training_parent':training_parent,'settings':settings,'arm_dir':str(arm),'smoke':self.smoke}
        save(arm/'config.json',config)
        (arm/'changes.diff').write_text(''.join(difflib.unified_diff(json.dumps(training_parent['settings'],indent=2).splitlines(True),json.dumps(settings,indent=2).splitlines(True),fromfile='parent settings',tofile='candidate settings')))
        save(arm/'changed_files.json',{'source_files_changed':[], 'config_fields':list(a['changes']), 'diff':'changes.diff'})
        shutil.copyfile(training_parent['reference'],arm/'reference.npz')  # fresh editable candidate; never alter read-only parent
        hooks=self.hooks(arm,parent);save(arm/'commands.json',hooks)
        if self.smoke:
            # Reuse data explicitly labelled replay; never pretend these are fresh evaluation results.
            src=ROOT/'docs/natural_walk/attempt_03'
            shutil.copy2(src/'policy.pt',arm/'policy.pt')
            shutil.copytree(src/'evaluation',arm/'evaluation',ignore=shutil.ignore_patterns('*.npz','*.log','_video_tmp'))
            (arm/'REPLAY_ONLY.txt').write_text('Existing attempt_03 artifacts. Zero training, zero simulator steps. Settings not applied.\n')
            for s in ('train','evaluate'):save(arm/f'{s}.receipt.json',{'status':'SKIPPED_SMOKE_REPLAY','training_updates':0})
            from adapter import comparison
            comparison(arm,Path(parent['video']),seconds=2)
            save(arm/'compare.receipt.json',{'status':'REAL_FFMPEG_COMPARISON_OF_HISTORICAL_VIDEO','seconds':2})
        else:
            for s in ('train','evaluate','compare'):
                receipt=execute(hooks[s],ROOT,arm/f'{s}.log',self.cfg['timeouts'][s],
                                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},lock_fd=self.lock_fd)
                save(arm/f'{s}.receipt.json',receipt)
                self.check_protected(read(d/'protected_before.json'))
        for rel in ['policy.pt','reference.npz','evaluation/metrics.json','evaluation/quality_metrics.json',
                    'evaluation/natural_metrics.json','evaluation/eval.mp4','comparison.mp4']:
            if not (arm/rel).is_file() or (arm/rel).stat().st_size==0: raise RuntimeError('Missing artifact '+rel)
        m=read(arm/'evaluation/metrics.json')
        errors=metric_gate(m,read(self.cfg['fixed_baseline_metrics']),self.cfg)
        natural=read(arm/'evaluation/natural_metrics.json')
        limit=natural.get('joint_limit_violation_max_rad')
        if not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit>1e-5 or limit<0:errors.append('joint limit violation')
        if (arm/'backward_gates.json').exists():errors+=read(arm/'backward_gates.json')['failures']
        save(arm/'metric_comparison.json',{'candidate':m,'champion':read(parent['metrics']),'gate_failures':errors})
        return {'name':a['name'],'metrics':m,'failures':errors}
    def run_one(self):
        parent=read(self.home/self.cfg.get('champion_file','CURRENT_CHAMPION.json'))
        for k,h in parent['hashes'].items():
            if digest(parent[k])!=h: raise RuntimeError('Champion hash mismatch: '+k)
        n=len(list((self.home/'loops').glob('loop_*')))+1
        d=self.home/'loops'/f'loop_{n:04d}';d.mkdir(parents=True)
        save(d/'config_snapshot.json',self.cfg);save(d/'parent.json',parent)
        before=self.protected();save(d/'protected_before.json',before)
        # Snapshots include the actual environment, evaluator, wrapper and this orchestrator.
        snap=d/'source_snapshot';snap.mkdir()
        for folder in ['rl/bingo_rl/bingo_rl/natural_walk','rl/bingo_rl/bingo_rl/walk_ref','rl/bingo_rl/experiment_loop']:
            shutil.copytree(ROOT/folder,snap/Path(folder).name,ignore=shutil.ignore_patterns('__pycache__'))
        save(d/'state.json',{'status':'RUNNING','started':time.time()})
        outcomes=[];decision={'decision':'REJECT','reason':'incomplete'}
        try:
            proposal=self.plan(d,parent)
            for a in proposal['arms']:
                outcomes.append(self.arm(d,a,parent));self.check_protected(before)
            valid=[o for o in outcomes if not o['failures']]
            if self.cfg.get('agent_autonomous') and not self.smoke:
                save(d/'review_context.json',{'parent':parent,'outcomes':outcomes,'goal_eligible':bool(valid),'report_inputs':str(d)})
                cmd=[x.format(context=str(d/'review_context.json'),output=str(d/'agent_review.json')) for x in self.cfg['review_hook']]
                execute(cmd,ROOT,d/'review.log',1800,lock_fd=self.lock_fd)
                decision=read(d/'agent_review.json')
                if decision.get('decision') not in {'KEEP','REJECT'} or not decision.get('reason'):raise ValueError('Invalid agent review')
                if decision['decision']=='KEEP':
                    chosen=next(o for o in outcomes if o['name']==decision.get('selected','A'))
                    m=chosen['metrics'];limit=read(d/chosen['name']/'evaluation/natural_metrics.json')['joint_limit_violation_max_rad']
                    if m['survival']!=1 or m['joint_accel_rms']>1.2*read(self.cfg['fixed_baseline_metrics'])['joint_accel_rms'] or limit>1e-5:raise ValueError('Mandatory retention gate failed')
                if decision.get('goal_met') and (not valid or decision['decision']!='KEEP'):raise ValueError('Cannot claim success with failed goal gates')
                if decision.get('stop') and not decision.get('goal_met') and n<4:raise ValueError('Premature blocked stop')
            elif self.smoke: decision={'decision':'REJECT','reason':'Smoke replay is never promotable'}
            elif valid:
                chosen=min(valid,key=lambda o:rank(o['metrics']))
                decision={'decision':'PENDING_VISUAL','selected':chosen['name'],
                          'reason':'Metrics eligible; human must judge full comparison video'}
            else: decision={'decision':'REJECT','reason':'All candidates failed metrics gates'}
        except BaseException as e:
            decision={'decision':'FAILED','reason':f'{type(e).__name__}: {e}'}
        finally:
            after=self.protected();save(d/'protected_after.json',after)
            if after!=before: decision={'decision':'FAILED','reason':'PROTECTED FILE MISMATCH'}
            save(d/'decision.json',decision)
            save(d/'state.json',{'status':'FINISHED','ended':time.time()})
            report=f"# {d.name}\n\n{decision['decision']}: {decision['reason']}\n\nParent: {parent['id']}\n\n"
            if (d/'hypothesis.json').exists():
                p=read(d/'hypothesis.json');report+=f"Defect: {p['defect']}\n\nEvidence: {p['evidence']}\n\nHypothesis: {p['hypothesis']}\n\n"
            report+='| Arm | vx mean | vx std | acceleration | Gate failures | Video |\n|---|---:|---:|---:|---|---|\n'
            for o in outcomes:
                m=o['metrics'];name=o['name']
                report+=f"| {name} | {m['vx_mean']:.4f} | {m['vx_std']:.4f} | {m['joint_accel_rms']:.3f} | {', '.join(o['failures']) or 'none'} | [comparison]({name}/comparison.mp4) |\n"
            report+='\nFull metrics, source snapshots, config diffs, process receipts and protected-file hashes are in this directory.\n'
            if self.smoke: report+='\nSMOKE REPLAY ONLY: no training or simulation; the existing video/metrics are not new results.\n'
            (d/'report.md').write_text(report);seal(d);index(self.home)
        if decision['decision']=='KEEP' and self.cfg.get('agent_autonomous'):
            a=d/decision.get('selected','A');ac=read(a/'config.json')
            updated={**parent,'id':d.name+'/'+a.name,'settings':ac['settings'],'status':'qualified backward champion' if decision.get('goal_met') else 'provisional backward best; target not yet met',
                     'checkpoint':str(a/'policy.pt'),'reference':str(a/'reference.npz'),'metrics':str(a/'evaluation/metrics.json'),'video':str(a/'evaluation/eval.mp4'),'report':str(d/'report.md')}
            updated['hashes']={k:digest(updated[k]) for k in ('checkpoint','reference','metrics','video')}
            save(self.home/self.cfg['champion_file'],updated)
        if decision['decision']=='PENDING_VISUAL' and not self.cfg['require_visual_approval']:
            self.review(d,'KEEP',decision['selected'],'Explicit config disabled human visual approval')
        return decision
    def review(self,d,verdict,arm_name,note):
        verify_seal(d);decision=read(d/'decision.json')
        if decision['decision']!='PENDING_VISUAL': raise ValueError('Loop is not eligible for visual review')
        review=self.home/'reviews'/f'{d.name}.json'
        if review.exists(): raise ValueError('Review is immutable and already exists')
        parent=read(d/'parent.json');current=read(self.home/'CURRENT_CHAMPION.json')
        if current!=parent: raise ValueError('Stale parent: champion changed since experiment')
        self.check_protected(read(d/'protected_before.json'))
        if not note.strip(): raise ValueError('Visual review note required')
        update=None
        if verdict=='KEEP':
            names=[a['name'] for a in read(d/'hypothesis.json')['arms']]
            if arm_name not in names: raise ValueError('Unknown arm')
            a=d/arm_name;c=read(a/'config.json');m=read(a/'evaluation/metrics.json')
            errors=metric_gate(m,read(c['fixed_baseline_metrics']),c)
            if (a/'metric_comparison.json').exists():errors+=read(a/'metric_comparison.json')['gate_failures']
            if errors: raise ValueError('Cannot override failed metrics: '+str(errors))
            update={**parent,'id':d.name+'/'+arm_name,'checkpoint':str(a/'policy.pt'),'reference':str(a/'reference.npz'),
                    'metrics':str(a/'evaluation/metrics.json'),'video':str(a/'evaluation/eval.mp4'),
                    'report':str(d/'report.md'),'settings':c['settings'],'contact_fix':c['training_parent']['contact_fix'],'status':'promoted',
                    'hashes':{k:digest(a/rel) for k,rel in [('checkpoint','policy.pt'),('reference','reference.npz'),('metrics','evaluation/metrics.json'),('video','evaluation/eval.mp4')]}}
        # Journal first, atomic pointer second. Resume reconciles interrupted promotion.
        save(review,{'decision':verdict,'arm':arm_name,'note':note,'time':time.time(),
                     'manifest_sha256':digest(d/'MANIFEST.json'),'parent':parent,'champion_after':update})
        if update:save(self.home/'CURRENT_CHAMPION.json',update)
        index(self.home)
    def recover(self):
        for d in sorted((self.home/'loops').glob('loop_*')):
            if not (d/'MANIFEST.json').exists():
                # Never rerun an uncertain stage or overwrite partial results.
                before=read(d/'protected_before.json') if (d/'protected_before.json').exists() else read(self.home/'PROTECTED_BASELINE.json')
                if not (d/'protected_before.json').exists():save(d/'protected_before.json',before)
                after=self.protected();save(d/'protected_after.json',after)
                save(d/'decision.json',{'decision':'FAILED','reason':'Interrupted loop; preserved without automatic training retry',
                                       'protected_match':before==after})
                (d/'report.md').write_text('# Interrupted loop\n\nRejected. Partial artifacts preserved; no training retry.\n')
                seal(d)
            verify_seal(d)
            if read(d/'protected_before.json')!=self.protected(): raise RuntimeError('Protected mismatch during resume')
            r=self.home/'reviews'/f'{d.name}.json'
            if r.exists():
                rec=read(r)
                if digest(d/'MANIFEST.json')!=rec['manifest_sha256']: raise RuntimeError('Review artifact mismatch')
                current=read(self.home/'CURRENT_CHAMPION.json')
                if rec['champion_after'] and current==rec['parent']:save(self.home/'CURRENT_CHAMPION.json',rec['champion_after'])
        index(self.home)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    for flag in ['run-one','run-continuous','stop-after-current','smoke-test','init']:group.add_argument('--'+flag,action='store_true')
    group.add_argument('--review',metavar='LOOP_ID')
    p.add_argument('--max-loops',type=int,default=1);p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path,default=HOME/'config.json')
    p.add_argument('--decision',choices=['KEEP','REJECT']);p.add_argument('--arm',default='A');p.add_argument('--note',default='')
    a=p.parse_args();cfg=read(a.config)
    home=HOME
    if a.smoke_test:home=HOME/'smoke'/time.strftime('%Y%m%d_%H%M%S')
    if a.stop_after_current:
        HOME.mkdir(exist_ok=True);save(HOME/'STOP_AFTER_CURRENT.json',{'requested':time.time()});print('Stop requested at next loop boundary');return
    if not 1<=a.max_loops<=100: p.error('--max-loops must be 1..100 (total campaign budget)')
    with locked(HOME/'runner.lock') as fd:
        initialize(home);r=Runner(home,cfg,a.smoke_test);r.lock_fd=fd
        guard=home/'PROTECTED_BASELINE.json'
        if not guard.exists():save(guard,r.protected())
        r.check_protected(read(guard))
        if a.init:print('Initialized champion; no training');return
        if a.review:
            if not re.fullmatch('loop_[0-9]{4}',a.review) or not a.decision:p.error('Valid loop ID and --decision required')
            r.review(home/'loops'/a.review,a.decision,a.arm,a.note);print('Review recorded');return
        if a.resume:
            r.recover();(home/'STOP_AFTER_CURRENT.json').unlink(missing_ok=True)
        elif any(not (d/'MANIFEST.json').exists() for d in (home/'loops').glob('loop_*')):
            raise RuntimeError('Interrupted loop exists; use --resume')
        pending=[d for d in (home/'loops').glob('loop_*') if (d/'decision.json').exists() and read(d/'decision.json')['decision']=='PENDING_VISUAL' and not (home/'reviews'/f'{d.name}.json').exists()]
        if pending:print('Awaiting visual review:',pending[-1]);return
        budget=1 if a.smoke_test else a.max_loops
        while len(list((home/'loops').glob('loop_*')))<budget:
            if (home/'STOP_AFTER_CURRENT.json').exists():break
            result=r.run_one();print(json.dumps(result))
            if result['decision']=='FAILED':raise RuntimeError(result['reason'])
            if result.get('stop') or result.get('goal_met'):break
            if a.run_one or a.smoke_test or (result['decision']=='PENDING_VISUAL' and cfg['require_visual_approval']):break
        print('Artifacts:',home)

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
