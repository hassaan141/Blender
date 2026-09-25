"""No-training regression checks, including promotion and interrupted-job recovery."""
import copy, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from core import *
from adapter import command

class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HOME)
        self.h=Path(self.temp.name);self.cfg=read(HOME/'infrastructure_config.json')
        self.cfg['protected']=[str(self.h/'protected.txt')]
        self.cfg['protected_manifest_sources']=[]
        (self.h/'protected.txt').write_text('original')
        initialize(self.h);self.r=Runner(self.h,self.cfg)
        self.m=read(ROOT/'docs/natural_walk/attempt_03/evaluation/metrics.json');self.m['vx_mean']=.21
        self.base=read(self.cfg['fixed_baseline_metrics'])
    def tearDown(self):
        for p in self.h.rglob('*'):
            if not p.is_symlink():p.chmod(0o755 if p.is_dir() else 0o644)
        self.temp.cleanup()
    def plan(self):
        return {'defect':'drag','evidence':'measured','hypothesis':'smoothing helps','expected_effect':'less jitter',
                'arms':[{'name':'A','changes':{'residual_ema_alpha':.28}}]}
    def pending(self):
        d=self.h/'loops/loop_0001';d.mkdir(parents=True)
        parent=read(self.h/'CURRENT_CHAMPION.json');save(d/'parent.json',parent)
        save(d/'protected_before.json',self.r.protected());save(d/'hypothesis.json',self.plan())
        save(d/'decision.json',{'decision':'PENDING_VISUAL','selected':'A'})
        a=d/'A';a.mkdir();(a/'evaluation').mkdir()
        save(a/'config.json',{**self.cfg,'settings':DEFAULTS,'training_parent':parent})
        save(a/'evaluation/metrics.json',self.m)
        for f in ['policy.pt','reference.npz','evaluation/eval.mp4','comparison.mp4']:(a/f).write_bytes(b'SYNTHETIC TEST ONLY')
        (d/'report.md').write_text('Synthetic test only');seal(d);return d
    def test_regression_cap_fixed(self):
        self.m['joint_accel_rms']=self.base['joint_accel_rms']*1.20001
        self.assertIn('fixed-baseline acceleration cap',metric_gate(self.m,self.base,self.cfg))
        self.m['joint_accel_rms']=self.base['joint_accel_rms']*1.2
        self.assertEqual([],metric_gate(self.m,self.base,self.cfg))
    def test_bad_metrics_fail_closed(self):
        for key in ['survival','vx_std','vx_mean','joint_accel_rms']:
            m=copy.deepcopy(self.m);m[key]=float('nan');self.assertTrue(metric_gate(m,self.base,self.cfg))
        m=copy.deepcopy(self.m);m['duration_s']=1;self.assertTrue(metric_gate(m,self.base,self.cfg))
    def test_speed_not_ranked(self):
        m=copy.deepcopy(self.m);m['vx_mean']=.29;self.assertEqual(rank(m),rank(self.m))
    def test_plan_bounds_and_physics(self):
        for changes in [{'Kp':30},{'residual_scale':float('nan')},{'residual_ema_alpha':.001}]:
            p=self.plan();p['arms'][0]['changes']=changes
            with self.assertRaises(ValueError):validate_plan(p)
    def test_ab_same_hypothesis_required(self):
        p=self.plan();p['arms'].append({'name':'B','changes':{'residual_ema_alpha':.26}})
        with self.assertRaises(ValueError):validate_plan(p)
        p['ab_rationale']='two credible smoothing strengths';validate_plan(p)
    def test_pending_does_not_promote(self):
        old=read(self.h/'CURRENT_CHAMPION.json');self.pending();self.assertEqual(old,read(self.h/'CURRENT_CHAMPION.json'))
    def test_human_keep_and_immutable_review(self):
        d=self.pending();self.r.review(d,'KEEP','A','Reviewed full comparison: softer planting')
        self.assertEqual(read(self.h/'CURRENT_CHAMPION.json')['id'],'loop_0001/A');verify_seal(d)
        with self.assertRaises(ValueError):self.r.review(d,'KEEP','A','again')
    def test_visual_rejection_preserves_champion(self):
        d=self.pending();old=read(self.h/'CURRENT_CHAMPION.json');self.r.review(d,'REJECT','A','Looks worse')
        self.assertEqual(old,read(self.h/'CURRENT_CHAMPION.json'))
    def test_stale_review(self):
        d=self.pending();c=read(self.h/'CURRENT_CHAMPION.json');c['id']='new';save(self.h/'CURRENT_CHAMPION.json',c)
        with self.assertRaises(ValueError):self.r.review(d,'KEEP','A','good')
    def test_protected_tamper_blocks_promotion(self):
        d=self.pending();(self.h/'protected.txt').write_text('changed')
        with self.assertRaises(RuntimeError):self.r.review(d,'KEEP','A','good')
    def test_artifact_tamper_blocks_promotion(self):
        d=self.pending();p=d/'A/policy.pt';p.chmod(0o644);p.write_bytes(b'tampered')
        with self.assertRaises(RuntimeError):self.r.review(d,'KEEP','A','good')
    def test_failed_loop_cannot_promote(self):
        d=self.h/'loops/loop_0001';d.mkdir(parents=True);save(d/'decision.json',{'decision':'FAILED'});seal(d)
        with self.assertRaises(ValueError):self.r.review(d,'KEEP','A','good')
    def test_interrupted_run_not_retrained(self):
        d=self.h/'loops/loop_0001';d.mkdir(parents=True);save(d/'protected_before.json',self.r.protected())
        (d/'partial.log').write_text('partial');self.r.recover()
        self.assertEqual(read(d/'decision.json')['decision'],'FAILED');verify_seal(d)
    def test_promotion_journal_recovery(self):
        d=self.pending();old=read(self.h/'CURRENT_CHAMPION.json');self.r.review(d,'KEEP','A','good')
        new=read(self.h/'CURRENT_CHAMPION.json');save(self.h/'CURRENT_CHAMPION.json',old);self.r.recover()
        self.assertEqual(new,read(self.h/'CURRENT_CHAMPION.json'))
    def test_timeout_terminates(self):
        with self.assertRaises(subprocess.TimeoutExpired):execute([sys.executable,'-c','import time; time.sleep(30)'],self.h,self.h/'timeout.log',.1)
    def test_nonzero_exit(self):
        with self.assertRaises(RuntimeError):execute([sys.executable,'-c','raise SystemExit(3)'],self.h,self.h/'fail.log',5)
    def test_lock(self):
        with locked(self.h/'lock'):
            with self.assertRaises(RuntimeError):
                with locked(self.h/'lock'):pass
    def test_real_command_wiring_without_execution(self):
        c={**self.cfg,'arm_dir':str(self.h/'arm'),'training_parent':self.cfg['candidate']}
        train=command('train',c);ev=command('evaluate',c)
        self.assertIn(self.cfg['candidate']['checkpoint'],train)
        self.assertIn(str(self.h/'arm/policy.pt'),ev)
        self.assertIn('agent.agent.experiment.directory='+str(self.h/'arm/training'),train)

if __name__=='__main__':unittest.main(verbosity=2)
