import unittest,json,pathlib,copy
from selection import decide,score
R=pathlib.Path(__file__).resolve().parent
class SelectionTests(unittest.TestCase):
 def setUp(self):
  self.b=json.loads((R/'baseline/metrics.json').read_text());self.q=json.loads((R/'baseline/quality_metrics.json').read_text());self.m=copy.deepcopy(self.b);self.m['joint_accel_rms']-=.2
  for j in ['fl_SP_J','fl_knee']:self.m['torque_saturation_pct_per_joint'][j]-=1
  self.c=copy.deepcopy(self.q);self.c['body_bobbing_detrended_rms_m']*=.95;self.c['body_vertical_velocity_rms_m_s']*=.95
 def check(self):return decide(self.m,self.c,self.b,self.q,self.b,self.q)
 def test_quality_improvement(self):self.assertEqual(self.check(),[])
 def test_speed_is_constraint_not_score(self):
  before=score(self.m,self.c,self.b,self.q);self.m['vx_mean']=.225;self.assertEqual(before,score(self.m,self.c,self.b,self.q));self.assertEqual(self.check(),[])
 def test_speed_outside_rejected(self):self.m['vx_mean']=.24;self.assertIn('mean speed outside constraint',self.check())
 def test_fixed_acceleration_baseline(self):self.m['joint_accel_rms']=self.b['joint_accel_rms']+.01;self.assertIn('joint acceleration not below retained baseline',self.check())
 def test_bobbing_guard(self):self.c['body_bobbing_detrended_rms_m']=self.q['body_bobbing_detrended_rms_m']*1.01;self.assertTrue(self.check())
 def test_cadence_guard(self):self.c['phase_cycle_period_s']+=.1;self.assertIn('cadence changed',self.check())
 def test_warmup_survival(self):self.c['warmup_falls']=1;self.assertIn('survival failure',self.check())
 def test_no_saturation_transfer_between_front_left_joints(self):self.m['torque_saturation_pct_per_joint']['fl_knee']=self.b['torque_saturation_pct_per_joint']['fl_knee']+.1;self.assertTrue(self.check())
if __name__=='__main__':unittest.main()
