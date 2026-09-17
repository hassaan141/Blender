import copy,json,pathlib,sys,unittest
OLD=pathlib.Path(__file__).resolve().parents[1]/'loop_runs'
sys.path.insert(0,str(OLD))
from selection_rules import protocol_rejections,old_selection,refinement_selection

def load(run):return json.loads((OLD/run/'metrics.json').read_text())
class SelectionTests(unittest.TestCase):
 def setUp(self):self.base=load('baseline');self.parent=load('run_06')
 def test_all_historical_acceleration_rejections(self):
  for run in ['run_02','run_03','run_04','run_05','run_07','run_08']:
   self.assertIn('acceleration exceeds 120% of original baseline',protocol_rejections(load(run),self.base))
 def test_boundary_fixed_to_original_not_parent(self):
  candidate=copy.deepcopy(self.parent);candidate['joint_accel_rms']=1.2*self.base['joint_accel_rms']
  self.assertNotIn('acceleration exceeds 120% of original baseline',protocol_rejections(candidate,self.base))
  candidate['joint_accel_rms']+=1e-8
  self.assertIn('acceleration exceeds 120% of original baseline',protocol_rejections(candidate,self.base))
  self.assertIn('acceleration exceeds 120% of original baseline',old_selection(load('run_05'),load('run_03'),self.base))
 def test_run06_beats_last_valid_run01(self):self.assertEqual(old_selection(self.parent,load('run_01'),self.base),[])
 def test_survival_mandatory(self):
  m=copy.deepcopy(self.parent);m['survival']=.999;m['any_fell']=True
  self.assertIn('survival must be 100%',refinement_selection(m,self.parent,self.base))
 def test_faster_shakier_is_rejected_below_absolute_cap(self):
  m=copy.deepcopy(self.parent);m['vx_mean']=.205;m['joint_accel_rms']+=.1
  self.assertIn('faster but shakier',refinement_selection(m,self.parent,self.base))
 def test_no_reward_for_speed_in_band(self):
  p=copy.deepcopy(self.parent);p['vx_mean']=.21;p['vx_std']=.055
  m=copy.deepcopy(p);m['vx_mean']=.29
  self.assertTrue(refinement_selection(m,p,self.base))
 def test_after_speed_gates_lower_acceleration_wins(self):
  p=copy.deepcopy(self.parent);p['vx_mean']=.21;p['vx_std']=.05
  m=copy.deepcopy(p);m['vx_std']=.055;m['joint_accel_rms']-=1
  self.assertEqual(refinement_selection(m,p,self.base),[])
 def test_peak_at_limit_rejected(self):
  m=copy.deepcopy(self.parent);m['vx_max']=.4
  self.assertIn('peak must be below 0.40 m/s',refinement_selection(m,self.parent,self.base))
if __name__=='__main__':unittest.main()
