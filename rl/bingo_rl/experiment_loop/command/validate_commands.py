"""Evaluate a packaged policy on nine explicit command cases, without training."""
import argparse,json,os,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a,extra=p.parse_known_args();bundle=a.bundle.resolve();sys.path.insert(0,str(bundle))
import common
common.CASES=json.loads(a.cases.read_text())
assert len(common.CASES)==9 and all(len(c)==3 for c in common.CASES)
os.environ['COMMAND_REFERENCE_DIR']=str(bundle/'references')
sys.argv=[str(Path(__file__).with_name('evaluate.py')),'--checkpoint',str(bundle/'policy.pt'),'--out',str(a.out),*extra]
runpy.run_path(sys.argv[0],run_name='__main__')
