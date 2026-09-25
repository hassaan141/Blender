"""Bounded file bridge to the active autonomous agent; no human input is requested."""
import argparse,time,shutil
from pathlib import Path
from core import HOME,read
p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','review']);p.add_argument('--context',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
loop=a.context.parent.name
source=a.context.parent.parent.parent/'agent_inbox'/f'{loop}_{a.mode}.json'
print(f'AGENT_{a.mode.upper()}_READY {source}',flush=True)
limit=time.monotonic()+1800
while not source.exists():
 if time.monotonic()>limit:raise RuntimeError('Active agent response timeout; no experiment is silently repeated')
 time.sleep(1)
read(source);shutil.copy2(source,a.output)
