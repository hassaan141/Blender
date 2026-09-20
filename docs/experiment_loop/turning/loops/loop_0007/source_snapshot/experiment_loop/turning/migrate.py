"""Append zero-weight yaw input, preserving pretrained zero-yaw action exactly."""
import sys,torch
from pathlib import Path
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if p['policy']['net_container.0.weight'].shape[1]==70:
 for model in ['policy','value']:
  w=p[model]['net_container.0.weight'];p[model]['net_container.0.weight']=torch.cat([w,torch.zeros_like(w[:,:1])],dim=1)
 for state in p.get('optimizer',{}).get('state',{}).values():
  for k,v in state.items():
   if hasattr(v,'shape') and tuple(v.shape)==(512,70):state[k]=torch.cat([v,torch.zeros_like(v[:,:1])],dim=1)
 pre=p['observation_preprocessor']
 for k,value in [('running_mean',0.),('running_variance',1.)]:pre[k]=torch.cat([pre[k],torch.tensor([value],dtype=pre[k].dtype)])
torch.save(p,sys.argv[2])
