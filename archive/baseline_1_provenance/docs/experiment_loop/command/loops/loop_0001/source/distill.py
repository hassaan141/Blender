"""Supervised native-controller distillation into a fresh 95-observation network."""
import argparse,json
import torch
from common import HOME,Policy,normalized
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=35);a=p.parse_args()
torch.manual_seed(42);torch.set_num_threads(4);device='cuda:0'
data=[torch.load(HOME/'teachers'/f'{t}.pt',weights_only=False) for t in ['forward','backward','turning']]
x=torch.cat([d['x'] for d in data]).to(device);y=torch.cat([d['y'] for d in data]).to(device)
order=torch.randperm(len(x),device=device);validation=order[:len(x)//10];train=order[len(x)//10:]
s={'running_mean':x[train].mean(0).double(),'running_variance':x[train].var(0).clamp_min(1e-6).double(),'current_count':torch.tensor(float(len(train)),device=device,dtype=torch.float64)}
c={'observation_preprocessor':s};x=normalized(x,c);model=Policy().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
history=[]
for epoch in range(a.epochs):
 model.train();order=train[torch.randperm(len(train),device=device)]
 for ids in order.split(2048):
  loss=(model(x[ids])-y[ids]).square().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
 with torch.no_grad():val=(model(x[validation])-y[validation]).square().mean().sqrt().item()
 history.append(val);print(epoch,'validation action RMSE',val,flush=True)
c['policy']=model.state_dict();c['value']=Policy(95,1).to(device).state_dict();c['value_preprocessor']={'running_mean':torch.zeros(1,device=device,dtype=torch.float64),'running_variance':torch.ones(1,device=device,dtype=torch.float64),'current_count':torch.tensor(1.,device=device,dtype=torch.float64)}
torch.save(c,a.output);open(a.output+'.json','w').write(json.dumps({'samples':len(x),'validation_action_rmse':history,'note':'random-transition holdout; not independent trajectory validation'},indent=2))
