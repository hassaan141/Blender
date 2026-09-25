#!/usr/bin/env python3
"""Export a BASELINE_1-format checkpoint to ONNX for browser probing only.

Same graph as tools/export_bingo_policy.py, but it never writes into the app's
public/policies directory (the installed browser policy stays untouched).
"""
import sys
from pathlib import Path
import torch
from mujoco_env import ROOT

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
if (ROOT / 'bingo-simulator').resolve() in out.parents: sys.exit('refusing to write inside bingo-simulator/')
c = torch.load(ckpt, map_location='cpu', weights_only=False); s = c['observation_preprocessor']

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(95,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12))
    def forward(self, x):
        x = ((x-s['running_mean'].float())/(torch.sqrt(s['running_variance'].float())+1e-8)).clamp(-5,5)
        return self.net(x)

n = Net(); n.net.load_state_dict({k.removeprefix('net_container.'):v for k,v in c['policy'].items() if k.startswith('net_container.')}); n.eval()
out.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(n, torch.zeros(1,95), out, input_names=['observation'], output_names=['actions'], opset_version=17)
print(f'ONNX={out}')
