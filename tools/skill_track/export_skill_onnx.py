#!/usr/bin/env python3
"""Export a skill policy (obs normalizer + actor mean) to ONNX and check it numerically.

Usage: export_skill_onnx.py <policy.pt> <out.onnx>
"""
import sys
import numpy as np
import torch
from skill_env import OBS_DIM
from train_skill import Actor

ckpt, out = sys.argv[1], sys.argv[2]
c = torch.load(ckpt, map_location="cpu", weights_only=False)
actor = Actor(); actor.load_state_dict(c["actor"]); actor.eval()


class Exported(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net = actor.net
        self.register_buffer("mean", torch.as_tensor(c["obs_mean"], dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.sqrt(c["obs_var"] + 1e-8), dtype=torch.float32))
    def forward(self, x): return self.net(((x - self.mean) / self.std).clamp(-5, 5))


m = Exported().eval()
torch.onnx.export(m, torch.zeros(1, OBS_DIM), out, input_names=["observation"], output_names=["actions"], opset_version=17)
import onnxruntime as ort
x = np.random.default_rng(0).normal(size=(64, OBS_DIM)).astype(np.float32) * 2
sess = ort.InferenceSession(out)
diff = max(np.abs(sess.run(None, {"observation": x[i:i+1]})[0] - m(torch.as_tensor(x[i:i+1])).detach().numpy()).max() for i in range(64))
print(f"ONNX={out} max|onnx-torch|={diff:.2e}")
assert diff < 1e-4
