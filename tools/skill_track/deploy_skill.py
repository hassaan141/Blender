#!/usr/bin/env python3
"""Deploy a validated tracked skill into the browser simulator.

Copies <onnx> -> public/policies/skill_<name>.onnx and <reference.json> ->
public/motions/<name>_skill.json, and registers it under "tracked" in
public/motions/index.json (removing any "not_ready" entry of the same name).
Run only after the acceptance gates pass (success_rate, probe_skill, parity_skill).
Usage: deploy_skill.py <Name> <policy.onnx> <reference.json> "<evidence>" <policy.pt>
"""
import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; PUB = ROOT / "bingo-simulator/app/public"
name, onnx, ref, evidence = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
import torch
from skill_env import scale_from_config
rscale = [round(float(x), 6) for x in scale_from_config(torch.load(sys.argv[5], map_location='cpu', weights_only=False)['config'])]
slug = name.lower()
dst_onnx, dst_ref = PUB / f"policies/skill_{slug}.onnx", PUB / f"motions/{slug}_skill.json"
shutil.copyfile(onnx, dst_onnx); shutil.copyfile(ref, dst_ref)
idx_path = PUB / "motions/index.json"; idx = json.loads(idx_path.read_text())
entry = {"name": name, "reference": f"./motions/{slug}_skill.json", "policy": f"./policies/skill_{slug}.onnx",
         "obs_dim": 82, "residual_scale": rscale, "evidence": evidence,
         "onnx_sha256": hashlib.sha256(dst_onnx.read_bytes()).hexdigest()}
idx["tracked"] = [e for e in idx.get("tracked", []) if e["name"] != name] + [entry]
idx["not_ready"] = [e for e in idx.get("not_ready", []) if e["name"] != name]
idx_path.write_text(json.dumps(idx, indent=1) + "\n")
print(f"deployed {name}: {dst_onnx.relative_to(ROOT)} {dst_ref.relative_to(ROOT)} sha256={entry['onnx_sha256'][:12]}")
