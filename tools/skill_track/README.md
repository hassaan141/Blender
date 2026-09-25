# Full-body skill tracking (template for every full-body skill)

Full-body skills are **not** open-loop playback. The authored motion is the reference and a
learned MuJoCo residual policy supplies balance on the 12 legs:

```
legs (12):  target = ref.legs[k+1] + residual_scale[i] * clip(action, -1, 1)
expr (9):   target = reference feed-forward (entering blend starts from the live expression)
```

Training and browser use the **same MJCF** (`bingo-simulator/app/public/robot/bingo_scene.xml`),
MuJoCo **3.11.0** (browser version; Python via `PYTHONPATH=$PWD/scratch/mujoco_3_11_0`),
120 Hz physics, 24 Hz control, targets held for the 5 substeps. Root is never touched.

## Pipeline (do not redesign per skill)

| Step | Tool | Output |
|---|---|---|
| Reference | `skill_ref.py <name>` (from `motions/<name>_v4.npz`) | stand → 1 s blend → clip → 1 s blend → 0.5 s stand; clip anchored at origin, root z corrected to MJCF paw height, paws planted through transitions |
| Train | `train_skill.py --ref … --out …` (`--init`, `--stand-prob`, `--focus lo,hi,p`, `--residual-scale`) | `policy.pt` (actor, critic, obs normaliser, config) |
| Python gates | `success_rate.py <pt> 400 <ref.npz>`, `eval_skill.py --checkpoint … --video`, `compare_frames.py` | completion rate, tracking, slip, torque, residual vs motion, handoff, video, stills |
| Export | `export_skill_onnx.py <pt> <onnx>` | ONNX with normaliser baked in (checked vs torch) |
| Browser gate | `bingo-simulator/app/tools/probe_skill.mjs <onnx> <ref.json> <out> <Name> 20` (`PROBE_RESIDUAL_SCALE='[…]'` if per-joint) | real runtime + button path, stand → skill → stand |
| Parity gate | `parity_skill.py <pt> <ref.npz> <probe.json>` | **PASS/FAIL**: same obs → same action, entering transition locked, same outcome |
| Deploy | `deploy_skill.py <Name> <onnx> <ref.json> "<evidence>" <pt>` then `probe_skill.mjs deployed - <out> <Name> 10` | `public/policies/skill_<name>.onnx`, `public/motions/<name>_skill.json`, `tracked` entry in `public/motions/index.json` |

Browser side (fixed): `src/game/runtime/skill_runtime.js` mirrors `skill_env.py` observe/step/termination
line by line; `skills/load.js` registers `tracked` entries; `skills/manager.js` uses the tracker's own
fall/completion; `runtime/sim.js` runs it in `controlStep`. The existing gesture button triggers it.

## Acceptance bar

- Python: ≥ 99.5 % of 400 browser-like stand starts complete (mm-level leg deviation, random expression phase).
- Browser: 20/20 probe runs complete and return to STAND (z ≥ 0.17, tilt < 5°); deployed-asset probe 10/10.
- Parity gate PASS.
- Residual rms < ½ of the reference motion rms; stills/video still read as the authored skill.
- Loosen residual bounds only with evidence the bound binds (see Timid attempt 04).

A new skill supplies: the reference npz, a training command, the trained policy, and the deploy step.

## Skill status

| Skill | Status | Python (browser-like starts) | Browser | Notes |
|---|---|---|---|---|
| Timid | **deployed** (attempt 05, ONNX `c4cdd208…`) | 400/400 | 20/20 + deployed 10/10, handoff clean | parity PASS; residual rms 0.071 vs motion 0.266 rad; ori err 3.3° mean |
| Laidback | **deployed** (attempt 02, ONNX `dc378bc5…`) | 398/400 | 20/20 + deployed 10/10, handoff clean | parity PASS; residual rms 0.052 vs motion 0.512 rad; ori err 3.0° mean |
| Enthusiastic | training (attempt 01: scratch 2000 it → push fine-tune 2000 it) | – | – | open-loop 35 % |

Open-loop survival ranking (for ordering): Laidback 56 %, Enthusiastic 35 %, Timid 25 %, Deadpan 19 %, Cheeky 13 %, Eccentric 6 %.

## Default recipe (Timid, confirmed on Laidback: scratch → push fine-tune gave 398/400 with no other changes)

1. From scratch: 2000 it, lr 3e-4, 30 % stand starts → ~92 %.
2. Fine-tune: 1000 it, lr 1e-4, 50 % stand starts → ~98 %.
3. Fine-tune with random root pushes (`--push 0.0208`), lr 5e-5, 2000 it → 400/400. **Pushes were the step that made it robust.**
- Not helpful: focused resets on the weak frames (97.3 %), knee residual 0.25 → 0.35 (96.3 %; the bound was not the blocker).
Results: `tools/experiment_loop/results/skill_timid_0{1..5}_20260924/`.
