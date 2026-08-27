"""Turn a Stage-4 physics rollout back into a motion .npz, so what the ROBOT
actually did can be baked onto the Blender skeleton and rendered in the same view
as Ashley and the reference.

rl/tools/track_v4_physics.py logs the achieved joint angles and root pose every
control step; this repacks them in the canonical DOF order (the log is in Isaac's
breadth-first order, the pipeline is per-leg, so the remap is by NAME) and writes
the schema stage2/bake_v4_motion.py consumes.

  python3 stage4/log_to_motion.py --log stage4/out/laidback_v4_stage4.npz \
      --out /tmp/laidback_phys.npz
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from v4_kinematics import DOF_ORDER

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = np.load(a.log, allow_pickle=True)
names = [str(x) for x in d["joint_names"]]
missing = [n for n in DOF_ORDER if n not in names]
if missing:
    raise SystemExit(f"log is missing {missing}")
idx = [names.index(n) for n in DOF_ORDER]
q = np.asarray(d["q_act"], float)[:, idx]
fps = float(d["fps"])
out = dict(fps=np.array(fps), dof_names=np.array(DOF_ORDER),
           dof_positions=q.astype(np.float32),
           dof_velocities=np.gradient(q, 1.0 / fps, axis=0).astype(np.float32),
           root_pos=np.asarray(d["root_pos"], np.float32),
           root_quat=np.asarray(d["root_quat"], np.float32),
           frames=np.arange(1, len(q) + 1, dtype=np.int32),
           source=np.array(os.path.basename(a.log)))
np.savez(a.out, **out)
print(f"[[ {len(q)} frames @ {fps:g} fps of ACHIEVED physics state -> {a.out}")
