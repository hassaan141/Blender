#!/bin/bash
# Full Stage 2 -> Stage 4-ready canonical pipeline for one clip.
#
#   ./run_clip.sh Cheeky              solve -> ground -> [recipe] -> bake -> conform
#   ./run_clip.sh Cheeky --no-solve   reuse stage2/out/<clip>_retarget.npz
#
# Anything a clip needs beyond the common path lives in recipes/<clip>.conf, next
# to the measurement that justified it (see recipes/README.md). Nothing is tuned
# inline here.
#
# Order matters. Every reference-modifying pass runs on the SOLVER output, before
# the Blender bake, so blend_sources/Bingo_<Clip>_V4_Retargeted.blend and
# motions/<clip>_v4.npz are produced from one and the same joint trajectory - the
# old order (bake, conform, ground_fix, re-bake) baked twice and left the two
# artefacts one pass apart.
set -e
cd /home/hassaan/Bingo/Blender
NAME=$1; shift || true
L=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
B=~/Bingo/local/blender-5.2.0-linux-x64/blender
URDF="URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"
O=stage2/out
GROUND=${GROUND:-1}; GROUND_ARGS=""; SOLVE_ARGS=""; POLISH=${POLISH:-0}; POLISH_ARGS=""
POLISH_STAGE=${POLISH_STAGE:-ground}   # ground | noglide
WRENCH=0; WRENCH_ARGS=""; NOGLIDE=0; RETIME=""; YAW_ALIGN=""
UNCOLLIDE=0; UNCOLLIDE_ARGS=""
[ -f "recipes/${L}.conf" ] && { echo "=== recipe recipes/${L}.conf"; . "recipes/${L}.conf"; }

if [[ " $* " != *" --no-solve "* ]]; then
  echo "=== Stage 2 solve"
  python3 stage2/solve_spatial_retarget.py --keypoints $O/${L}_source.npz \
    --contacts $O/${L}_contacts.npz --urdf "$URDF" --out $O/${L}_retarget.npz $SOLVE_ARGS \
    | grep -vE "^\[\[ ear frame"
fi
SRC=$O/${L}_retarget.npz

if [ "$GROUND" = "1" ]; then
  echo "=== Stage 2.5 ground / contact projection"
  python3 stage4/ground_fix.py --motion $SRC --source $O/${L}_source.npz \
    --out $O/${L}_grounded.npz --w-ground 200 --w-reg 0.5 --clearance 0.0 $GROUND_ARGS
  SRC=$O/${L}_grounded.npz
fi

# stance_polish is OFF by default. It does what it says - Laidback's reference skate
# falls 1.14 -> 0.69 mm/frame - but that is not the objective that predicts physical
# tracking: on DeadPan it took reference slip 1.19 -> 0.81 and the PHYSICS fall
# frame 227 -> 121, on Cheeky 2.05 -> 1.33 and the fall 90 -> 40. Leg-only
# (--root 0) and joint-smoothed (--w-smooth 20/80) variants regress identically, so
# it is the extra leg authority itself, not the body offset or target roughness.
# Kept as a tool, and as the reminder that a better reference number can still be a
# worse robot.
if [ "$POLISH" = "1" ] && [ "$POLISH_STAGE" = "ground" ]; then
  echo "=== Stage 2.6 stance polish (material-velocity no-slip)"
  python3 stage2/stance_polish.py --motion $SRC --out $O/${L}_polished.npz $POLISH_ARGS
  SRC=$O/${L}_polished.npz
fi

if [ "$UNCOLLIDE" = "1" ]; then
  echo "=== Stage 2.7 push the shank out of the torso (local, paw-preserving)"
  python3 stage2/uncollide.py --motion $SRC --out $O/${L}_uncollided.npz $UNCOLLIDE_ARGS
  SRC=$O/${L}_uncollided.npz
fi

if [ "$WRENCH" = "1" ]; then
  echo "=== Stage 4.1 contact-wrench feasibility (body offset)"
  python3 stage4/wrench_refine.py --motion $SRC --out $O/${L}_wrench.npz $WRENCH_ARGS
  echo "=== Stage 4.2 consume the offset with leg IK (paws stay put)"
  python3 stage4/reik_root.py --motion $O/${L}_wrench.npz --out $O/${L}_wrenchik.npz
  SRC=$O/${L}_wrenchik.npz
fi

if [ "$NOGLIDE" = "1" ]; then
  echo "=== Stage 4.3 remove the unsupported horizontal glide"
  python3 - "$SRC" $O/${L}_noglide.npz <<'PY'
import sys, numpy as np
m = np.load(sys.argv[1], allow_pickle=True); d = {k: m[k] for k in m.files}
rp = m["root_pos"].astype(float).copy()
print(f"[[ removed horizontal body travel: max "
      f"{np.linalg.norm(rp[:,:2]-rp[0,:2],axis=1).max()*1000:.0f} mm")
rp[:, :2] = rp[0, :2]
d["root_pos"] = rp.astype(np.float32); d["glide_removed"] = np.array(True)
if "body_positions" in d:
    bp = d["body_positions"].astype(float).copy(); bp[:, 0] = rp
    d["body_positions"] = bp.astype(np.float32)
np.savez(sys.argv[2], **d)
PY
  SRC=$O/${L}_noglide.npz
fi

if [ "$POLISH" = "1" ] && [ "$POLISH_STAGE" = "noglide" ]; then
  # After the glide is gone, not before: the anchor for a stance run is only
  # meaningful once the body is no longer travelling through it.
  echo "=== Stage 4.35 stance polish (material-velocity no-slip)"
  python3 stage2/stance_polish.py --motion $SRC --out $O/${L}_polished.npz $POLISH_ARGS
  SRC=$O/${L}_polished.npz
fi

if [ -n "$RETIME" ]; then
  echo "=== Stage 4.4 uniform retime x$RETIME"
  python3 stage4/retime.py --motion $SRC --out $O/${L}_retimed.npz --factor $RETIME
  SRC=$O/${L}_retimed.npz
fi

if [ -n "$YAW_ALIGN" ]; then
  echo "=== Stage 4.5 world-frame yaw alignment (root only, joints untouched)"
  python3 stage4/yaw_align.py --motion $SRC --out $O/${L}_yaw.npz --axis $YAW_ALIGN
  SRC=$O/${L}_yaw.npz
fi

echo "=== bake onto the exact v4 physical skeleton"
$B -b blend_sources/Bingo_V4_AnimatorRig.blend -P stage2/bake_v4_motion.py -- \
   --motion $SRC --out blend_sources/Bingo_${NAME}_V4_Retargeted.blend 2>&1 \
   | grep -E "^\[\[ (baked|wrote)|Traceback"

echo "=== read the bake back (Stage 3 source of truth)"
$B -b blend_sources/Bingo_${NAME}_V4_Retargeted.blend -P scripts/bake_conform.py -- \
   --rig Bingo_Robot --dof 21 --hz 24 --out /tmp/${L}_baked.npz 2>&1 \
   | grep -E "^\[\[ wrote|Traceback"

echo "=== verify bake == solve, attach physical contacts"
python3 stage2/finalize_motion_contacts.py --baked /tmp/${L}_baked.npz \
   --stage2 $SRC --out motions/${L}_v4.npz

echo "=== QA"
python3 stage2/slip_audit.py motions/${L}_v4.npz --contacts $O/${L}_contacts.npz | tail -1
python3 stage4/support_audit.py motions/${L}_v4.npz
python3 stage4/wrench_refine.py --motion motions/${L}_v4.npz | grep -E "BEFORE"
