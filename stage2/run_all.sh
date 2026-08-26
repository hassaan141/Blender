#!/bin/bash
set -e
cd /home/hassaan/Bingo/Blender
B=~/Bingo/local/blender-5.2.0-linux-x64/blender
URDF="URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"
for NAME in Cheeky DeadPan Eccentric Enthusiastic Laidback Timid; do
  L=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  echo "==================== $NAME ===================="
  $B -b blend_sources/Bingo_$NAME.blend -P stage2/extract_source_motion.py -- \
     --out stage2/out/${L}_source.npz 2>&1 | grep -E "extracted|Traceback" | head -1
  python3 stage2/detect_contacts.py --keypoints stage2/out/${L}_source.npz \
     --out stage2/out/${L}_contacts.npz 2>&1 | grep -E "wrote|Error" | tail -1
  python3 stage2/solve_spatial_retarget.py --keypoints stage2/out/${L}_source.npz \
     --contacts stage2/out/${L}_contacts.npz --urdf "$URDF" \
     --out stage2/out/${L}_retarget.npz 2>&1 | grep -E "axis align|body tilt|ground placement|min paw z|ankle IK|root solve" | head -6
  $B -b blend_sources/Bingo_V4_AnimatorRig.blend -P stage2/bake_v4_motion.py -- \
     --motion stage2/out/${L}_retarget.npz \
     --out blend_sources/Bingo_${NAME}_V4_Retargeted.blend 2>&1 | grep -E "baked|Traceback" | head -1
done
echo "ALLDONE_BATCH"
