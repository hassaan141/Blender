#!/bin/bash
cd /home/hassaan/Bingo/Blender
BL=~/Bingo/local/blender-5.2.0-linux-x64/blender
for NAME in Cheeky DeadPan Eccentric Enthusiastic Laidback Timid; do
  L=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  echo "==================== $NAME ===================="
  python3 stage4/ground_fix.py --motion motions/${L}_v4.npz --source stage2/out/${L}_source.npz \
    --out motions/${L}_v4_g.npz --w-ground 200 --w-reg 0.5 --clearance 0.0 2>&1 \
    | grep -E "ON the floor|lowest point|penetrating|joint change"
  mv -f motions/${L}_v4_g.npz motions/${L}_v4.npz
  $BL -b blend_sources/Bingo_V4_AnimatorRig.blend -P stage2/bake_v4_motion.py -- \
    --motion motions/${L}_v4.npz --out blend_sources/Bingo_${NAME}_V4_Retargeted.blend 2>&1 \
    | grep -E "^\[\[ baked"
done
echo ALLDONE2
