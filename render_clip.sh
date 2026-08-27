#!/bin/bash
# Matched three-way stills for one clip: Ashley source | v4 reference | Isaac physics.
#
#   ./render_clip.sh Cheeky 1,45,90,135,180
#
# The camera is placed from each character's OWN anatomical axes, at the same 3/4
# angle and the same multiple of its own body size, so the only differences left in
# the sheet are pose differences. Ashley's floor height comes from her clip's
# contact file; the robot's floor is z=0 by construction.
set -e
cd /home/hassaan/Bingo/Blender
NAME=$1; FRAMES=${2:-1,45,90,135,180}; RES=${3:-420}
L=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
B=~/Bingo/local/blender-5.2.0-linux-x64/blender
O=outputs/compare/$L
mkdir -p $O
G=$(python3 -c "import numpy as np;print(float(np.load('stage2/out/${L}_contacts.npz')['ground']))")
# A retimed clip no longer shares a frame index with its source, so the sheet would
# put unrelated moments side by side. Map the robot frames back through the factor.
AFRAMES=$(python3 - "$L" "$FRAMES" <<'PY'
import sys, numpy as np
L, fr = sys.argv[1], sys.argv[2]
m = np.load(f"motions/{L}_v4.npz", allow_pickle=True)
f = float(m["stage4_retime_factor"]) if "stage4_retime_factor" in m.files else 1.0
print(",".join(str(max(1, int(round(1 + (int(x) - 1) / f)))) for x in fr.split(",")))
PY
)
[ "$AFRAMES" != "$FRAMES" ] && echo "[[ retimed clip: Ashley frames $AFRAMES <- robot frames $FRAMES"

$B -b blend_sources/Bingo_${NAME}.blend -P stage2/render_compare.py -- \
   --target ashley --outdir $O/ashley --frames $AFRAMES --res $RES --ground $G 2>&1 | grep -E "^\[\[ (rendered|heading|ashley heading|robot heading)"
$B -b blend_sources/Bingo_${NAME}_V4_Retargeted.blend -P stage2/render_compare.py -- \
   --target robot --outdir $O/reference --frames $FRAMES --res $RES 2>&1 | grep -E "^\[\[ (rendered|heading|ashley heading|robot heading)"

if [ -f stage4/out/${L}_v4_stage4.npz ]; then
  python3 stage4/log_to_motion.py --log stage4/out/${L}_v4_stage4.npz --out /tmp/${L}_phys.npz
  $B -b blend_sources/Bingo_V4_AnimatorRig.blend -P stage2/bake_v4_motion.py -- \
     --motion /tmp/${L}_phys.npz --out /tmp/${L}_phys.blend 2>&1 | grep "^\[\[ baked"
  $B -b /tmp/${L}_phys.blend -P stage2/render_compare.py -- \
     --target robot --outdir $O/physics --frames $FRAMES --res $RES 2>&1 | grep -E "^\[\[ (rendered|heading|ashley heading|robot heading)"
  python3 stage2/make_compare.py --left $O/reference --right $O/physics \
     --label-left "v4 reference" --label-right "Isaac physics" --out $O/ref_vs_physics.png
fi
python3 stage2/make_compare.py --left $O/ashley --right $O/reference --by-order \
   --label-left "Ashley" --label-right "v4 reference" --out $O/ashley_vs_reference.png
echo "[[ sheets in $O"
