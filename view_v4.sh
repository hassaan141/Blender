#!/usr/bin/env bash
# View the Bingo v4 robot in the Isaac Sim viewport.
#
#   ./view_v4.sh                 zero pose, window stays open
#   ./view_v4.sh --pose default  the crouch from BINGO_V4_CFG
#   ./view_v4.sh --pose physics  let it fall under gravity
#   ./view_v4.sh --rebuild       re-convert the URDF -> USD first, then view
#
# Any other flags are passed straight through to rl/tools/view_v4.py.
set -e

BINGO=/home/hassaan/Bingo/Blender
ISAAC=${ISAACLAB:-$HOME/robotics/IsaacLab}
URDF="$BINGO/URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"
USD="$BINGO/rl/v4_usd/bingo_v4.usd"

REBUILD=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--rebuild" ]; then REBUILD=1; else ARGS+=("$a"); fi
done

if [ "$REBUILD" = "1" ] || [ ! -f "$USD" ]; then
  [ -f "$USD" ] || echo ">> $USD missing - building it"
  echo ">> converting URDF -> USD (this opens Isaac headless, takes a minute)"
  cd "$ISAAC"
  ./isaaclab.sh -p "$BINGO/rl/tools/convert_v4_usd2.py" \
      --urdf "$URDF" --out "$USD" --headless 2>&1 \
    | grep -E "USD_JOINTS|EAR_JOINTS|CONVERT_" || true
fi

echo ">> opening Isaac Sim (first launch warms shader cache - be patient)"
cd "$ISAAC"
exec ./isaaclab.sh -p "$BINGO/rl/tools/view_v4.py" "${ARGS[@]}"
