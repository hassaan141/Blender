# Stage 2 Cheeky visual sign-off

The Stage 2 output is:

`blend_sources/Bingo_Cheeky_V4_Retargeted.blend`

It contains 180 baked frames at 24 fps on the exact v4 physical skeleton. Camera
setup and video rendering are not required for sign-off.

## Quick check in Blender

1. Open `Bingo_Cheeky_V4_Retargeted.blend` in Blender 5.2.
2. Stay in **Object Mode** and use **Material Preview** or **Solid** viewport shading.
3. Select the dog (or `Bingo_Robot` in the Outliner) and press **Numpad `.`** to
   frame it. Press **Home** if the full motion path is outside the view.
4. Press **Spacebar** to play frames 1-180. The timeline should show **24 fps**.
5. For a closer check, type each frame below into the timeline's current-frame box,
   then press Numpad `.` again if the moving dog has left the view.

## Frames worth checking

| Frame | What to inspect |
|---:|---|
| 1 | Starting pose; no exploded meshes or detached limbs |
| 31 | Early travel and roughly 10-degree body turn |
| 39 | Strong head-roll gesture |
| 60-61 | Strong upward tail-pitch gesture |
| 81 | Recognizable crouch |
| 91-94 | Mid-turn pose and strongest head-yaw gesture |
| 121 | Recognizable play-bow and roughly 89-degree body turn |
| 124-128 | Strong head and asymmetric ear expression |
| 151 | Recovery/return portion of the performance |
| 180 | Clean finish, with no floor penetration or mesh separation |

## Pass criteria

- The crouch, play-bow, travel, and turns read as one continuous Cheeky performance.
- All four legs stay attached and bend at the real v4 pivots.
- The paws do not visibly pass below the floor. A small amount of planted-foot drift
  is expected; the corrected mean consecutive-frame slide during stance is 2.6 mm.
- Head, tail, and both ears visibly animate. Their physical joints, rather than the
  animator controls, carry the baked keys.
- Wide paw placement and large paw raises may look more splayed or reduced than the
  Ashley source. This is expected because the v4 `SY` joints are limited to about
  +/-24 degrees; it is a real mechanical reach limit, not a retargeting failure.

For an A/B check, open `blend_sources/Bingo_Cheeky.blend` in a second Blender window
and compare the same numbered frames. Judge the gesture, body direction, crouch,
head, tail, and ears rather than expecting identical paw positions—the two rigs have
different morphology and joint limits.

## After sign-off

If the motion passes visually, Stage 2 is complete. Do not change the retarget solve
to polish unreachable paw poses. The next project stage is physics/Isaac tracking,
which should begin only after this human visual approval.
