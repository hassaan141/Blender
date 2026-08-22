# Bingo — Blender → Simulation Retarget

Turning authored dog animation in Blender into motion references the Bingo quadruped
can actually be trained on in Isaac Lab. The robot is defined by
`bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf`; everything here exists to get
Blender motion into the `.npz` motion schema that URDF's simulation consumes
(schema: [docs/BingoMocapPipelineSpec.md](docs/BingoMocapPipelineSpec.md) §A.7).

**Read first:** [CONTEXT.md](CONTEXT.md) — project context, team structure, and the open
verification question about whether the animators' rig matches the engineering URDF.
Then [MEMORY.md](MEMORY.md) — the verified working notes, including the
non-obvious facts that cost real time to discover. Then the spec.

## The two paths

**Path B — conform rig.** `build_rig.py` generates a Blender rig directly from
the URDF, so the rig *is* the robot: bones at the exact joint positions, local Z = the motor
axis, hard limit constraints at the real joint limits. Animation authored on it bakes
straight to `.npz` with **no retargeting** — baked angles pushed through the URDF's own FK
reproduce Blender's foot positions to 0.0005 mm.

```
build_rig.py  →  check_rig.py / test_ik.py  →  [animator]  →  bake_conform.py  →  .npz
```

**Path A — legacy art rig.** The animators' Rigify rig was not derived from the URDF, so its
motion is baked to world-space Cartesian and IK-retargeted onto Bingo's kinematics; earlier
work cut amplitude to 50–70%. ⚠️ **Whether that loss is actually necessary is under active
review — see [CONTEXT.md](CONTEXT.md).** Measurement shows the rig's leg chain is structurally
correct and proportionally within ~12–22% per segment, so which path is preferred is currently
an open decision, not a settled one.

```
bake_motion.py  →  *_raw.json  →  retarget.py  →  .npz
```

Both paths end at the same `.npz`, verified with `replay_motion.py` (kinematic) and
`playback_physics.py` (PD actuators + gravity, i.e. "could the machine do this?").

Of the six delivered performances, only **DeadPan** retargets cleanly (0.01 mm IK error,
1.4% clamping). They are an expressive layer, not a locomotion dataset — gait still needs
the capture in spec §B.2.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Pipeline spec (Part A: sim as-built, Part B: what 1:1 needs), the animator-facing robot spec, and the source animation guide PDF |
| `scripts/` | The whole pipeline; every script takes its paths as CLI args, run from this folder |
| `blend/conform/` | `Bingo_ConformRig_v4.blend` (**current**, built from v4 — 34 bones incl. ears, 22 meshes), the older rev_3 rig, and test animations |
| `blend/legacy_art_rig/` | The Rigify art rig, master scene, and 6 personality performances |
| `motions/` | Packed `.npz` motion files (spec §A.7) — the deliverable |
| `raw/` | Path A intermediates: world-space Cartesian JSON out of Blender |
| `output/video/`, `output/frames/` | Replay renders |
| `bingo_urdf v4_w_ear_joints/` | **Latest** engineering URDF — 21 joints (adds 4 ear joints). Legs identical to rev_3. |
| `bingo_urdf_rev_3/` | Previous URDF; all existing pipeline work and the USD scene target this |
| `BS Bingo_Final_Export.stp` | 70 MB STEP CAD assembly from the Berlin engineers (2026-05-13) |

`BS MP Gentle Systems Inbound/` (2 GB) is unrelated marketing/CAD material, left untouched.

## Running it

Run from this folder. Blender scripts need `blender` on PATH; the sim scripts need the
`isaaclab` conda env (see MEMORY.md §7 for the environment gotchas — the RTX driver flag
and the missing H.264 encoder will both bite).

```sh
# Path B — rebuild and verify the conform rig (v4: 21 joints, ears included)
blender -b --factory-startup -P scripts/build_rig.py -- \
    --urdf "bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \
    --out blend/conform/Bingo_ConformRig_v4.blend
blender -b blend/conform/Bingo_ConformRig_v4.blend -P scripts/check_rig.py -- \
    --urdf "bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf"
blender -b blend/conform/Bingo_ConformRig_v4.blend -P scripts/test_ik.py

# Path B — bake an animated conform rig straight to a motion file
# --dof 12 (default, legs only) or --dof 21 (legs + head + tail + ears)
blender -b blend/conform/<animated>.blend -P scripts/bake_conform.py -- \
    --out motions/clip.npz --hz 120 --dof 21

# Path A — legacy art rig
blender -b blend/legacy_art_rig/Bingo_DeadPan.blend -P scripts/bake_motion.py -- \
    --out raw/deadpan_raw.json
python3 scripts/retarget.py raw/deadpan_raw.json \
    --urdf bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf \
    --out motions/bingo_deadpan.npz

# Verify before spending GPU time
python scripts/replay_motion.py   --motion motions/bingo_deadpan.npz --out output/video/deadpan --headless
python scripts/playback_physics.py --motion motions/bingo_deadpan.npz --out output/video/deadpan_phys --headless
```

## Handoff to the animator

Send `blend/conform/Bingo_ConformRig.blend` + `docs/Bingo_Robot_Spec_for_Animation.md`.
(`.blend1` is Blender's stale autosave — do not send it.) She returns the animated `.blend`;
run `check_rig.py` on it, then `bake_conform.py`. No export step, no format choice, no unit
conversion on her side.

## Open

The full list is MEMORY.md §6. The ones that gate 1:1 retargeting:

1. **Schema: 21-DOF now implemented on the data side** (`--dof 21`, see [CONTEXT.md](CONTEXT.md)
   §5A.1) — legs + head + tail + ears, a strict extension of the 12-DOF layout. **The RL side
   still needs updating** (obs dims, `robot_ctrl_indexes`); that code is not in this repo.
2. **No torque feasibility check** — position and velocity only. v4 ships `effort=0` on every
   joint, so rev_3's (also placeholder) values are carried forward. Blocked on real motor data.
3. **SY overrun**: the policy scales SY ×0.3 (≈±0.126 rad) but IK spends up to 0.30 rad
   holding feet planted. Warned, not solved.
4. **Training side unchanged** — no tracking objective, phase input, or early termination
   (spec §B.4). AMP is style imitation; 1:1 is a different objective.
