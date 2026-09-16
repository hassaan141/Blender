# Bingo RL / Isaac Lab side of the project

Two live sub-projects share the v4 robot model (`bingo_rl/bingo_v4.py`, `v4_usd/bingo_v4.usd`):

1. **`locomotion/`** — Task 1, command-conditioned `[vx,vy,yaw_rate] -> 12 leg joints` velocity
   control via RSL-RL PPO. This is the actively developed branch of work; see
   `docs/locomotion/TASK1_DESIGN.md` and `docs/locomotion/TASK1_STATUS_AND_PLAN.md` for status.
   Driven by `rl/tools/{train,play,eval}_velocity.py` and exported via `rl/tools/export_onnx.py`.
2. **`stage5/`** — residual RL on top of the Stage 4 animation-pipeline reference (currently a
   Timid checkpoint under `logs/skrl/bingo_stage5_timid/`), driven by `rl/tools/eval_stage5.py`.
   `stage5/` imports helper classes from `track/`, `track_v4/`, and `amp/` (its `BingoTrackV4Env`
   base class and `BINGO_IMPROVED_CFG` asset config) — those three subpackages are kept as
   **library dependencies of stage5**, not as standalone tasks in current use; their own gym
   task IDs (`Bingo-AMP-Trot-Direct-v0`, `Bingo-Track-Deadpan-Direct-v0`,
   `Bingo-TrackV4-Deadpan-Direct-v0`) predate the v4/locomotion curriculum and aren't part of
   any documented current workflow.

## Layout

```
rl/
  bingo_rl/bingo_rl/       IsaacLab task package (add rl/bingo_rl to sys.path -> `import bingo_rl`)
    bingo_v4.py             canonical v4 robot articulation cfg
    improved_walking_cfg.py kept only because amp/ and track/ import BINGO_IMPROVED_CFG from it
    locomotion/              Task 1 - current
    stage5/                  residual RL on Stage 4 reference - current
    track/, track_v4/, amp/  library dependencies of stage5/ (see above) - not run standalone
  v4_usd/                   converted v4 URDF -> USD (21 joints incl. ears)
  tools/                    eval / replay / export / diagnostics - see below
```

Superseded code (the pre-v4 `bingo.py`/`env_cfg.py`/`agents.py` walking cfg, and the orphaned
`track_expr/` 17-DOF WIP) was moved to `archive/rl_orphaned/` — nothing current imports it.
`URDF/bingo_urdf_rev_1/` (used only by the archived `bingo.py`) moved to `archive/URDF_rev1/`.

## Running (Kiwi machine: `/pub0/muhammadf/IsaacLab`, conda env `isaaclab`)

```sh
source /pub0/muhammadf/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd /pub0/muhammadf/IsaacLab
./isaaclab.sh -p /pub0/muhammadf/Blender/rl/tools/train_velocity.py --headless \
  --kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window" ...
```

- Task 1 training/eval: `rl/tools/{train,play,eval}_velocity.py` — see `HANDOFF.md`.
- Task 1 ONNX export: `rl/tools/export_onnx.py`.
- Stage 5 eval: `rl/tools/eval_stage5.py`.
- GUI replay of a raw motion clip (no policy): `rl/tools/gui_replay.py --motion <clip>.npz`.
