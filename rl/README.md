# Bingo RL / Sim-training (moved here from the old BingoRobotics repo)

This is the reinforcement-learning + Isaac Sim side of the project. It was migrated into the
Blender folder (now the single source of truth). All asset paths resolve relative to the
`Blender/` root, so nothing here depends on the old BingoRobotics repo anymore.

## Layout

```
rl/
  bingo_rl/                     IsaacLab task package  (add rl/bingo_rl to sys.path -> `import bingo_rl`)
    bingo_rl/
      bingo.py                  rev_1 articulation cfg (base)
      env_cfg.py                velocity-tracking locomotion env
      improved_walking_cfg.py   the v7 flat/rough walking cfg  ✅ original RL walking success
      agents.py                 rsl_rl PPO runner cfgs
      amp/                      dog-mocap AMP (natural trot, v9–v14)  ✅ success
      track/                    DeepMimic-style deadpan tracker + residual control
                                 ✅ BEST tracking: 0.099 rad leg error (task Bingo-TrackRes-…)
      track_expr/               17-DOF: legs (RL) + head/tail (fed from reference)   ⚠ WIP
      track_v4/ , bingo_v4.py   21-DOF on the v4 USD (adds ears)                     ⚠ WIP:
                                 ear joints currently flail — zero-mass `continuous` joints
                                 need a drive fix (add inertia or drive kinematically)
      motions/                  reference .npz (dog trot/pace, deadpan clips, deadpan_ears)
      scripts/                  train.py, play.py, train_amp.py, play_amp.py, eval_*.py
  v4_usd/                       converted v4 URDF -> USD (21 joints incl. ears)
  bingo_scene*.usd, bingo_scene.py
  tools/                        eval / replay / render / diagnostics (see below)

../URDF/                        robot assets: bingo_urdf_rev_1, bingo_urdf_rev_3, "bingo_urdf v4_w_ear_joints"
../motions/                     the retargeted animation .npz (bingo_deadpan3.npz, bingo_laidback.npz, …)
```

## Registered gym tasks
`Bingo-Improved-Walking-Flat-v0` (walking, rsl_rl PPO) · `Bingo-AMP-Trot-Direct-v0` (dog AMP, skrl) ·
`Bingo-Track-Deadpan-Direct-v0` (absolute tracker) · `Bingo-TrackRes-Deadpan-Direct-v0` (**residual, best**) ·
`Bingo-TrackExpr-Deadpan-Direct-v0` (17-DOF) · `Bingo-TrackV4-Deadpan-Direct-v0` (21-DOF + ears).

## Running (from `~/robotics/IsaacLab`, `./isaaclab.sh -p <script>`)
- Train walking:  `rl/bingo_rl/scripts/train.py --task Bingo-Improved-Walking-Flat-v0 --headless`
- Train tracker:  `rl/bingo_rl/scripts/train_amp.py --task Bingo-TrackRes-Deadpan-Direct-v0 --algorithm PPO --num_envs 512 --headless`
- Metrics (B.4.5): `rl/tools/eval_metrics.py --task <T> --checkpoint <ckpt.pt> --agent_cfg rl/bingo_rl/bingo_rl/track/agents/skrl_ppo_cfg.yaml`
- GUI replay (faithful animation, no policy): `rl/tools/gui_replay.py --motion ../motions/bingo_laidback.npz`  (no `--headless`)
- Offscreen video: `rl/tools/dump_policy.py` (or `dump_poses.py`) → `rl/tools/render_poses.py` → ffmpeg

Gotchas: pass `--kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"` when headless;
Isaac's RTX camera render is intractable on the local 2070 (use the pose-based `render_poses.py`);
training checkpoints are written under `~/robotics/IsaacLab/logs/skrl/…`, not in this folder.

## Status (see ~/Bingo/AUDIT_DIAGNOSIS.md and EXPRESSION_TRANSFER_PROBLEM.md for detail)
- Leg tracking of deadpan hit the **<0.10 rad target (0.099)** with residual control + longer training.
- Head/tail expression works; **ears are the open WIP** (v4 drive fix needed).
