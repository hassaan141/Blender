# Per-clip Stage 2->4 recipes

`run_clip.sh <Clip>` sources `recipes/<clip>.conf` if it exists. Everything a clip
needs beyond the common pipeline lives in that one file, so the canonical motion is
reproducible from one command and every deviation from Ashley's authored motion is
written down next to the measurement that justified it.

Variables (all optional):

  SOLVE_ARGS    extra arguments for stage2/solve_spatial_retarget.py
  GROUND_ARGS   extra arguments for stage4/ground_fix.py
  UNCOLLIDE     1 = run stage2/uncollide.py (shank vs torso self-collision)
  UNCOLLIDE_ARGS extra arguments for it
  WRENCH        1 = run stage4/wrench_refine.py + stage4/reik_root.py
  WRENCH_ARGS   extra arguments for wrench_refine
  NOGLIDE       1 = delete the reference's horizontal body travel
  RETIME        uniform slowdown factor (stage4/retime.py); empty = keep tempo
  YAW_ALIGN     x|y - rotate the finished clip about world Z so its net travel runs
                along that axis (stage4/yaw_align.py). Root only; no DOF changes.
