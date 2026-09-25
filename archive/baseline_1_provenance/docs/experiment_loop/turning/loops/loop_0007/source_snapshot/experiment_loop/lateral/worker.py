"""Apply the SAME bounded config to LateralWalkEnv's TRAIN script (reused,
unmodified, from natural_walk/train.py -- it is fully generic, gated only by
--task), without editing it. Mirrors rl/bingo_rl/experiment_loop/worker.py
exactly, except the gym.make() interception matches 'Bingo-Lateral-' instead of
'Bingo-NaturalWalk-'.

TRAIN ONLY. Unlike backward (which reuses natural_walk/evaluate.py verbatim via
a --cmd_vx CLI flag it already accepts), lateral's evaluate stage needs a
genuinely different measured axis and a new _cmd_vy attribute that
natural_walk/evaluate.py has no way to set -- see lateral/evaluate.py, invoked
directly by lateral_adapter.py, not through this generic worker.
"""
import sys, json, runpy
from pathlib import Path

EXPERIMENT_LOOP = Path(__file__).resolve().parents[1]  # rl/bingo_rl/experiment_loop
sys.path.insert(0, str(EXPERIMENT_LOOP))
from core import ROOT, ALLOWED, read, save  # noqa: E402

sys.path.insert(0, str(EXPERIMENT_LOOP / "lateral"))
# NOTE: do NOT import env.py at this top level -- it imports isaaclab.utils,
# which needs Isaac Sim's `pxr` (USD) modules that only become importable AFTER
# AppLauncher() runs inside the reused train.py below. Defer the import into
# make() (see below), which only ever fires from INSIDE train.py's main(),
# i.e. safely after AppLauncher.

stage = sys.argv.pop(1)
if stage != 'train':
    raise RuntimeError("lateral/worker.py handles only 'train'; evaluate goes through lateral/evaluate.py directly")
i = sys.argv.index('--context'); context = Path(sys.argv[i + 1]); del sys.argv[i:i + 2]
c = read(context)
if c.get('smoke'): raise RuntimeError('Simulator forbidden during smoke test')

import gymnasium as gym  # noqa: E402
original_make = gym.make
original_spec = gym.spec

_registered = False


def _ensure_registered(task: str):
    # Registration must happen before isaaclab_tasks' own gym.spec()-based
    # lookups (register_task_to_hydra -> load_cfg_from_registry), which fire
    # from INSIDE train.py's main(), i.e. after AppLauncher has already run --
    # env.py's `from isaaclab.utils import configclass` needs Isaac Sim's `pxr`
    # (USD) modules, which don't exist until AppLauncher() has launched Kit.
    # Patching gym.spec (not just gym.make) is required: register_task_to_hydra
    # calls gym.spec() directly, never gym.make(), to read env_cfg_entry_point.
    global _registered
    if task.startswith('Bingo-Lateral-') and not _registered:
        import env as lateral_env  # noqa: F401  (registers Bingo-Lateral-{Train,Play}-v0; safe here, post-AppLauncher)
        _registered = True


def make(task, *args, **kwargs):
    _ensure_registered(task)
    if task.startswith('Bingo-Lateral-'):
        cfg = kwargs['cfg']
        for k, v in c['settings'].items():
            if k not in ALLOWED or not ALLOWED[k][0] <= v <= ALLOWED[k][1]:
                raise ValueError(k)
            setattr(cfg, k, v)
        save(Path(c['arm_dir']) / (stage + '.effective_settings.json'),
             {**{k: getattr(cfg, k, None) for k in c['settings']}, 'motion_file': cfg.motion_file,
              'cmd_vy': cfg.cmd_vy, 'vx_range': list(cfg.vx_range),
              'physics_dt': cfg.sim.dt, 'decimation': cfg.decimation})
    return original_make(task, *args, **kwargs)


def spec(task, *args, **kwargs):
    _ensure_registered(task.split(":")[-1] if isinstance(task, str) else task)
    return original_spec(task, *args, **kwargs)


gym.make = make
gym.spec = spec
path = ROOT / 'rl/bingo_rl/bingo_rl/natural_walk/train.py'
sys.argv[0] = str(path)
runpy.run_path(str(path), run_name='__main__')
