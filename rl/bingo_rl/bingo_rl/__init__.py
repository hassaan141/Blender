"""Bingo quadruped RL tasks. Importing this package registers the gym environments."""

import gymnasium as gym

from . import amp  # noqa: F401  (registers Bingo-AMP-Trot-Direct-v0; dependency of track_v4/stage5)
from . import track  # noqa: F401  (registers Bingo-Track-Deadpan-Direct-v0; dependency of track_v4/stage5)
from . import track_v4  # noqa: F401  (registers Bingo-TrackV4-Deadpan-Direct-v0, 21 DOF incl. ears; dependency of stage5)
from . import stage5  # noqa: F401  (registers Bingo-Stage5-Timid-Direct-v0, residual RL on Stage 4)
from . import locomotion  # noqa: F401  (registers Bingo-Velocity-Flat-v4-v0, free locomotion on v4)
from . import walk_ref  # noqa: F401  (registers Bingo-WalkRef-v4-{A,B,C,Play}-v0, cyclic-walk residual RL)
from . import locomotion2_amp  # noqa: F401  (registers Bingo-Locomotion2-AMP-Direct-{,Play}-v0, dog-style AMP prior on v4)

# track_expr/ (a track_v4/stage5 ancestor, superseded) archived to archive/rl_orphaned/ -
# nothing current imports it. agents.py, bingo.py, env_cfg.py were archived there too in an
# earlier pass but restored: improved_walking_cfg.py imports BingoFlatPPORunnerCfg/BINGO_CFG/
# BASE_LINK/FOOT_BODIES/BingoFlatEnvCfg from them directly, and improved_walking_cfg.py itself
# is load-bearing for bingo_v4.py (BINGO_V4_CFG = deepcopy(BINGO_IMPROVED_CFG)) - i.e. Task 1's
# locomotion env, not just amp/track. The top-level Bingo-Improved-Walking-* /
# Bingo-Velocity-Flat-v0 / Bingo-Velocity-Rough-v0 gym IDs that exposed them directly still
# predate the v4/locomotion curriculum (see HANDOFF.md) and stay dropped.
