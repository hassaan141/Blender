"""Isolated natural-walk tasks; original controller and physics are inherited unchanged."""
import os
from pathlib import Path
import gymnasium as gym
from isaaclab.utils import configclass
from bingo_rl.walk_ref.bingo_walk_ref_env_cfg import BingoWalkRefEnvCfg_C, BingoWalkRefPlayEnvCfg
ROOT=Path(__file__).resolve().parents[4]
REFERENCE=os.environ.get('BINGO_NATURAL_REFERENCE',str(ROOT/'docs/natural_walk/reference_01/reference.npz'))
@configclass
class NaturalTrainCfg(BingoWalkRefEnvCfg_C):
    correct_bl_contact: bool=os.environ.get("BINGO_NATURAL_CONTACT_FIX","0")=="1"
    motion_file: str=REFERENCE
    ear_file: str=REFERENCE
    vx_range=(0.2,0.3)
    stand_prob=0.0
@configclass
class NaturalPlayCfg(BingoWalkRefPlayEnvCfg):
    correct_bl_contact: bool=os.environ.get("BINGO_NATURAL_CONTACT_FIX","0")=="1"
    motion_file: str=REFERENCE
    ear_file: str=REFERENCE
for name,cfg in [('Train',NaturalTrainCfg),('Play',NaturalPlayCfg)]:
    gym.register(id=f'Bingo-NaturalWalk-{name}-v0',entry_point='bingo_rl.natural_walk.env:NaturalWalkEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':cfg,'skrl_cfg_entry_point':'bingo_rl.track.agents:skrl_ppo_cfg.yaml'})
