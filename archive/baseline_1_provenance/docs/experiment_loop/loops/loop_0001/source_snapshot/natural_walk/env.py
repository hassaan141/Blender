"""One opt-in contact-definition correction; inherited controller and physics unchanged."""
from pathlib import Path
import numpy as np,torch
from isaaclab.utils.math import quat_apply
from bingo_rl.walk_ref.bingo_walk_ref_env import BingoWalkRefEnv
class NaturalWalkEnv(BingoWalkRefEnv):
    def _get_rewards(self):
        reward=super()._get_rewards()
        if not self.cfg.correct_bl_contact:return reward
        if not hasattr(self,'_natural_bl_hull'):
            root=Path(__file__).resolve().parents[4]
            hull=np.load(root/'stage4/out/collision_hulls.npz')['bl_knee']
            self._natural_bl_hull=torch.tensor(hull,dtype=torch.float32,device=self.device)
        d=self.robot.data;idx=self.key_body_indexes[2];h=self._natural_bl_hull;N=self.num_envs;V=len(h)
        quat=d.body_quat_w[:,idx,None,:].expand(N,V,4).reshape(-1,4)
        vertices=quat_apply(quat,h[None].expand(N,V,3).reshape(-1,3)).reshape(N,V,3)
        lowest=vertices[:,:,2].min(dim=1).values+d.body_pos_w[:,idx,2]-self.scene.env_origins[:,2]
        tips,_=self._foot_tips_local();old_contact=tips[:,2,2]<.03;new_contact=lowest<.003
        fidx=np.clip(np.round(self._current_times()*self._ref_fps).astype(int),0,self._n_ref_frames-1)
        wanted=self._ref_contacts[fidx,2]>.5
        # Original contact_match weight .10 is the mean across 4 feet.
        # Replace exactly the BL quarter; no weight sweep or other reward change.
        return reward+.025*((new_contact==wanted).float()-(old_contact==wanted).float())
