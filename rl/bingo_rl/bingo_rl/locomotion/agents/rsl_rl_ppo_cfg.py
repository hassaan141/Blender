"""PPO runner config for Bingo velocity locomotion (Task 1).

Structure follows bingo_rl/agents.py, which already runs against the locally
installed rsl_rl. Only the values that Bingo's own scale justifies are changed.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BingoVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "bingo_velocity_v4"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # [128,128,128] is what the project's successful flat walking policy used
        # (agents.BingoFlatPPORunnerCfg). Bingo's observation is 66-dim and the task
        # is flat-ground locomotion; a larger net was not what was missing before.
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BingoVelocityPPORunnerCfg_PLAY(BingoVelocityPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1
