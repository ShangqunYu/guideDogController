# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeGo1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000 # previously 4000
    save_interval = 100
    experiment_name = "unitree_go1_rough"
    empirical_normalization = False    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticDepth", # TODO: implement ActorCriticDepth, then switch to it.
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[512, 256, 128],
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
    depth_backbone = True
    depth_backbone_cfg = {
        "FC_output_dims": 32,
        "hidden_dims": 512,
        "learning_rate": 1.0e-3,
        "num_steps_per_env": 120,
    }


@configclass
class UnitreeGo1FlatPPORunnerCfg(UnitreeGo1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.depth_backbone = True
        self.max_iterations = 10000
        self.experiment_name = "unitree_go1_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
