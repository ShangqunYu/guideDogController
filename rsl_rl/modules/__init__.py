#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_depth import ActorCriticDepth
from .normalizer import EmpiricalNormalization
from .depth_backbone import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

__all__ = ["ActorCritic", 
           "ActorCriticRecurrent", 
           "EmpiricalNormalization", 
           "RecurrentDepthBackbone", 
           "DepthOnlyFCBackbone58x87",
           "ActorCriticDepth"]
