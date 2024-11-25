# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains import TerrainImporterCfg
import math
from dataclasses import MISSING
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import (
    RayCasterCameraCfg, 
    RayCaster,
    RayCasterCfg,
    ContactSensorCfg,
    patterns)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from .terrain_cfg import UNITREE_GO1_FLAT_TERRAIN_CFG

from .rough_env_cfg import (
    UnitreeGo1SceneCfg,
    UnitreeGo1RoughEnvCfg
)

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class UnitreeGo1FlatSceneCfg(UnitreeGo1SceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UNITREE_GO1_FLAT_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

@configclass
class UnitreeGo1FlatRewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination = RewTerm(
        func=mdp.is_terminated,
        weight = -1.0
    )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    # )
    # collision = RewTerm(
    #     func=mdp.collision, 
    #     weight=-1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")}
    # )
    orientation = RewTerm(
        func = mdp.flat_orientation_l2,
        weight = -1.0
    )
    
    smoothness = RewTerm(
        func = mdp.action_smoothness_penalty,
        weight = -0.05
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.001,
        params={"target_height": 31},
    )

@configclass
class UnitreeGo1CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.2), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    scene = UnitreeGo1FlatSceneCfg(num_envs=250, env_spacing=0)
    commands: UnitreeGo1CommandsCfg = UnitreeGo1CommandsCfg()
    rewards: UnitreeGo1FlatRewardsCfg = UnitreeGo1FlatRewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None


@configclass
class UnitreeGo1CommandsCfg_PLAY:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # want robots to always move forward
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(0, 0), ang_vel_z=(0, 0), heading=(0, 0)
        ),
    )

@configclass
class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    commands: UnitreeGo1CommandsCfg_PLAY = UnitreeGo1CommandsCfg_PLAY()
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
