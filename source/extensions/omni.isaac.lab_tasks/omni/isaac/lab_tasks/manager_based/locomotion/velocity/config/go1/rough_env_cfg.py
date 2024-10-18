# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import (
    RayCasterCameraCfg, 
    RayCaster,
    patterns)
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    RewardsCfg,
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    TerminationsCfg,
    CurriculumCfg
)

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO1_CFG  # isort: skip

@configclass
class UnitreeGo1SceneCfg(MySceneCfg):
    depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_tilt_link/head_camera",
            mesh_prim_paths=["/World/envs/envs_0/Cabinet"],
            # mesh_prim_paths=["/World/GroundPlane"],
            update_period=0.1,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.055, 0.02, 0.0225), rot=(0,0,1,0)),
            data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
            debug_vis=True,
            max_distance=10.0,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=480,
                width=640,
            ),
        )


##
# MDP Settings
##

class UnitreeGo1RewardsCfg(RewardsCfg):
    # TODO: migrate custom rewards from IsaacGym
    # rewards are written in omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp
    # # format: 
    # reward_name = RewTerm(func = mdp.<reward_func>, 
    #                       weight = <reward_weight>,
    #                       params=mdp.<reward_params>)
    pass
    
class UnitreeGo1CommandsCfg(CommandsCfg):
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: UnitreeGo1SceneCfg = UnitreeGo1SceneCfg()
    rewards : UnitreeGo1RewardsCfg = UnitreeGo1RewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


##
# PLAY Configs
##

class UnitreeGo1CommandsCfg_PLAY(CommandsCfg):
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
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0, 0), ang_vel_z=(0, 0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    commands: UnitreeGo1CommandsCfg_PLAY = UnitreeGo1CommandsCfg_PLAY()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
