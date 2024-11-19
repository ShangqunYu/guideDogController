# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from dataclasses import MISSING
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import (
    RayCasterCameraCfg, 
    RayCaster,
    RayCasterCfg,
    ContactSensorCfg,
    patterns)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from .terrain_cfg import UNITREE_GO1_ROUGH_TERRAINS_CFG

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
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UNITREE_GO1_ROUGH_TERRAINS_CFG,
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

    # robot: ArticulationCfg = MISSING
    """Configuration for the terrain scene with a legged robot."""
    depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk", # previously "{ENV_REGEX_NS}/Robot/head_tilt_link/head_camera"
            mesh_prim_paths=["/World/ground"],
            update_period=0.1,
            attach_yaw_only=True,
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.245+0.027, 0.0075, 0.072+0.02),  # imported from Isaacgym
                rot=(1,0,0,0), # previously (0, 0, 1, 0) in "ros" convention 
                convention="world"),
            data_types=["distance_to_image_plane"],
            debug_vis=False,
            max_distance=10.0,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=58,
                width=87,
            ),
        )
    
    

##
# MDP Settings
# ##

@configclass
class UnitreeGo1RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0, # prev 0.25
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    termination = RewTerm(
        func=mdp.is_terminated,
        weight = -1.0
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=1.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=1.0)
    collision = RewTerm(
        func=mdp.collision, 
        weight=-.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")}
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.001,
        params={"target_height": 27.0},
    )

@configclass
class UnitreeGo1ObservationsCfg(ObservationsCfg):
    
    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
    
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    class DepthBackboneCfg(ObsGroup):
        depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
    def __post_init__(self):
        super().__post_init__()
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    depth_image: DepthBackboneCfg = DepthBackboneCfg()


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
            lin_vel_x=(0.5, 2.0), lin_vel_y=(0, 0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: UnitreeGo1SceneCfg = UnitreeGo1SceneCfg(num_envs=300, env_spacing=2.5)
    rewards: UnitreeGo1RewardsCfg = UnitreeGo1RewardsCfg()
    observations: UnitreeGo1ObservationsCfg = UnitreeGo1ObservationsCfg()
    commands: UnitreeGo1CommandsCfg = UnitreeGo1CommandsCfg()
    class env:
        n_proprio = 48
        n_privileged = 235

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # whether to render camera
        self.depth_camera_render = False

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
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
        self.rewards.dof_torques_l2.weight = -0.0001
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


##
# PLAY Configs
##
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
            lin_vel_x=(1.0, 2.0), lin_vel_y=(0, 0), ang_vel_z=(0, 0), heading=(0, 0)
        ),
    )


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    scene: UnitreeGo1SceneCfg = UnitreeGo1SceneCfg(num_envs=10, env_spacing=2.5)
    commands: UnitreeGo1CommandsCfg_PLAY = UnitreeGo1CommandsCfg_PLAY()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # enabling rendering depth camera:
        self.depth_camera_render = True

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
