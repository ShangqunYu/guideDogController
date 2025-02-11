# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from omni.isaac.lab.terrains import TerrainGeneratorCfg

UNITREE_GO1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope_inv_left": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_left": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "horizontal_rails_left": terrain_gen.HfHorizontalRailsTerrainCfg(
            proportion=0.2, 
            rail_height_range=(0.04, 0.07), 
            rail_thickness=0.25, 
            num_rails=3,
            horizontal_scale=0.005,
        ),
        # "trenches_left": terrain_gen.HfHorizontalTrenchesTerrainCfg(
        #     proportion=0.1, trench_width_range=(0.06, 0.15), num_trenches = 2.0, horizontal_scale=0.03),
        "random_rough_left": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25
        ),
        "pyramid_stairs_inv_left": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "floating_platform": terrain_gen.MeshFloatingPlatformTerrainCfg(
        #     proportion=0.2,
        #     step_height_range = (0.05, 0.18),
        #     step_width= 0.3,   
        #     platform_width = 3.0,
        #     border_width = 1.0
        # ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range = (0.05, 0.18),
            step_width= 0.3,   
            platform_width = 3.0,
            border_width = 1.0
        ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.18),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        
        "pyramid_stairs_inv_right": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_rough_right": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25
        ),
        "trenches_right": terrain_gen.HfHorizontalTrenchesTerrainCfg(
            proportion=0.1, trench_width_range=(0.05, 0.15), num_trenches = 2.0, horizontal_scale=0.05),
        "horizontal_rails_right": terrain_gen.HfHorizontalRailsTerrainCfg(
            proportion=0.2, 
            rail_height_range=(0.04, 0.07), 
            rail_thickness=0.25, 
            num_rails=3,
            horizontal_scale=0.005,

        ),
        "hf_pyramid_slope_right": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv_right": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        )
    },
)

UNITREE_GO1_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "horizontal_rails": terrain_gen.HfHorizontalRailsTerrainCfg(
            proportion=1.0, 
            rail_height_range=(0.04, 0.04), 
            rail_thickness=0.2, 
            num_rails=5,
            horizontal_scale=0.005,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
