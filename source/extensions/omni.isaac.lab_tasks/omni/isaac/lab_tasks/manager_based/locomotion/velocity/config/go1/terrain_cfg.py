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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.1), noise_step=0.01, border_width=0.25
        ),
        # "wave": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.1, amplitude_range=(0.01, 0.33), num_waves = 2.0
        # ),
        "horizontal_rails": terrain_gen.HfHorizontalRailsTerrainCfg(
            proportion=0.2, 
            rail_height_range=(0.05, 0.1), 
            rail_thickness=0.2, 
            num_rails=3,
            horizontal_scale=0.005,

        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
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
        # "wave": terrain_gen.HfWaveTerrainCfg(
        #     proportion=1.0, amplitude_range=(0.05, 0.33), num_waves = 2.0
        # )
    },
)
"""Rough terrains configuration."""
