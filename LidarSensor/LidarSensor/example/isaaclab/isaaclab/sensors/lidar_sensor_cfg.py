# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiDAR sensor."""

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

from .ray_caster import RayCasterCfg
from .lidar_sensor import LidarSensor


@configclass
class LidarSensorCfg(RayCasterCfg):
    """Configuration for the LiDAR sensor.
    
    This configuration extends RayCasterCfg to add LiDAR-specific parameters
    like noise simulation, point cloud generation, and sensor-specific settings.
    """

    class_type: type = LidarSensor
    ray_alignment: Literal["base", "yaw", "world"] = "yaw"
    # LiDAR-specific timing parameters
    update_frequency: float = 50.0
    """LiDAR update frequency in Hz. Defaults to 50.0 Hz."""
    
    # Range settings
    min_range: float = 0.2
    """Minimum sensing range in meters. Defaults to 0.2m."""
    
    # Output settings
    return_pointcloud: bool = True
    """Whether to generate point cloud data. Defaults to True."""
    
    pointcloud_in_world_frame: bool = False
    """Whether to return point cloud in world frame or sensor frame. Defaults to False (sensor frame)."""
    
    # Noise settings
    enable_sensor_noise: bool = False
    """Whether to enable sensor noise simulation. Defaults to False."""
    
    random_distance_noise: float = 0.03
    """Standard deviation of Gaussian noise added to distances. Defaults to 0.03m."""
    
    random_angle_noise: float = 0.15 * 3.14159 / 180
    """Standard deviation of angular noise in radians. Defaults to 0.15 degrees."""
    
    pixel_dropout_prob: float = 0.01
    """Probability of pixel dropout (no return). Defaults to 0.01."""
    
    pixel_std_dev_multiplier: float = 0.01
    """Multiplier for pixel-wise noise standard deviation. Defaults to 0.01."""
    
    # Data normalization
    normalize_range: bool = False
    """Whether to normalize range values. Defaults to False."""
    
    far_out_of_range_value: float = -1.0
    """Value to assign to far out-of-range measurements. Defaults to -1.0."""
    
    near_out_of_range_value: float = -1.0
    """Value to assign to near out-of-range measurements. Defaults to -1.0."""
