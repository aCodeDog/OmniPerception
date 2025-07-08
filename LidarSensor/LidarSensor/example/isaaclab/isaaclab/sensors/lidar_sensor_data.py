# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data class for the LiDAR sensor."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ray_caster import RayCasterData

if TYPE_CHECKING:
    pass


@dataclass
class LidarSensorData(RayCasterData):
    """Data container for the LiDAR sensor.
    
    This data container extends RayCasterData to include LiDAR-specific measurements
    like distances and point clouds.
    """

    distances: torch.Tensor = None
    """Distance measurements for each ray in meters. Shape is (num_instances, num_rays)."""
    
    pointcloud: torch.Tensor = None
    """Point cloud data in either world or sensor coordinates. Shape is (num_instances, num_rays, 3).
    
    Note:
        This is only populated if ``return_pointcloud`` is True in the sensor configuration.
        The coordinate frame depends on the ``pointcloud_in_world_frame`` setting.
    """
