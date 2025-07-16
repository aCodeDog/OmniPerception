# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR sensor implementation extending the RayCaster."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.utils.math import quat_apply
from .ray_caster import RayCaster
from .lidar_sensor_data import LidarSensorData

if TYPE_CHECKING:
    from .lidar_sensor_cfg import LidarSensorCfg


class LidarSensor(RayCaster):
    """A LiDAR sensor implementation based on ray-casting.
    
    This sensor extends the base RayCaster to provide LiDAR-specific functionality
    including support for Livox sensors and traditional spinning LiDAR patterns.
    It includes features like dynamic ray pattern updates, point cloud generation,
    and noise simulation.
    """

    cfg: LidarSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: LidarSensorCfg):
        """Initializes the LiDAR sensor.

        Args:
            cfg: The configuration parameters.
        """
        # Initialize base class
        super().__init__(cfg)
        # Create LiDAR-specific data container
        self._data = LidarSensorData()
        
        # LiDAR-specific timing parameters
        self.update_frequency = cfg.update_frequency
        self.update_dt = 1.0 / self.update_frequency
        self.sensor_t = 0.0
        
        # Pattern parameters for dynamic updating
        self.pattern_start_index = 0
        
    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"LiDAR Sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tupdate frequency (Hz): {self.cfg.update_frequency}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\tmax range (m)        : {self.cfg.max_distance}\n"
            f"\tnoise enabled        : {self.cfg.enable_sensor_noise}"
        )

    @property
    def data(self) -> LidarSensorData:
        """The sensor data object."""
        # Update buffers if needed
        self._update_outdated_buffers()
        return self._data
    def _get_true_sensor_pos(self) -> torch.Tensor:
        """Calculates the true world position of the sensor, including the local offset."""
        # Get the stored base prim pose
        base_pos_w = self._data.pos_w
        base_quat_w = self._data.quat_w

        # Get the local offset from the config
        local_offset = torch.tensor(self.cfg.offset.pos, device=self.device)
        local_offset = local_offset.expand(base_pos_w.shape[0], -1)

        # Rotate the local offset to align with the base prim's world orientation
        world_offset = quat_apply(base_quat_w, local_offset)
        
        # The true sensor position is the base position plus the world-space offset
        return base_pos_w + world_offset
    def _initialize_rays_impl(self):
        """Initialize ray patterns for LiDAR sensor."""
        # Call parent implementation
        super()._initialize_rays_impl()
        
        # Initialize LiDAR-specific data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        
        # Initialize distance data
        self._data.distances = torch.zeros(self._view.count, self.num_rays, device=self._device)
        
        # Initialize point cloud data if needed
        if self.cfg.return_pointcloud:
            self._data.pointcloud = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update sensor buffers with LiDAR-specific processing."""
        # Update sensor time
        self.sensor_t += self.cfg.update_period
        
        # Update ray patterns if using dynamic patterns (e.g., Livox)
        if hasattr(self.cfg.pattern_cfg, 'sensor_type') and not self.cfg.pattern_cfg.use_simple_grid:
            self._update_dynamic_rays()
        
        # Call parent implementation for ray casting
        super()._update_buffers_impl(env_ids)
        
        
        sensor_pos = self._get_true_sensor_pos()[env_ids].unsqueeze(1)

        # Calculate distances
        hit_points = self._data.ray_hits_w[env_ids]
        distances = torch.norm(hit_points - sensor_pos, dim=2)
        
        
        # Handle out-of-range values
        inf_mask = torch.isinf(hit_points).any(dim=2)
        distances[inf_mask] = self.cfg.max_distance
        
        # Apply noise if enabled
        if self.cfg.enable_sensor_noise:
            distances = self._apply_noise(distances, env_ids)
        
        self._data.distances[env_ids] = distances
        
        # Generate point cloud if requested
        if self.cfg.return_pointcloud:
            self._generate_pointcloud(env_ids)

    def _update_dynamic_rays(self):
        """Update ray directions for dynamic patterns (e.g., Livox sensors)."""
        # For Livox sensors, we can simulate rotating/changing patterns
        # This is a simplified implementation - real Livox sensors have complex patterns
        
        if hasattr(self.cfg.pattern_cfg, 'samples'):
            # Rotate the pattern slightly to simulate Livox behavior
            rotation_angle = self.sensor_t * 0.1  # Slow rotation
            cos_rot = math.cos(rotation_angle)
            sin_rot = math.sin(rotation_angle)
            
            # Apply small rotation around Z-axis to all environments simultaneously
            # Get current x and y components of ray directions for all environments
            x_dirs = self.ray_directions[:, :, 0]  # Shape: (num_envs, num_rays)
            y_dirs = self.ray_directions[:, :, 1]  # Shape: (num_envs, num_rays)
            
            # Apply rotation matrix using vectorized operations
            rotated_x = x_dirs * cos_rot - y_dirs * sin_rot
            rotated_y = x_dirs * sin_rot + y_dirs * cos_rot
            
            # Update ray directions for all environments at once
            self.ray_directions[:, :, 0] = rotated_x
            self.ray_directions[:, :, 1] = rotated_y

    def _apply_noise(self, distances: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        """Apply noise to distance measurements."""
        # Apply Gaussian noise to distances
        noise = torch.randn_like(distances) * self.cfg.random_distance_noise
        distances_noisy = distances + noise
        
        # Apply dropout
        dropout_mask = torch.rand_like(distances) < self.cfg.pixel_dropout_prob
        distances_noisy[dropout_mask] = self.cfg.max_distance
        
        # Clamp to valid range
        distances_noisy = torch.clamp(distances_noisy, self.cfg.min_range, self.cfg.max_distance)
        
        return distances_noisy

    def _generate_pointcloud(self, env_ids: Sequence[int]):
        """Generate point cloud from ray hits."""
        if self.cfg.pointcloud_in_world_frame:
            # Point cloud in world coordinates
            self._data.pointcloud[env_ids] = self._data.ray_hits_w[env_ids].clone()
        else:
            # Point cloud in sensor frame - simplified approach
            # Since the ray directions are already in sensor frame, we can compute
            # distances and multiply by the original ray directions
            sensor_pos = self._data.pos_w[env_ids].unsqueeze(1)
            hit_points_world = self._data.ray_hits_w[env_ids]
            
            # Calculate distances
            distances = torch.norm(hit_points_world - sensor_pos, dim=2, keepdim=True)
            
            # Get original ray directions in sensor frame
            ray_directions_sensor = self.ray_directions[env_ids]
            
            # Generate point cloud in sensor frame
            pointcloud_sensor = ray_directions_sensor * distances
            
            # Handle infinite distances (no hits)
            inf_mask = torch.isinf(hit_points_world).any(dim=2, keepdim=True)
            pointcloud_sensor[inf_mask.expand_as(pointcloud_sensor)] = float('inf')
            
            self._data.pointcloud[env_ids] = pointcloud_sensor

    def get_distances(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get distance measurements for specified environments.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Distance measurements tensor of shape (num_envs, num_rays).
        """
        if env_ids is None:
            return self.data.distances
        return self.data.distances[env_ids]

    def get_pointcloud(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get point cloud data for specified environments.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Point cloud tensor of shape (num_envs, num_rays, 3).
        """
        if not self.cfg.return_pointcloud:
            raise ValueError("Point cloud generation is disabled. Set 'return_pointcloud=True' in config.")
        
        if env_ids is None:
            return self.data.pointcloud
        return self.data.pointcloud[env_ids]
