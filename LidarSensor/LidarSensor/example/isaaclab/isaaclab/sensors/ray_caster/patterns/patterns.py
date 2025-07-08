# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import patterns_cfg


def grid_pattern(cfg: patterns_cfg.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    The grid pattern is made from rays that are parallel to each other. They span a 2D grid in the sensor's
    local coordinates from ``(-length/2, -width/2)`` to ``(length/2, width/2)``, which is defined
    by the ``size = (length, width)`` and ``resolution`` parameters in the config.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.

    Raises:
        ValueError: If the ordering is not "xy" or "yx".
        ValueError: If the resolution is less than or equal to 0.
    """
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
    if cfg.resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")

    # resolve mesh grid indexing (note: torch meshgrid is different from numpy meshgrid)
    # check: https://github.com/pytorch/pytorch/issues/15301
    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    # define grid pattern
    x = torch.arange(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    y = torch.arange(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    # define ray-cast directions
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)

    return ray_starts, ray_directions


def pinhole_camera_pattern(
    cfg: patterns_cfg.PinholeCameraPatternCfg, intrinsic_matrices: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """The image pattern for ray casting.

    .. caution::
        This function does not follow the standard pattern interface. It requires the intrinsic matrices
        of the cameras to be passed in. This is because we want to be able to randomize the intrinsic
        matrices of the cameras, which is not possible with the standard pattern interface.

    Args:
        cfg: The configuration instance for the pattern.
        intrinsic_matrices: The intrinsic matrices of the cameras. Shape is (N, 3, 3).
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays. The shape of the tensors are
        (N, H * W, 3) and (N, H * W, 3) respectively.
    """
    # get image plane mesh grid
    grid = torch.meshgrid(
        torch.arange(start=0, end=cfg.width, dtype=torch.int32, device=device),
        torch.arange(start=0, end=cfg.height, dtype=torch.int32, device=device),
        indexing="xy",
    )
    pixels = torch.vstack(list(map(torch.ravel, grid))).T
    # convert to homogeneous coordinate system
    pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
    # move each pixel coordinate to the center of the pixel
    pixels += torch.tensor([[0.5, 0.5, 0]], device=device)
    # get pixel coordinates in camera frame
    pix_in_cam_frame = torch.matmul(torch.inverse(intrinsic_matrices), pixels.T)

    # robotics camera frame is (x forward, y left, z up) from camera frame with (x right, y down, z forward)
    # transform to robotics camera frame
    transform_vec = torch.tensor([1, -1, -1], device=device).unsqueeze(0).unsqueeze(2)
    pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec
    # normalize ray directions
    ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
    # for camera, we always ray-cast from the sensor's origin
    ray_starts = torch.zeros_like(ray_directions, device=device)

    return ray_starts, ray_directions


def bpearl_pattern(cfg: patterns_cfg.BpearlPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The RS-Bpearl pattern for ray casting.

    The `Robosense RS-Bpearl`_ is a short-range LiDAR that has a 360 degrees x 90 degrees super wide
    field of view. It is designed for near-field blind-spots detection.

    .. _Robosense RS-Bpearl: https://www.roscomponents.com/en/lidar-laser-scanner/267-rs-bpearl.html

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    h = torch.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res, device=device)
    v = torch.tensor(list(cfg.vertical_ray_angles), device=device)

    pitch, yaw = torch.meshgrid(v, h, indexing="xy")
    pitch, yaw = torch.deg2rad(pitch.reshape(-1)), torch.deg2rad(yaw.reshape(-1))
    pitch += torch.pi / 2
    x = torch.sin(pitch) * torch.cos(yaw)
    y = torch.sin(pitch) * torch.sin(yaw)
    z = torch.cos(pitch)

    ray_directions = -torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


def lidar_pattern(cfg: patterns_cfg.LidarPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # Vertical angles
    vertical_angles = torch.linspace(cfg.vertical_fov_range[0], cfg.vertical_fov_range[1], cfg.channels)

    # If the horizontal field of view is 360 degrees, exclude the last point to avoid overlap
    if abs(abs(cfg.horizontal_fov_range[0] - cfg.horizontal_fov_range[1]) - 360.0) < 1e-6:
        up_to = -1
    else:
        up_to = None

    # Horizontal angles
    num_horizontal_angles = math.ceil((cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]) / cfg.horizontal_res)
    horizontal_angles = torch.linspace(cfg.horizontal_fov_range[0], cfg.horizontal_fov_range[1], num_horizontal_angles)[
        :up_to
    ]

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = torch.deg2rad(horizontal_angles)

    # Meshgrid to create a 2D array of angles
    v_angles, h_angles = torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing="ij")

    # Spherical to Cartesian conversion using LidarSensor convention
    # theta = horizontal angle (azimuth), phi = vertical angle (elevation)
    cos_theta = torch.cos(h_angles)
    sin_theta = torch.sin(h_angles)
    cos_phi = torch.cos(v_angles)
    sin_phi = torch.sin(v_angles)
    
    # LidarSensor coordinate system: x=forward, y=left, z=up
    x = cos_theta * cos_phi  # forward component
    y = sin_theta * cos_phi  # left component
    z = sin_phi              # up component

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


def livox_pattern(cfg: patterns_cfg.LivoxPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Livox LiDAR sensor pattern for ray casting.

    This function generates ray patterns that mimic real Livox sensor behavior.
    It can either use predefined scan patterns or fall back to a simple grid pattern.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    if cfg.use_simple_grid:
        # Use simple grid pattern as fallback
        return _livox_simple_grid_pattern(cfg, device)
    else:
        # Use Livox-specific scan pattern
        return _livox_scan_pattern(cfg, device)


def _livox_simple_grid_pattern(cfg: patterns_cfg.LivoxPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate simple grid pattern for Livox sensor."""
    # Convert FOV to radians
    horizontal_fov_min = math.radians(cfg.horizontal_fov_deg_min)
    horizontal_fov_max = math.radians(cfg.horizontal_fov_deg_max)
    vertical_fov_min = math.radians(cfg.vertical_fov_deg_min)
    vertical_fov_max = math.radians(cfg.vertical_fov_deg_max)
    
    ray_directions = torch.zeros(
        (cfg.vertical_line_num, cfg.horizontal_line_num, 3), 
        dtype=torch.float32, 
        device=device
    )
    
    for i in range(cfg.vertical_line_num):
        for j in range(cfg.horizontal_line_num):
            # Calculate angles
            if cfg.vertical_line_num > 1:
                vertical_angle = vertical_fov_min + (vertical_fov_max - vertical_fov_min) * i / (cfg.vertical_line_num - 1)
            else:
                vertical_angle = (vertical_fov_min + vertical_fov_max) / 2
                
            if cfg.horizontal_line_num > 1:
                horizontal_angle = horizontal_fov_min + (horizontal_fov_max - horizontal_fov_min) * j / (cfg.horizontal_line_num - 1)
            else:
                horizontal_angle = (horizontal_fov_min + horizontal_fov_max) / 2
            
            # Convert spherical to cartesian coordinates using LidarSensor convention
            # theta = horizontal_angle (azimuth), phi = vertical_angle (elevation)
            cos_theta = math.cos(horizontal_angle)
            sin_theta = math.sin(horizontal_angle)
            cos_phi = math.cos(vertical_angle)
            sin_phi = math.sin(vertical_angle)
            
            # LidarSensor coordinate system: x=forward, y=left, z=up
            x = cos_theta * cos_phi  # forward component
            y = sin_theta * cos_phi  # left component  
            z = sin_phi              # up component
            
            ray_directions[i, j] = torch.tensor([x, y, z], device=device)
    
    # Flatten the ray directions
    ray_directions = ray_directions.reshape(-1, 3)
    
    # Normalize directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    # Ray starts at origin
    ray_starts = torch.zeros_like(ray_directions)
    
    return ray_starts, ray_directions


def _livox_scan_pattern(cfg: patterns_cfg.LivoxPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate realistic Livox scan pattern based on sensor type.
    
    This function loads precomputed scan patterns from .npy files for accurate Livox sensor emulation.
    The pattern files contain [theta, phi] angles in radians corresponding to the actual scan patterns
    captured from real Livox sensors.
    """
    import numpy as np
    import os
    
    # Define Livox sensor parameters and pattern file mapping
    livox_params = {
        "avia": {"horizontal_fov": 70.4, "vertical_fov": 77.2, "samples": 24000, "pattern_file": "avia.npy"},
        "horizon": {"horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000, "pattern_file": "horizon.npy"},
        "HAP": {"horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 45300, "pattern_file": "HAP.npy"},
        "mid360": {"horizontal_fov": 360, "vertical_fov": 59, "samples": 20000, "pattern_file": "mid360.npy"},
        "mid40": {"horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000, "pattern_file": "mid40.npy"},
        "mid70": {"horizontal_fov": 70.4, "vertical_fov": 70.4, "samples": 10000, "pattern_file": "mid70.npy"},
        "tele": {"horizontal_fov": 14.5, "vertical_fov": 16.1, "samples": 24000, "pattern_file": "tele.npy"}
    }
    
    if cfg.sensor_type not in livox_params:
        raise ValueError(f"Unsupported Livox sensor type: {cfg.sensor_type}")
    
    params = livox_params[cfg.sensor_type]
    
    # Try to load precomputed scan pattern
    pattern_file = params["pattern_file"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use unified scan patterns location in sensor_pattern/sensor_lidar/scan_mode
    # Navigate from patterns.py location to the unified patterns directory
    omniperc_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))))
    unified_patterns_dir = os.path.join(omniperc_root, "LidarSensor", "LidarSensor", "sensor_pattern", "sensor_lidar", "scan_mode")
    pattern_path = os.path.join(unified_patterns_dir, pattern_file)
    
    # Fallback to local scan_patterns if unified location doesn't exist
    if not os.path.exists(pattern_path):
        pattern_path = os.path.join(script_dir, "scan_patterns", pattern_file)
    
    if os.path.exists(pattern_path):
        # Load precomputed pattern from .npy file
        pattern_data = np.load(pattern_path)  # Shape: (N, 2) where columns are [theta, phi]
        
        # Get rolling window sample based on cfg.start_index for temporal consistency
        total_pattern_size = pattern_data.shape[0]
        samples = min(cfg.samples, total_pattern_size)
        
        # Implement rolling window sampling like the original LivoxGenerator
        start_idx = cfg.rolling_window_start % total_pattern_size
        
        if start_idx + samples <= total_pattern_size:
            # Simple case: no wraparound needed
            selected_pattern = pattern_data[start_idx:start_idx + samples]
        else:
            # Wraparound case: need to sample from end and beginning
            end_samples = total_pattern_size - start_idx
            begin_samples = samples - end_samples
            selected_pattern = np.vstack([
                pattern_data[start_idx:],
                pattern_data[:begin_samples]
            ])
        
        # Convert to torch tensors
        theta = torch.from_numpy(selected_pattern[:, 0]).to(device)  # horizontal angles
        phi = torch.from_numpy(selected_pattern[:, 1]).to(device)    # vertical angles
        
    else:
        # Fallback to random pattern generation if .npy file not found
        print(f"Warning: Pattern file {pattern_path} not found. Using random pattern generation.")
        samples = min(cfg.samples, params["samples"])
        
        # Generate random angles within FOV using LidarSensor convention
        h_fov = math.radians(params["horizontal_fov"])
        v_fov = math.radians(params["vertical_fov"])
        
        # Generate uniform distribution for Livox pattern
        torch.manual_seed(42)  # For reproducible patterns
        theta = (torch.rand(samples, device=device) - 0.5) * h_fov  # horizontal angles
        phi = (torch.rand(samples, device=device) - 0.5) * v_fov    # vertical angles
    
    # Apply downsampling if requested
    if cfg.downsample > 1:
        indices = torch.arange(0, len(theta), cfg.downsample, device=device)
        theta = theta[indices]
        phi = phi[indices]
    
    # Convert to Cartesian coordinates using LidarSensor convention
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    # LidarSensor coordinate system: x=forward, y=left, z=up
    x = cos_theta * cos_phi  # forward component
    y = sin_theta * cos_phi  # left component
    z = sin_phi              # up component
    
    ray_directions = torch.stack([x, y, z], dim=1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    # Ray starts at origin
    ray_starts = torch.zeros_like(ray_directions)
    
    return ray_starts, ray_directions
