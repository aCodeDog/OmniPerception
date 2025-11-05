from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
import re
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

try:
    import fpsample
except ImportError:
    fpsample = None
    print("Warning: fpsample gpu version not available. Install with: pip install git+https://github.com/aCodeDog/pytorch_fpsample.git")

# Added fast batched FPS sampler
try:
    import torch_fpsample
except ImportError:
    torch_fpsample = None
    print("Warning: torch_fpsample not available. Install with: pip install torch-fpsample")

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv




class SphereRayCaster(RayCaster):
    """A sphere ray-caster that converts hit points to spherical coordinates and visualizes based on phi values.
    
    This sensor extends the regular RayCaster to provide spherical coordinate conversion and 
    phi-based color visualization of ray hit points.
    """
    
    cfg: SphereRayCasterCfg
    """The configuration parameters."""
    
    def __init__(self, cfg: SphereRayCasterCfg):
        """Initialize the sphere ray caster.
        
        Args:
            cfg: The configuration parameters.
        """
        # Initialize visualization markers BEFORE calling parent __init__
        self.red_visualizer = VisualizationMarkers(cfg.visualizer_cfg_red)
        self.blue_visualizer = VisualizationMarkers(cfg.visualizer_cfg_blue)
        
        # Store spherical coordinates - initialized after parent's _initialize_impl
        self._spherical_coords = None
        
        # Now call parent initialization
        super().__init__(cfg)
    
    def _initialize_impl(self):
        """Initialize implementation - called after parent initialization."""
        super()._initialize_impl()
    
    def _initialize_rays_impl(self):
        """Initialize rays implementation - called after _initialize_impl."""
        super()._initialize_rays_impl()
        # Now we can safely access self._view.count and self.num_rays
        if self._view is not None:
            self._spherical_coords = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        else:
            raise RuntimeError("View not initialized properly in SphereRayCaster.")
    
    @property
    def spherical_coords(self) -> torch.Tensor:
        """Get spherical coordinates of ray hit points.
        
        Returns:
            Tensor of shape (N, B, 3) with [r, theta, phi] for each ray hit.
        """
        if self._spherical_coords is None:
            raise RuntimeError("SphereRayCaster not initialized. Call initialize() first.")
        return self._spherical_coords
    
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update buffers and compute spherical coordinates."""
        # Call parent to update ray hit data
        super()._update_buffers_impl(env_ids)
        
        # Convert ray hits to spherical coordinates
        self._compute_spherical_coordinates(env_ids)
    
    def _compute_spherical_coordinates(self, env_ids: Sequence[int]):
        """Convert ray hit points from world to sensor frame and then to spherical coordinates.
        
        Args:
            env_ids: Environment indices to update.
        """
        if self._spherical_coords is None:
            return
            
        # Get sensor pose and ray hits
        sensor_pos_w = self._data.pos_w[env_ids]  # (N, 3)
        sensor_quat_w = self._data.quat_w[env_ids]  # (N, 4)  
        ray_hits_w = self._data.ray_hits_w[env_ids]  # (N, B, 3)
        
        # Transform to sensor frame
        ray_hits_sensor_frame = ray_hits_w - sensor_pos_w.unsqueeze(1)  # (N, B, 3)
        
        # Expand quaternion and apply inverse rotation
        sensor_quat_expanded = sensor_quat_w.unsqueeze(1).expand(-1, ray_hits_sensor_frame.shape[1], -1)  # (N, B, 4)
        ray_hits_sensor_base = math_utils.quat_apply_inverse(
            sensor_quat_expanded, ray_hits_sensor_frame
        )  # (N, B, 3)
        
        # Convert to spherical coordinates
        self._spherical_coords[env_ids] = self._cart2sphere(ray_hits_sensor_base)
    
    def _cart2sphere(self, cart: torch.Tensor) -> torch.Tensor:
        """Convert Cartesian to spherical coordinates.
        
        Args:
            cart: Cartesian coordinates (..., 3) with [x, y, z]
            
        Returns:
            Spherical coordinates (..., 3) with [r, theta, phi]
        """
        epsilon = 1e-9
        x = cart[..., 0]
        y = cart[..., 1] 
        z = cart[..., 2]
        r = torch.norm(cart, dim=-1)
        theta = torch.atan2(y, x)
        phi = torch.asin(z / (r + epsilon))
        return torch.stack((r, theta, phi), dim=-1)
    
    def _debug_vis_callback(self, event):
        """Visualize ray hit points with colors based on phi values."""
        if not self.cfg.debug_vis or self._spherical_coords is None:
            return
            
        # Check if visualizers are initialized
        if not (hasattr(self, 'red_visualizer') and hasattr(self, 'blue_visualizer')):
            return
            
        # Get all ray hit points in world frame
        viz_points = self._data.ray_hits_w.reshape(-1, 3)  # (N*B, 3)
        spherical_all = self._spherical_coords.reshape(-1, 3)  # (N*B, 3)
        
        # Remove inf values
        valid_mask = ~torch.any(torch.isinf(viz_points), dim=1)
        viz_points = viz_points[valid_mask]
        spherical_valid = spherical_all[valid_mask]
        
        if len(viz_points) == 0:
            return
            
        # Get phi values
        phi_values = spherical_valid[:, 2]  # Extract phi column
        
        # Separate points based on phi threshold
        red_mask = phi_values > self.cfg.phi_threshold
        blue_mask = ~red_mask
        
        # Visualize red points (phi > threshold)
        if torch.any(red_mask):
            red_points = viz_points[red_mask]
            self.red_visualizer.visualize(red_points)
        else:
            # Clear red visualization if no points
            self.red_visualizer.visualize(torch.empty(0, 3, device=self._device))
            
        # Visualize blue points (phi <= threshold)  
        if torch.any(blue_mask):
            blue_points = viz_points[blue_mask]
            self.blue_visualizer.visualize(blue_points)
        else:
            # Clear blue visualization if no points
            self.blue_visualizer.visualize(torch.empty(0, 3, device=self._device))
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization state."""
        super()._set_debug_vis_impl(debug_vis)
        
        # Check if our visualizers are initialized
        if hasattr(self, 'red_visualizer') and hasattr(self, 'blue_visualizer'):
            # Set visibility of our custom markers
            if debug_vis:
                # Enable visualization for both red and blue markers
                self.red_visualizer.set_visibility(True)
                self.blue_visualizer.set_visibility(True)
            else:
                self.red_visualizer.set_visibility(False)
                self.blue_visualizer.set_visibility(False)

@configclass
class SphereRayCasterCfg(RayCasterCfg):
    """Configuration for the sphere ray-cast sensor with phi-based visualization."""
    
    phi_threshold: float = 0.0
    """Phi threshold for color visualization. Points with phi > threshold are red, others are blue."""
    
    visualizer_cfg_red: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SphereRayCaster/Red",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    """Visualization config for points with phi > threshold."""
    
    visualizer_cfg_blue: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SphereRayCaster/Blue", 
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """Visualization config for points with phi <= threshold."""
    class_type: type = SphereRayCaster
    
    
def sphere_raycaster_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get spherical coordinates from the sphere ray caster sensor.

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration.

    Returns:
        Tensor of shape (N, B, 3) with [r, theta, phi] for each ray hit.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ray_hits_w

def raycaster_scan_dist(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get spherical coordinates from the sphere ray caster sensor.

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration.

    Returns:
        Tensor of shape (N, B, 3) with [r, theta, phi] for each ray hit.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ray_hits_distance
    #return sensor.spherical_coords
    # if isinstance(sensor, SphereRayCaster):
    #     return sensor.spherical_coords
    # else:
    #     raise RuntimeError(f"Sensor {sensor_cfg.name} is not a SphereRayCaster. Use SphereRayCasterCfg instead of RayCasterCfg.")


def proximal_points_fps(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, num_points: int = 256) -> torch.Tensor:
    """Get proximal points (phi <= threshold) using fast batched FPS (torch_fpsample) without loops.

    Steps (all vectorized, no Python loops over batch):
    1. Build proximal mask (phi <= threshold & valid).
    2. Transform hits to sensor frame.
    3. Pack variable-length per-batch point sets into padded (N, M_max, 3) via scatter.
    4. Duplicate first valid point to fill padding (avoids FPS degeneracy) while later zero-masking invalid samples.
    5. Run torch_fpsample.sample on padded tensor (with k = min(num_points, M_max)).
    6. Zero-out sampled points whose indices map to padding and pad (if needed) to requested num_points.
    """
    if torch_fpsample is None:
        raise ImportError("torch_fpsample not installed. Install with: pip install torch-fpsample")

    sensor = env.scene.sensors[sensor_cfg.name]
    if not isinstance(sensor, SphereRayCaster):
        raise RuntimeError(f"Sensor {sensor_cfg.name} is not a SphereRayCaster.")

    # Data
    sensor_pos_w = sensor.data.pos_w          # (N, 3)
    sensor_quat_w = sensor.data.quat_w        # (N, 4)
    ray_hits_w = sensor.data.ray_hits_w       # (N, B, 3)
    spherical_coords = sensor.spherical_coords  # (N, B, 3)

    batch_size, B = ray_hits_w.shape[:2]
    device = ray_hits_w.device

    valid_mask = ~torch.any(torch.isinf(ray_hits_w), dim=2)  # (N, B)
    phi_values = spherical_coords[:, :, 2]
    proximal_mask = (phi_values <= sensor.cfg.phi_threshold) & valid_mask  # (N, B)

    counts = proximal_mask.sum(dim=1)  # (N,)
    max_count_tensor = counts.max()    # scalar tensor
    if max_count_tensor.item() == 0:
        return torch.zeros(batch_size, num_points, 3, device=device)
    max_count = int(max_count_tensor.item())

    # Transform to sensor frame
    ray_hits_sensor_frame = ray_hits_w - sensor_pos_w.unsqueeze(1)  # (N, B, 3)
    sensor_quat_expanded = sensor_quat_w.unsqueeze(1).expand(-1, B, -1)  # (N, B, 4)
    hits_sensor_base = math_utils.quat_apply_inverse(sensor_quat_expanded, ray_hits_sensor_frame)  # (N, B, 3)

    # Positions (0..count_i-1) for proximal points
    cumsum = proximal_mask.cumsum(dim=1) - 1  # (N, B)
    positions = torch.where(proximal_mask, cumsum, torch.zeros_like(cumsum))  # (N, B)

    # Allocate padded tensor
    padded = torch.zeros(batch_size, max_count, 3, device=device)

    # Scatter proximal points
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, B)  # (N, B)
    valid_batch_indices = batch_indices[proximal_mask]      # (total_valid,)
    valid_positions = positions[proximal_mask]              # (total_valid,)
    valid_points = hits_sensor_base[proximal_mask]          # (total_valid, 3)
    padded[valid_batch_indices, valid_positions] = valid_points

    # Fill padding slots with first point (or zeros if none) to keep distances finite
    arange_max = torch.arange(max_count, device=device)  # (max_count,)
    fill_mask = arange_max.unsqueeze(0) >= counts.unsqueeze(1)  # (N, max_count)
    if fill_mask.any():
        first_points = padded[:, 0:1, :].expand(-1, max_count, -1)
        padded[fill_mask] = first_points[fill_mask]

    # Effective k for FPS (cannot exceed max_count)
    k_eff = num_points if num_points <= max_count else max_count

    sampled_points_eff, sampled_indices_eff = torch_fpsample.sample(padded, k_eff)  # (N, k_eff, 3), (N, k_eff)

    # Mask out samples that correspond to padding (indices >= count_i)
    counts_exp = counts.unsqueeze(1)  # (N,1)
    invalid_samples = sampled_indices_eff >= counts_exp  # (N, k_eff)
    if invalid_samples.any():
        sampled_points_eff = sampled_points_eff.masked_fill(invalid_samples.unsqueeze(-1), 0.0)

    # If k_eff < num_points, pad remaining with zeros
    if k_eff < num_points:
        out = torch.zeros(batch_size, num_points, 3, device=device)
        out[:, :k_eff, :] = sampled_points_eff
        return out
    return sampled_points_eff

def distal_points_average(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, num_points: int = 128) -> torch.Tensor:
    """Get distal points (phi > threshold) using spherical sorting and vectorized downsampling.

    Vectorized implementation (no Python loop over batch):
    - Build distal mask (phi > threshold & valid).
    - Sort by combined angular key (phi primary, theta secondary).
    - For batches with more than num_points valid points, pick uniformly spaced indices in the sorted order.
    - For batches with <= num_points valid points, keep the first k sorted points and zero-pad the rest.
    - Convert selected spherical [r, theta, phi] back to Cartesian in the sensor frame.

    Returns: (N, num_points, 3)
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if not isinstance(sensor, SphereRayCaster):
        raise RuntimeError(f"Sensor {sensor_cfg.name} is not a SphereRayCaster.")

    # Data
    sensor_pos_w = sensor.data.pos_w           # (N, 3)  (not required for vectorized path; kept for consistency)
    sensor_quat_w = sensor.data.quat_w         # (N, 4)
    ray_hits_w = sensor.data.ray_hits_w        # (N, B, 3)
    spherical_coords = sensor.spherical_coords # (N, B, 3) in sensor frame

    N, B = ray_hits_w.shape[:2]
    device = ray_hits_w.device

    # Valid hits and distal mask
    valid_mask = ~torch.any(torch.isinf(ray_hits_w), dim=2)  # (N, B)
    theta = spherical_coords[:, :, 1]
    phi = spherical_coords[:, :, 2]
    distal_mask = (phi > sensor.cfg.phi_threshold) & valid_mask  # (N, B)

    counts = distal_mask.sum(dim=1)  # (N,)
    if (counts == 0).all():
        return torch.zeros(N, num_points, 3, device=device)

    # Sort by combined angular key: phi primary, theta secondary
    # Non-distal entries get +inf to push them to the end
    sort_key = torch.where(
        distal_mask,
        phi * (2 * torch.pi + 1.0) + theta,
        torch.full_like(phi, torch.inf),
    )  # (N, B)
    sorted_indices = torch.argsort(sort_key, dim=1)  # (N, B)

    # Determine sampling positions per batch
    M = num_points
    pos = torch.arange(M, device=device).unsqueeze(0).expand(N, -1)  # (N, M) -> [0..M-1]
    counts_clamped = counts.clamp(min=1)

    # Uniform positions for downsampling when counts > M
    denom = max(M - 1, 1)
    uniform_pos = torch.round(
        pos.float() * (counts_clamped.unsqueeze(1).float() - 1.0) / float(denom)
    ).long()  # (N, M) in [0, counts_i-1]

    # Sequential positions for the case counts <= M
    seq_pos = pos  # (N, M); we'll mask out positions >= counts later

    use_uniform = (counts > M).unsqueeze(1)  # (N, 1)
    positions = torch.where(use_uniform, uniform_pos, seq_pos)
    positions = torch.minimum(positions, (counts_clamped - 1).unsqueeze(1))  # clamp to valid

    # Map positions to original ray indices via the sorted order
    selected_sorted_idx = torch.gather(sorted_indices, 1, positions)  # (N, M)

    # Gather spherical coords of the selected points: (N, M, 3)
    idx_expanded = selected_sorted_idx.unsqueeze(-1).expand(-1, -1, 3)
    selected_spherical = torch.gather(spherical_coords, 1, idx_expanded)

    # Convert spherical [r, theta, phi] back to Cartesian in sensor frame
    r_sel = selected_spherical[..., 0]
    theta_sel = selected_spherical[..., 1]
    phi_sel = selected_spherical[..., 2]
    x = r_sel * torch.cos(phi_sel) * torch.cos(theta_sel)
    y = r_sel * torch.cos(phi_sel) * torch.sin(theta_sel)
    z = r_sel * torch.sin(phi_sel)
    out = torch.stack((x, y, z), dim=-1)  # (N, M, 3)

    # Zero-out columns beyond available counts when counts < M (pad with zeros)
    out_len = torch.minimum(counts, torch.tensor(M, device=device))  # (N,)
    keep_mask = pos < out_len.unsqueeze(1)  # (N, M)
    out = out * keep_mask.unsqueeze(-1)

    return out
        

def cart2sphere(cart):
    """Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        cart: Cartesian coordinates tensor of shape (..., 3) with [x, y, z]
        
    Returns:
        Spherical coordinates tensor of shape (..., 3) with [r, theta, phi]
        where:
        - r: distance from origin
        - theta: azimuthal angle (atan2(y, x))
        - phi: elevation angle (asin(z/r))
    """
    epsilon = 1e-9
    x = cart[..., 0]
    y = cart[..., 1]
    z = cart[..., 2]
    r = torch.norm(cart, dim=-1)
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)


def raycaster_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Ray cast scan from the given sensor w.r.t. the sensor's frame converted to spherical coordinates.

    This function transforms ray hit points from world coordinates to sensor-base coordinates,
    then converts them to spherical coordinates.
    """
    # extract the used quantities (to enable type-hinting)
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get sensor pose in world frame
    sensor_pos_w = sensor.data.pos_w  # Shape: (N, 3) - sensor position in world frame
    sensor_quat_w = sensor.data.quat_w  # Shape: (N, 4) - sensor orientation (w,x,y,z) in world frame
    ray_hits_w = sensor.data.ray_hits_w  # Shape: (N, B, 3) - ray hit points in world frame
    ray_hits_dist = sensor.data.ray_hits_distance  # Shape: (N, B) - ray hit distances in world frame
    ray_hits_normal = sensor.data.ray_hits_normal  # Shape: (N, B, 3) - ray hit normals in world frame
    # Transform ray hits from world coordinates to sensor-base coordinates
    # First, translate points to sensor origin
    ray_hits_sensor_frame = ray_hits_w - sensor_pos_w.unsqueeze(1)  # Shape: (N, B, 3)
    
    # Then rotate points to sensor-base frame using inverse quaternion
    # Need to expand quaternion to match ray hits shape
    sensor_quat_expanded = sensor_quat_w.unsqueeze(1).expand(-1, ray_hits_sensor_frame.shape[1], -1)  # Shape: (N, B, 4)
    
    # Apply inverse quaternion rotation to transform from world frame to sensor frame
    ray_hits_sensor_base = math_utils.quat_apply_inverse(
        sensor_quat_expanded, ray_hits_sensor_frame
    )  # Shape: (N, B, 3)
    
    # Convert sensor-base Cartesian coordinates to spherical coordinates
    spherical_coords = cart2sphere(ray_hits_sensor_base)  # Shape: (N, B, 3) with [r, theta, phi]
    
    return spherical_coords
