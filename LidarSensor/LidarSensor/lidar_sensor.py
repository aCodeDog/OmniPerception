import torch
import math
import warp as wp

# Handle both relative and absolute imports
try:
    from .sensor_kernels.lidar_kernels_warp import LidarWarpKernels
    from .base_sensor import BaseSensor
    from .sensor_config.lidar_sensor_config import LidarConfig, LidarType
    from .sensor_pattern.sensor_lidar.genera_lidar_scan_pattern import (
        LidarRayGeneratorFactory, 
        LivoxGenerator, 
        SpinningLidarGenerator
    )
except ImportError:
    # Fallback to absolute imports
    from sensor_kernels.lidar_kernels_warp import LidarWarpKernels
    from base_sensor import BaseSensor
    from sensor_config.lidar_sensor_config import LidarConfig, LidarType
    from sensor_pattern.sensor_lidar.genera_lidar_scan_pattern import (
        LidarRayGeneratorFactory, 
        LivoxGenerator, 
        SpinningLidarGenerator
    )


class LidarSensor(BaseSensor):
    """Optimized LidarSensor with unified ray generation for all sensor types"""
    
    def __init__(self, env, env_cfg, sensor_config: LidarConfig, num_sensors=1, device='cuda:0'):
        self.env = env
        self.env_cfg = env_cfg
        self.sensor_cfg = sensor_config
        
        # Initialize config if needed
        if hasattr(self.sensor_cfg, '__post_init__'):
            self.sensor_cfg.__post_init__()
        
        # Timing configuration
        self.env_dt = self.sensor_cfg.dt
        self.update_frequency = self.sensor_cfg.update_frequency
        self.update_dt = 1 / self.update_frequency
        self.sensor_t = 0
        
        # Basic sensor parameters
        self.num_sensors = num_sensors
        self.device = device
        self.num_envs = self.env['num_envs']
        self.mesh_ids = self.env['mesh_ids']
        self.far_plane = self.sensor_cfg.max_range
        self.pointcloud_in_world_frame = self.sensor_cfg.pointcloud_in_world_frame
        
        # Validate sensor positions
        assert self.env['sensor_pos_tensor'] is not None
        assert self.env['sensor_quat_tensor'] is not None
        self.lidar_positions_tensor = self.env['sensor_pos_tensor']
        self.lidar_quat_tensor = self.env['sensor_quat_tensor']
        
        # Initialize ray generation
        self._setup_ray_generation()
        
        # Initialize warp and tensors
        wp.init()
        self.lidar_positions = None
        self.lidar_quat_array = None
        self.graph = None
        
        self.initialize_ray_vectors()
        self.init_tensors()
    
    def _setup_ray_generation(self):
        """Setup ray generation based on sensor type"""
        sensor_type_str = self.sensor_cfg.sensor_type.value
        
        if self.sensor_cfg.is_simple_grid:
            self._setup_grid_parameters()
            self.ray_generator = None
        elif self.sensor_cfg.is_height_scanner:
            self._setup_height_scanner_parameters()
            self.ray_generator = None
        elif self.sensor_cfg.is_livox_sensor:
            self.ray_generator = LivoxGenerator(sensor_type_str)
            # For Livox sensors, dimensions will be set in initialize_ray_vectors
            self.num_vertical_lines = 1  # Temporary value, will be updated
            self.num_horizontal_lines = 1  # Temporary value, will be updated
        elif self.sensor_cfg.is_spinning_lidar:
            self.ray_generator = SpinningLidarGenerator()
            # For spinning lidars, dimensions will be set in initialize_ray_vectors
            self.num_vertical_lines = 1  # Temporary value, will be updated
            self.num_horizontal_lines = 1  # Temporary value, will be updated
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type_str}")
    
    def _setup_grid_parameters(self):
        """Setup parameters for simple grid-based lidar"""
        self.num_vertical_lines = self.sensor_cfg.vertical_line_num
        self.num_horizontal_lines = self.sensor_cfg.horizontal_line_num
        
        # Convert FOV to radians
        self.horizontal_fov_min = math.radians(self.sensor_cfg.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.sensor_cfg.horizontal_fov_deg_max)
        self.vertical_fov_min = math.radians(self.sensor_cfg.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.sensor_cfg.vertical_fov_deg_max)
        
        # Validate FOV ranges
        horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min
        vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        
        if horizontal_fov > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2π")
        if vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than π")

    def _setup_height_scanner_parameters(self):
        """Setup parameters for height scanner"""
        # Validate configuration
        if self.sensor_cfg.height_scanner_resolution <= 0:
            raise ValueError(f"Height scanner resolution must be greater than 0. Received: {self.sensor_cfg.height_scanner_resolution}")
        
        if self.sensor_cfg.height_scanner_ordering not in ["xy", "yx"]:
            raise ValueError(f"Height scanner ordering must be 'xy' or 'yx'. Received: '{self.sensor_cfg.height_scanner_ordering}'")
        
        # Calculate grid dimensions
        size_x, size_y = self.sensor_cfg.height_scanner_size
        resolution = self.sensor_cfg.height_scanner_resolution
        
        # Calculate number of points in each dimension
        num_x = int(size_x / resolution) + 1
        num_y = int(size_y / resolution) + 1
        
        # Set grid dimensions based on ordering
        if self.sensor_cfg.height_scanner_ordering == "xy":
            self.num_horizontal_lines = num_x
            self.num_vertical_lines = num_y
        else:  # "yx"
            self.num_horizontal_lines = num_y
            self.num_vertical_lines = num_x
        
        # Store grid parameters for ray generation
        self.height_scanner_size = self.sensor_cfg.height_scanner_size
        self.height_scanner_resolution = self.sensor_cfg.height_scanner_resolution
        self.height_scanner_direction = self.sensor_cfg.height_scanner_direction
        self.height_scanner_ordering = self.sensor_cfg.height_scanner_ordering

    def initialize_ray_vectors(self):
        """Initialize ray vectors based on sensor type"""
        if self.sensor_cfg.is_simple_grid:
            self._initialize_grid_rays()
        elif self.sensor_cfg.is_height_scanner:
            self._initialize_height_scanner_rays()
        else:
            self._initialize_pattern_rays()
            
        print(f"Ray vectors initialized for {self.sensor_cfg.sensor_type.value} "
              f"with shape ({self.num_vertical_lines}, {self.num_horizontal_lines})")

    def _initialize_grid_rays(self):
        """Initialize ray vectors for simple grid-based lidar"""
        ray_vectors = torch.zeros(
            (self.num_vertical_lines, self.num_horizontal_lines, 3),
            dtype=torch.float32,
            device=self.device,
        )
        
        for i in range(self.num_vertical_lines):
            for j in range(self.num_horizontal_lines):
                # Calculate angles
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_horizontal_lines - 1))
                
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_vertical_lines - 1))
                
                # Convert to Cartesian coordinates
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 2] = math.sin(elevation_angle)
        
        # Normalize and convert to warp tensor
        normalized_rays = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        self.ray_vectors = wp.from_torch(normalized_rays, dtype=wp.vec3)

    def _initialize_height_scanner_rays(self):
        """Initialize rays for height scanner pattern"""
        
        # Get grid parameters
        size_x, size_y = self.sensor_cfg.height_scanner_size  # e.g., [4.0, 4.0] meters
        resolution = self.sensor_cfg.height_scanner_resolution  # e.g., 0.5 meters
        
        # Calculate number of rays per axis
        self.num_horizontal_lines = int(size_x / resolution) + 1  # +1 to include endpoints
        self.num_vertical_lines = int(size_y / resolution) + 1
        
        print(f"Height scanner setup: {self.num_horizontal_lines}x{self.num_vertical_lines} grid, {resolution}m spacing")
        
        # Create grid coordinates relative to robot center
        x_coords = torch.arange(start=-size_x/2, end=size_x/2+ 1.0e-9, step=resolution, device=self.device)
        y_coords = torch.arange(start=-size_y/2, end=size_y/2+ 1.0e-9, step=resolution, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Ray directions (all pointing downward)
        direction = self.sensor_cfg.height_scanner_direction  # [0, 0, -1]
        base_direction = torch.tensor(direction, device=self.device, dtype=torch.float32)
        
        # Initialize ray origins and directions tensors
        # IMPORTANT: Match tensor dimensions with grid dimensions
        # grid_x, grid_y have shape (num_horizontal_lines, num_vertical_lines)
        ray_origins = torch.zeros((self.num_horizontal_lines, self.num_vertical_lines, 3), 
                                device=self.device, dtype=torch.float32)
        ray_directions = torch.zeros((self.num_horizontal_lines, self.num_vertical_lines, 3), 
                                   device=self.device, dtype=torch.float32)
        
        # For height scanner: rays start above the ground grid positions
        # Define ray origins in ROBOT'S LOCAL COORDINATE FRAME
        # This ensures the grid follows robot orientation
        height_above_ground = self.sensor_cfg.height_scanner_height_above_ground  # Configurable height above ground
        offset_x, offset_y = self.sensor_cfg.height_scanner_offset  # Configurable x, y offset
        
        num_rays = grid_x.numel()
        ray_origins = torch.zeros(num_rays, 3, device=self.device)
        ray_origins[:, 0] = grid_x.flatten()
        ray_origins[:, 1] = grid_y.flatten()
        ray_origins[:, 2] = height_above_ground
        ray_directions = torch.zeros_like(ray_origins)
        ray_directions[..., :] = torch.tensor(list(self.sensor_cfg.height_scanner_direction), device=self.device)

        # Store both origins and directions for height scanner
        self.ray_origins = ray_origins
        self.ray_directions = ray_directions
        
        # Also store as regular ray_vectors for compatibility (use directions)
        self.ray_vectors = wp.from_torch(ray_directions.view(-1,3), dtype=wp.vec3)
        
        print(f"Height scanner rays initialized: {self.num_horizontal_lines}x{self.num_vertical_lines} = {self.num_horizontal_lines * self.num_vertical_lines} rays")
        print(f"Grid spans: X=[{grid_x.min():.2f}, {grid_x.max():.2f}], Y=[{grid_y.min():.2f}, {grid_y.max():.2f}]")
        print(f"Ray origins start at height: {height_above_ground}m above robot base")
        print(f"Ray offset: X={offset_x:.2f}m, Y={offset_y:.2f}m")
        
    def _initialize_pattern_rays(self):
        """Initialize ray vectors for pattern-based lidars (Livox, spinning)"""
        rays_theta, rays_phi = self._generate_ray_angles()
        
        # Set dimensions for compatibility
        self.num_rays = len(rays_phi)
        self.num_vertical_lines = self.num_rays
        self.num_horizontal_lines = 1
        
        # Convert spherical to Cartesian coordinates
        rays_theta_tensor = torch.tensor(rays_theta, dtype=torch.float32, device=self.device)
        rays_phi_tensor = torch.tensor(rays_phi, dtype=torch.float32, device=self.device)
        
        cos_phi = torch.cos(rays_phi_tensor)
        sin_phi = torch.sin(rays_phi_tensor)
        cos_theta = torch.cos(rays_theta_tensor)
        sin_theta = torch.sin(rays_theta_tensor)
        
        # Calculate Cartesian coordinates
        x = cos_phi * cos_theta
        y = cos_phi * sin_theta
        z = sin_phi
        
        # Stack and reshape
        ray_vectors = torch.stack([x, y, z], dim=1)
        ray_vectors = ray_vectors.reshape(self.num_rays, 1, 3)
        
        # Normalize and store
        normalized_rays = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        self.ray_vectors = wp.from_torch(normalized_rays, dtype=wp.vec3)

    def _generate_ray_angles(self):
        """Generate ray angles using the appropriate generator"""
        sensor_type_str = self.sensor_cfg.sensor_type.value
        
        if self.sensor_cfg.is_livox_sensor and isinstance(self.ray_generator, LivoxGenerator):
            return self.ray_generator.sample_ray_angles()
        elif self.sensor_cfg.is_spinning_lidar and isinstance(self.ray_generator, SpinningLidarGenerator):
            if sensor_type_str == "hdl64":
                return self.ray_generator.generate_HDL64()
            elif sensor_type_str == "vlp32":
                return self.ray_generator.generate_VLP32()
            elif sensor_type_str == "os128":
                return self.ray_generator.generate_OS128()
        
        raise ValueError(f"Cannot generate ray angles for sensor type: {sensor_type_str}")

    def update_ray_vectors(self):
        """Update ray vectors for sensors with dynamic patterns"""
        if self.sensor_cfg.is_simple_grid or self.sensor_cfg.is_height_scanner:
            return  # Grid-based sensors and height scanners don't need updates
        
        if self.sensor_cfg.is_livox_sensor and isinstance(self.ray_generator, LivoxGenerator):
            # Update Livox ray pattern
            rays_theta, rays_phi = self.ray_generator.sample_ray_angles()
            self._update_ray_tensor(rays_theta, rays_phi)
        # Spinning lidars could also be updated here if needed

    def _update_ray_tensor(self, rays_theta, rays_phi):
        """Update the ray tensor with new angles"""
        rays_theta_tensor = torch.tensor(rays_theta, dtype=torch.float32, device=self.device)
        rays_phi_tensor = torch.tensor(rays_phi, dtype=torch.float32, device=self.device)
        
        cos_phi = torch.cos(rays_phi_tensor)
        sin_phi = torch.sin(rays_phi_tensor)
        cos_theta = torch.cos(rays_theta_tensor)
        sin_theta = torch.sin(rays_theta_tensor)
        
        x = cos_phi * cos_theta
        y = cos_phi * sin_theta
        z = sin_phi
        
        ray_vectors = torch.stack([x, y, z], dim=1)
        ray_vectors = ray_vectors.reshape(self.num_rays, 1, 3)
        normalized_rays = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        
        # Update by recreating the warp array
        self.ray_vectors = wp.from_torch(normalized_rays, dtype=wp.vec3)

    def create_render_graph_pointcloud(self):
        """Create the warp computation graph for point cloud rendering"""
        # Graph capture only works on CUDA devices
        if 'cuda' not in str(self.device).lower():
            print(f"Note: Graph capture not available on {self.device}, using direct kernel launch")
            self.graph = None  # Will use direct kernel launch
            return
        
        # Temporarily disable CUDA error verification during graph capture
        original_verify_cuda = getattr(wp.config, 'verify_cuda', False)
        if hasattr(wp.config, 'verify_cuda'):
            wp.config.verify_cuda = False
        
        try:
            wp.capture_begin(device=self.device)
            
            if self.sensor_cfg.is_height_scanner:
                # Use height scanner kernel with different ray origins
                wp.launch(
                    kernel=LidarWarpKernels.draw_height_scanner_kernel,
                    dim=(
                        self.num_envs,
                        self.num_sensors,
                        self.num_vertical_lines,
                        self.num_horizontal_lines,
                    ),
                    inputs=[
                        self.mesh_ids,
                        self.lidar_positions,
                        self.lidar_quat_array,
                        self.height_scanner_ray_origins,
                        self.height_scanner_ray_directions,
                        self.far_plane,
                        self.lidar_warp_tensor,
                        self.local_dist,
                        self.pointcloud_in_world_frame,
                    ],
                    device=self.device,
                )
            else:
                # Use regular kernel for other sensor types
                wp.launch(
                    kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,
                    dim=(
                        self.num_envs,
                        self.num_sensors,
                        self.num_vertical_lines,
                        self.num_horizontal_lines,
                    ),
                    inputs=[
                        self.mesh_ids,
                        self.lidar_positions,
                        self.lidar_quat_array,
                        self.ray_vectors,
                        self.far_plane,
                        self.lidar_warp_tensor,
                        self.local_dist,
                        self.pointcloud_in_world_frame,
                    ],
                    device=self.device,
                )
            
            self.graph = wp.capture_end(device=self.device)
            kernel_type = "height_scanner" if self.sensor_cfg.is_height_scanner else "regular"
            print(f"Render graph created for {self.sensor_cfg.sensor_type.value} lidar ({kernel_type} kernel)")
            
        finally:
            # Restore original CUDA verification setting
            if hasattr(wp.config, 'verify_cuda'):
                wp.config.verify_cuda = original_verify_cuda

    def init_tensors(self):
        """Initialize warp tensors for computation"""
        self.lidar_positions = wp.from_torch(
            self.lidar_positions_tensor.view(self.num_envs, 1, 3), dtype=wp.vec3
        )
        self.lidar_quat_array = wp.from_torch(
            self.lidar_quat_tensor.view(self.num_envs, 1, 4), dtype=wp.quat
        )
        
        # Initialize output tensors
        self.lidar_tensor = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.num_vertical_lines,
                self.num_horizontal_lines,
                3,
            ),
            device=self.device,
            requires_grad=False,
        )
        
        self.lidar_dist_tensor = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.num_vertical_lines,
                self.num_horizontal_lines,
            ),
            device=self.device,
            requires_grad=False,
        )
        
        # Convert to warp tensors
        self.local_dist = wp.from_torch(self.lidar_dist_tensor, dtype=wp.float32)
        self.lidar_pixels_tensor = torch.zeros_like(self.lidar_tensor, device=self.device)
        self.lidar_warp_tensor = wp.from_torch(self.lidar_tensor, dtype=wp.vec3)
        
        # For height scanner, initialize additional tensors
        if self.sensor_cfg.is_height_scanner:
            # Expand ray origins and directions to include env and sensor dimensions
            ray_origins_expanded = self.ray_origins.unsqueeze(0).unsqueeze(0).expand(
                self.num_envs, self.num_sensors, -1, -1, -1
            )
            ray_directions_expanded = self.ray_directions.unsqueeze(0).unsqueeze(0).expand(
                self.num_envs, self.num_sensors, -1,-1, -1
            )
            
            self.height_scanner_ray_origins = wp.from_torch(ray_origins_expanded.view( self.num_envs, self.num_sensors, self.num_vertical_lines, self.num_horizontal_lines,-1), dtype=wp.vec3)
            self.height_scanner_ray_directions = wp.from_torch(ray_directions_expanded.view( self.num_envs, self.num_sensors, self.num_vertical_lines, self.num_horizontal_lines,-1), dtype=wp.vec3)

    def capture(self):
        """Capture the render graph if not already created"""
        if self.graph is None:
            self.create_render_graph_pointcloud()
    def tensor_indices_to_slice(idx: torch.Tensor):
        # expects 1-D int tensor
        idx = idx.to(dtype=torch.long)
        if idx.numel() == 0:
            return slice(0, 0)
        sorted_idx, _ = torch.sort(idx)
        first = int(sorted_idx[0].item())
        last = int(sorted_idx[-1].item())
        if last - first + 1 == sorted_idx.numel():
            return slice(first, last + 1)  # stop is exclusive
        return None  # not contiguous
    def reset(self, env_ids=None, value=0.0):
        """Reset selected environments (or all) in lidar_warp_tensor"""
        if env_ids is None:
            self.lidar_warp_tensor.fill_(value)
            return

        # Accept single int
        if isinstance(env_ids, int):
            self.lidar_warp_tensor[env_ids].fill_(value)
            return

        # Accept slice directly
        if isinstance(env_ids, slice):
            self.lidar_warp_tensor[env_ids].fill_(value)
            return

        # Torch tensor or list/tuple of ints
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            sl = self.tensor_indices_to_slice(env_ids)
            if sl is not None:
                self.lidar_warp_tensor[sl].fill_(value)
            else:
                # Fallback: per-index fill
                for i in env_ids.tolist():
                    self.lidar_warp_tensor[int(i)].fill_(value)
            return

        if isinstance(env_ids, (list, tuple)):
            if len(env_ids) == 0:
                return
            t = torch.tensor(env_ids, dtype=torch.long)
            sl = self.tensor_indices_to_slice(t)
            if sl is not None:
                self.lidar_warp_tensor[sl].fill_(value)
            else:
                for i in env_ids:
                    self.lidar_warp_tensor[int(i)].fill_(value)
            return

        raise TypeError(f"Unsupported env_ids type: {type(env_ids)}")
    def update(self):
        """Update sensor and return point cloud data"""
        self.sensor_t += self.env_dt
        # Update ray vectors if needed
        if self.sensor_t > self.update_dt:
            self.update_ray_vectors()
            self.sensor_t = 0.001
        
        # Execute rendering
        if self.graph is not None:
            # Use captured graph (CUDA only)
            wp.capture_launch(self.graph)
        else:
            # Direct kernel launch (for CPU or first-time setup)
            if self.sensor_cfg.is_height_scanner:
                # Use height scanner kernel
                wp.launch(
                    kernel=LidarWarpKernels.draw_height_scanner_kernel,
                    dim=(
                        self.num_envs,
                        self.num_sensors,
                        self.num_vertical_lines,
                        self.num_horizontal_lines,
                    ),
                    inputs=[
                        self.mesh_ids,
                        self.lidar_positions,
                        self.lidar_quat_array,
                        self.height_scanner_ray_origins,
                        self.height_scanner_ray_directions,
                        self.far_plane,
                        self.lidar_warp_tensor,
                        self.local_dist,
                        self.pointcloud_in_world_frame,
                    ],
                    device=self.device,
                )
            else:
                # Use regular kernel for other sensor types
                wp.launch(
                    kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,
                    dim=(
                        self.num_envs,
                        self.num_sensors,
                        self.num_vertical_lines,
                        self.num_horizontal_lines,
                    ),
                    inputs=[
                        self.mesh_ids,
                        self.lidar_positions,
                        self.lidar_quat_array,
                        self.ray_vectors,
                        self.far_plane,
                        self.lidar_warp_tensor,
                        self.local_dist,
                        self.pointcloud_in_world_frame,
                    ],
                    device=self.device,
                )
        
        # Convert results back to torch tensors
        self.lidar_pixels_tensor = wp.to_torch(self.lidar_warp_tensor)
        self.lidar_dist_tensor = wp.to_torch(self.local_dist)
        
        return self.lidar_pixels_tensor, self.lidar_dist_tensor 
