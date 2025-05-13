import torch
import math

# import nvtx
import warp as wp

from .sensor_kernels.lidar_kernels import LidarWarpKernels
from .sensor_kernels_taichi.base_sensor import BaseSensor
from .sensor_config.lidar_sensor_config import LidarConfig
from .sensor_pattern.sensor_lidar.genera_lidar_scan_pattern import \
    LivoxGenerator, \
    generate_HDL64, \
    generate_vlp32, \
    generate_os128
class LidarSensor(BaseSensor):
    def __init__(self, env,env_cfg,sensor_config, num_sensors=1, device='cuda:0'):
        self.env =env
        self.env_cfg =env_cfg
        self.sensor_cfg: LidarConfig = sensor_config


        #self.sensor_cfg.type = "lidar"
        
        self.env_dt = self.sensor_cfg.dt
        self.update_frequency = self.sensor_cfg.update_frequency
        self.update_dt = 1/self.update_frequency
        
        self.sensor_t = 0
        
        
        self.num_sensors = num_sensors
        self.device = device
        self.robot_position = None
        self.robot_orientation = None
        self.robot_linvel = None
        self.robot_angvel = None
        
        self.num_envs = self.env['num_envs']

        self.mesh_ids = self.env['mesh_ids']
        self.num_vertical_lines = self.sensor_cfg.vertical_line_num
        self.num_horizontal_lines = self.sensor_cfg.horizontal_line_num
        self.pointcloud_in_world_frame = self.sensor_cfg.pointcloud_in_world_frame
        
        self.horizontal_fov_min = math.radians(self.sensor_cfg.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.sensor_cfg.horizontal_fov_deg_max)
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2
        if self.horizontal_fov > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2pi")

        self.vertical_fov_min = math.radians(self.sensor_cfg.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.sensor_cfg.vertical_fov_deg_max)
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2
        if self.vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than pi")
        self.far_plane = self.sensor_cfg.max_range
        self.device = device
        
        assert self.env['sensor_pos_tensor'] is not None
        assert self.env['sensor_quat_tensor'] is not None
        self.lidar_positions_tensor = self.env['sensor_pos_tensor']
        self.lidar_quat_tensor = self.env['sensor_quat_tensor']

        self.lidar_positions = None
        self.lidar_quat_array = None
        self.graph = None
        
        self.livox_generator = LivoxGenerator("mid360")
        
        
        
        wp.init()
        self.initialize_ray_vectors()
        self.init_tensors()

    def initialize_ray_vectors(self):
        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        if self.sensor_cfg.sensor_type == "lidar":
            ray_vectors = torch.zeros(
                (self.num_vertical_lines, self.num_horizontal_lines, 3),
                dtype=torch.float32,
                device=self.device,
            )
            
            for i in range(self.num_vertical_lines):
                for j in range(self.num_horizontal_lines):
                    # Rays go from +HFoV/2 to -HFoV/2 and +VFoV/2 to -VFoV/2
                    azimuth_angle = self.horizontal_fov_max - (
                        self.horizontal_fov_max - self.horizontal_fov_min
                    ) * (j / (self.num_horizontal_lines - 1))
                    
                    elevation_angle = self.vertical_fov_max - (
                        self.vertical_fov_max - self.vertical_fov_min
                    ) * (i / (self.num_vertical_lines - 1))
                    
                    ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                    ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                    ray_vectors[i, j, 2] = math.sin(elevation_angle)
            
            # Normalize ray vectors
            ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
            self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)
            #Debug: Print ray vector statistics
            print(f"Simple Lidar:Ray vectors initialized with shape {ray_vectors.shape}")
            print(f"Sample ray vector: {ray_vectors[0, 0]}")
        
        else:
            print(f"Initial Lidar ",self.sensor_cfg.sensor_type)
            rays_theta, rays_phi = self.livox_generator.sample_ray_angles()  #水平，垂直
            assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta must be same"
            
            self.num_vertical_lines = len(rays_phi)
            self.num_horizontal_lines = len(rays_theta)
            # Store dimensions
            self.num_rays = len(rays_phi)
            self.num_vertical_lines = self.num_rays  # For compatibility
            self.num_horizontal_lines = 1  # For compatibility

            rays_theta_tensor = torch.tensor(rays_theta, dtype=torch.float32, device=self.device)
            rays_phi_tensor = torch.tensor(rays_phi, dtype=torch.float32, device=self.device)
            r = 1.0  # Unit vectors
            cos_phi = torch.cos(rays_phi_tensor)
            sin_phi = torch.sin(rays_phi_tensor)
            cos_theta = torch.cos(rays_theta_tensor)
            sin_theta = torch.sin(rays_theta_tensor)
            
            # Calculate x, y, z components
            x = r * cos_phi * cos_theta
            y = r * cos_phi * sin_theta
            z = r * sin_phi
            # Stack into ray vectors tensor
            ray_vectors = torch.stack([x, y, z], dim=1)
            
            # Reshape to match expected format (num_vertical_lines, num_horizontal_lines, 3)
            # This maintains compatibility with the rest of your code
            ray_vectors = ray_vectors.reshape(self.num_rays, 1, 3)
            
            # Normalize ray vectors (already unit vectors, but just to be safe)
            self.ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
        
        
        
    def update_ray_vectors(self):
        if self.sensor_cfg.sensor_type == "lidar":
            return 
        rays_theta, rays_phi = self.livox_generator.sample_ray_angles()  #水平，垂直
        assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta must be same"
        
        # Store dimensions
        self.num_rays = len(rays_phi)
        self.num_vertical_lines = self.num_rays  # For compatibility
        self.num_horizontal_lines = 1  # For compatibility

        rays_theta_tensor = torch.tensor(rays_theta, dtype=torch.float32, device=self.device)
        rays_phi_tensor = torch.tensor(rays_phi, dtype=torch.float32, device=self.device)
        r = 1.0  # Unit vectors
        cos_phi = torch.cos(rays_phi_tensor)
        sin_phi = torch.sin(rays_phi_tensor)
        cos_theta = torch.cos(rays_theta_tensor)
        sin_theta = torch.sin(rays_theta_tensor)
        
        # Calculate x, y, z components
        x = r * cos_phi * cos_theta
        y = r * cos_phi * sin_theta
        z = r * sin_phi
        # Stack into ray vectors tensor
        ray_vectors = torch.stack([x, y, z], dim=1)
        
        # Reshape to match expected format (num_vertical_lines, num_horizontal_lines, 3)
        # This maintains compatibility with the rest of your code
        ray_vectors = ray_vectors.reshape(self.num_rays, 1, 3)
        
        # Normalize ray vectors (already unit vectors, but just to be safe)
        self.ray_vectors[:] = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
    
        #self.ray_vectors[:] = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph_pointcloud(self):
        wp.capture_begin(device=self.device)
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
        print(f"####starting render lidar!")

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # init buffers. None when uninitialized
        if self.sensor_cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.sensor_cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)

        if self.sensor_cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
        else:
            self.segmentation_pixels = segmentation_pixels



    def init_tensors(self):
        #set sensor position and orientation
        
        self.lidar_positions = wp.from_torch(self.lidar_positions_tensor.view(self.num_envs,1,3), dtype=wp.vec3)
        
        #test = wp.to_torch(self.lidar_positions)
        self.lidar_quat_array = wp.from_torch(self.lidar_quat_tensor.view(self.num_envs,1,4), dtype=wp.quat)
        self.lidar_tensor = torch.zeros(
                        (
                            self.num_envs,  #4
                            self.num_sensors, #1
                            self.num_vertical_lines, #128
                            self.num_horizontal_lines, #512
                            3, #3
                        ),
                        device=self.device,
                        requires_grad=False,
                    )
        self.lidar_dist_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.num_sensors, #1
                    self.num_vertical_lines, #128
                    self.num_horizontal_lines, #512
                ),
                device=self.device,
                requires_grad=False,
            )
        self.local_dist = wp.from_torch(self.lidar_dist_tensor, dtype=wp.float32)
        self.lidar_pixels_tensor = torch.zeros_like(self.lidar_tensor,device=self.device)
        self.lidar_warp_tensor = wp.from_torch(self.lidar_tensor, dtype=wp.vec3)
    # @nvtx.annotate()
    def capture(self):
        if self.graph is None:
            self.create_render_graph_pointcloud()


    def update(self):
        self.sensor_t += self.env_dt
        if self.sensor_t > self.update_dt:
            self.update_ray_vectors()
            self.sensor_t=0.001
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.capture()
        self.lidar_pixels_tensor = wp.to_torch(self.lidar_warp_tensor)
        self.lidar_dist_tensor = wp.to_torch(self.local_dist)
        return self.lidar_pixels_tensor,self.lidar_dist_tensor 
