import numpy as np


import numpy as np
import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi,gymtorch
from isaacgym.torch_utils import *
from math import sqrt
import torch
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType
import random
import time
import trimesh
import warp as wp
import threading
from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain
from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg
from LidarSensor import SENSOR_ROOT_DIR,RESOURCES_DIR
import os

def save_video(frame_stack, key, format=None, fps=20, **imageio_kwargs):
    """
    Let's do the compression here. Video frames are first written to a temporary file
    and the file containing the compressed data is sent over as a file buffer.

    Save a stack of images to

    :param frame_stack: the stack of video frames
    :param key: the file key to which the video is logged.
    :param format: Supports 'mp4', 'gif', 'apng' etc.
    :param imageio_kwargs: (map) optional keyword arguments for `imageio.mimsave`.
    :return:
    """
    if format:
        key += "." + format
    else:
        # noinspection PyShadowingBuiltins
        _, format = os.path.splitext(key)
        if format:
            # noinspection PyShadowingBuiltins
            format = format[1:]  # to remove the dot
        else:
            # noinspection PyShadowingBuiltins
            format = "mp4"
            key += "." + format

    filename ="lidar_demo"
    import tempfile, imageio  # , logging as py_logging
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        from skimage import img_as_ubyte
        try:
            imageio.mimsave(key, img_as_ubyte(frame_stack), format=format, fps=fps, macro_block_size=1,**imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(key, img_as_ubyte(frame_stack), format=format, fps=fps, macro_block_size=1,**imageio_kwargs)
        ntp.seek(0)



KEY_W = gymapi.KEY_W
KEY_A = gymapi.KEY_A
KEY_S = gymapi.KEY_S
KEY_D = gymapi.KEY_D
KEY_Q = gymapi.KEY_Q
KEY_E = gymapi.KEY_E
KEY_UP = gymapi.KEY_UP
KEY_DOWN = gymapi.KEY_DOWN
KEY_LEFT = gymapi.KEY_LEFT
KEY_RIGHT = gymapi.KEY_RIGHT
KEY_ESCAPE = gymapi.KEY_ESCAPE

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

@torch.jit.script
def cart2sphere(cart):
    epsilon = 1e-9
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    r = torch.norm(cart, dim=1)
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)


class sim:
    dt =  0.005
    substeps = 1
    gravity = [0., 0. ,-9.81]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    class physx:
        num_threads = 10
        solver_type = 1  # 0: pgs, 1: tgs
        num_position_iterations = 4
        num_velocity_iterations = 0
        contact_offset = 0.01  # [m]
        rest_offset = 0.0   # [m]
        bounce_threshold_velocity = 0.5 #0.5 [m/s]
        max_depenetration_velocity = 1.0
        max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
        default_buffer_size_multiplier = 5
        contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"},
        {"name": "--headless", "type": bool, "default": False, "help": "Run in headless mode"},])

headless = args.headless

def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)



def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
def farthest_point_sampling(point_cloud, sample_size):
    """
    Sample points using the farthest point sampling algorithm
    Args:
        point_cloud: Tensor of shape (num_envs, 1, num_points,1, 3)
        sample_size: Number of points to sample
    Returns:
        Downsampled point cloud of shape (num_envs, 1, sample_size, 3)
    """
    num_envs, _, num_points, _ = point_cloud.shape
    device = point_cloud.device
    result = []
    
    for env_idx in range(num_envs):
        points = point_cloud[env_idx, 0]  # (num_points, 3)
        
        # Initialize with a random point
        sampled_indices = torch.zeros(sample_size, dtype=torch.long, device=device)
        sampled_indices[0] = torch.randint(0, num_points, (1,), device=device)
        
        # Calculate distances
        distances = torch.norm(points - points[sampled_indices[0]], dim=1)
        
        # Iteratively select farthest points
        for i in range(1, sample_size):
            # Select the farthest point
            sampled_indices[i] = torch.argmax(distances)
            
            # Update distances
            if i < sample_size - 1:
                new_distances = torch.norm(points - points[sampled_indices[i]], dim=1)
                distances = torch.min(distances, new_distances)
        
        # Get the sampled points
        sampled_points = points[sampled_indices]
        result.append(sampled_points.unsqueeze(0))  # Add sensor dimension back
    
    return torch.stack(result)

def downsample_spherical_points_vectorized(sphere_points, num_theta_bins=10, num_phi_bins=10):
    """
    Downsample points in spherical coordinates by binning theta and phi values.
    
    Args:
        sphere_points: Tensor of shape (num_envs, num_points, 3) where dim 2 is (r, theta, phi)
        num_theta_bins: Number of bins for theta range (-3.14, 3.14)
        num_phi_bins: Number of bins for phi range (-0.12, 0.907)
        
    Returns:
        Downsampled points tensor of shape (num_envs, num_theta_bins*num_phi_bins, 3)
    """
    num_envs = sphere_points.shape[0]
    num_points = sphere_points.shape[1]
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins
    
    # Define bin ranges
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.12, 0.907
    
    # Extract r, theta, phi for all environments
    r = sphere_points[:, :, 0]       # [num_envs, num_points]
    theta = sphere_points[:, :, 1]   # [num_envs, num_points]
    phi = sphere_points[:, :, 2]     # [num_envs, num_points]
    
    # Compute bin indices for theta and phi
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
    
    # Clamp to valid bin indices
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
    
    # Compute linear bin index (flatten 2D bin indices to 1D)
    bin_indices = theta_bin * num_phi_bins + phi_bin  # [num_envs, num_points]
    
    # Create an environment index tensor to handle multiple environments
    env_indices = torch.arange(num_envs, device=device).view(-1, 1).expand(-1, num_points)
    
    # Flatten tensors for scatter operation
    flat_bin_indices = bin_indices.reshape(-1)            # [num_envs * num_points]
    flat_env_indices = env_indices.reshape(-1)            # [num_envs * num_points]
    flat_r = r.view(-1)                               # [num_envs * num_points]
    
    # Create 2D indices for scatter operation (env_idx, bin_idx)
    scatter_indices = torch.stack([flat_env_indices, flat_bin_indices], dim=1)  # [num_envs * num_points, 2]
    
    # Prepare tensors for scatter operations
    r_sum = torch.zeros(num_envs, num_bins, device=device)
    bin_count = torch.zeros(num_envs, num_bins, device=device)
    
    # Use scatter_add_ to compute sum and count for each bin
    r_sum.scatter_add_(1, bin_indices, r)
    ones = torch.ones_like(r)
    bin_count.scatter_add_(1, bin_indices, ones)
    
    # Avoid division by zero for empty bins
    bin_count = torch.clamp(bin_count, min=1.0)
    
    # Compute average r per bin
    avg_r = r_sum / bin_count  # [num_envs, num_bins]
    
    # Create bin centers for theta and phi
    theta_centers = torch.linspace(
        theta_min + (theta_max - theta_min) / (2 * num_theta_bins),
        theta_max - (theta_max - theta_min) / (2 * num_theta_bins),
        num_theta_bins, device=device
    )
    
    phi_centers = torch.linspace(
        phi_min + (phi_max - phi_min) / (2 * num_phi_bins),
        phi_max - (phi_max - phi_min) / (2 * num_phi_bins),
        num_phi_bins, device=device
    )
    
    # Create meshgrid of bin centers
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)  # [num_bins]
    phi_centers_flat = phi_grid.reshape(-1)      # [num_bins]
    
    # Create final output tensor
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = avg_r                              # r values
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)    # theta values
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)      # phi values
    
    return downsampled
class G1Env:
    def __init__(self, 
                 num_envs=1, 
                 num_obstacles=5,
                 publish_ros=True,
                 save_data=False,
                 save_interval=0.1  # 每1秒保存一次数据
                ):
        """Initialize a minimal lidar sensor environment."""
        self.gym = gymapi.acquire_gym()
        self.num_envs = num_envs
        self.num_obstacles = num_obstacles
        self.headless = args.headless
        self.show_viewer = not args.headless
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sensor_cfg = LidarConfig()
        self.sensor_cfg.sensor_type ="avia" # mid360,horizon,HAP,mid70,mid40,tele,avia
        self.sim_time = 0
        self.sensor_update_time = 0
        self.state_update_time = 0
        self.sensor_cfg.update_frequency
        
        self.save_data = save_data
        self.save_interval = save_interval
        self.save_time = 0
        self.last_save_time = 0
        self.sequence_number=0
        wp.init()
        if self.save_data:
            # 创建保存数据的目录
            self.data_dir = f"./sensor_data_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.data_dir, exist_ok=True)
            
            # 初始化数据存储列表
            self.stored_local_pixels = []  # 存储局部点云数据
            self.stored_robot_positions = []  # 存储机器人位置
            self.stored_robot_orientations = []  # 存储机器人方向
            self.stored_terrain_heights = []  # 存储地形高度测量值
            self.stored_timestamps = []  # 存储时间戳
            
            print(f"######Data will be saved to: {self.data_dir}")
    
        # Create sim
        self.create_sim()
        
        # Create ground plane
        self.create_ground()
        self.create_viewer()
        # Create obstacles
        self.create_env()
        
        self.gym.prepare_sim(self.sim)
        
        self._init_buffer()
        self.create_warp_env()

        self.crete_warp_tensor()
        
        self.sensor = LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()
        self.record_video = True
        # if recording video, set up camera
        if self.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[0],
                                                                        self.camera_props)
            self.gym.set_camera_location(self.rendering_camera_eval, self.envs[0],
                                            gymapi.Vec3(1.5, 1, 3.0),
                                            gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []
        # Initialize keyboard state dictionary
        self.key_pressed = {}
        
        # Movement and rotation speeds
        self.linear_speed = 0.0  # m/s
        self.angular_speed = 0.0  # rad/s
        self.selected_env_idx = 0  # Environment to control (default to env 0 as in your draw code)
        
        # Initialize random movement parameters
        self.enable_random_movement = False
        self.movement_update_interval = 0.2  # Generate new target every 3 seconds
        self.movement_speed = 2.0  # Movement speed scalar
        self.rotation_speed = 1.0  # Rotation speed scalar
        
        # Initialize target position and orientation for all environments
        self.target_pos = None
        self.target_quat = None
        self.target_timer = None
        
        # Generate initial random targets for all environments
        if self.enable_random_movement:
            
            #self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            terrain_length = self.terrain_cfg.terrain_length
            terrain_width = self.terrain_cfg.terrain_width
            terrain_x_limit = terrain_length * self.terrain_cfg.num_rows
            terrain_y_limit = terrain_width * self.terrain_cfg.num_cols
            
            # Generate random x, y within terrain bounds for all environments at once
            random_x = torch.rand(self.num_envs, device=self.device) * terrain_x_limit
            random_y = torch.rand(self.num_envs, device=self.device) * terrain_y_limit
            self.measured_heights=self._get_heights()
            height_at_target = torch.mean(self.measured_heights, dim=1) 
            height_offset = torch.rand(self.num_envs, device=self.device) * 0.2-0.1
            random_z = height_at_target + height_offset+0.34
            self.env_origins[:] = torch.stack((random_x, random_y, random_z), dim=1)
            self.generate_random_targets()
            
            
        if self.viewer:
            # 订阅所有需要的键
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_W,"move_forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_A,"move_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_S,"move_backward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_D,"move_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_Q,"move_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_E,"move_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_UP,"rotate_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_DOWN,"rotate_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_LEFT,"rotate_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_RIGHT,"rotate_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_ESCAPE,"exit")
        
        

            
        # Print control instructions
        print("Keyboard controls:")
        print("  WASD: Move robot horizontally")
        print("  Q/E: Move robot up/down")
        print("  Arrow keys: Rotate robot")
        print("  ESC: Exit simulation")
        
    def generate_random_targets(self):
        """Generate random velocities for all environments and limit positions within bounds"""
        # Get terrain dimensions for position limits
        terrain_length = self.terrain_cfg.horizontal_scale
        terrain_width = self.terrain_cfg.horizontal_scale
        terrain_x_limit = terrain_length * self.terrain_cfg.num_rows
        terrain_y_limit = terrain_width * self.terrain_cfg.num_cols
        
        # Generate random velocities for all environments at once
        # Using smaller values for better control
        lin_vel_scale = 4.0
        ang_vel_scale = 3
        
        # Random linear velocities in x, y, z directions
        random_lin_vel = (torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0) * lin_vel_scale
        
        # Random angular velocities around x, y, z axes
        random_ang_vel = (torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0) * ang_vel_scale
        
        # Store velocities for updating positions later
        self.target_lin_vel = random_lin_vel
        self.target_ang_vel = random_ang_vel
        
        # Reset the timer with variation to prevent all robots from changing velocities at the same time
        variation = torch.rand(self.num_envs, device=self.device) * 0.5
        self.target_timer = self.movement_update_interval + variation
        
        # Print first few velocities for debugging
        num_to_print = min(2, self.num_envs)
        for i in range(num_to_print):
            print(f"Env {i} - New velocities: lin_vel={self.target_lin_vel[i]}, ang_vel={self.target_ang_vel[i]}")
        
    def move_to_targets(self):
        """Move robots with random velocities and enforce position/orientation constraints"""
        if not hasattr(self, 'target_lin_vel') or not hasattr(self, 'target_ang_vel'):
            return
        
        # Get current positions and orientations
        current_pos = self.root_states[:, 0:3].clone()
        current_quat = self.root_states[:, 3:7].clone()
        
        # Convert current quaternions to euler angles
        current_roll, current_pitch, current_yaw = euler_from_quaternion(current_quat)
        
        # Apply linear velocity (in global frame to avoid quaternion application per environment)
        dt = self.dt
        new_pos = current_pos + self.target_lin_vel * dt
        
        # Update orientation using angular velocities
        new_roll = current_roll + self.target_ang_vel[:, 0] * dt
        new_pitch = current_pitch + self.target_ang_vel[:, 1] * dt
        new_yaw = current_yaw + self.target_ang_vel[:, 2] * dt
        
        # Get terrain dimensions for position limits
        terrain_length = self.terrain_cfg.horizontal_scale
        terrain_width = self.terrain_cfg.horizontal_scale
        terrain_x_limit = terrain_length * self.terrain_cfg.num_rows
        terrain_y_limit = terrain_width * self.terrain_cfg.num_cols
        
        # Get height limits
        if hasattr(self, 'measured_heights') and self.measured_heights is not None:
            height_at_pos = torch.mean(self.measured_heights, dim=1)
            min_height = height_at_pos + 0.2
            max_height = height_at_pos + 0.5
        else:
            min_height = torch.zeros(self.num_envs, device=self.device) + 0.2
            max_height = torch.zeros(self.num_envs, device=self.device) + 0.5
        
        # Check position constraints and reverse velocities if needed
        # X constraints [0, terrain_x_limit]
        x_too_low = new_pos[:, 0] < 0
        x_too_high = new_pos[:, 0] > terrain_x_limit
        if torch.any(x_too_low) or torch.any(x_too_high):
            # Reverse x velocity for robots that would go out of bounds
            self.target_lin_vel[:, 0] = torch.where(
                (x_too_low | x_too_high), 
                -self.target_lin_vel[:, 0], 
                self.target_lin_vel[:, 0]
            )
            # Correct position to stay within bounds
            new_pos[:, 0] = torch.clamp(new_pos[:, 0], 0, terrain_x_limit)
        
        # Y constraints [0, terrain_y_limit]
        y_too_low = new_pos[:, 1] < 0
        y_too_high = new_pos[:, 1] > terrain_y_limit
        if torch.any(y_too_low) or torch.any(y_too_high):
            # Reverse y velocity
            self.target_lin_vel[:, 1] = torch.where(
                (y_too_low | y_too_high), 
                -self.target_lin_vel[:, 1], 
                self.target_lin_vel[:, 1]
            )
            # Correct position
            new_pos[:, 1] = torch.clamp(new_pos[:, 1], 0, terrain_y_limit)
        
        # Z constraints [min_height, max_height]
        z_too_low = new_pos[:, 2] < min_height
        z_too_high = new_pos[:, 2] > max_height
        if torch.any(z_too_low) or torch.any(z_too_high):
            # Reverse z velocity
            self.target_lin_vel[:, 2] = torch.where(
                (z_too_low | z_too_high), 
                -self.target_lin_vel[:, 2], 
                self.target_lin_vel[:, 2]
            )
            # Correct height
            new_pos[:, 2] = torch.where(z_too_low, min_height, new_pos[:, 2])
            new_pos[:, 2] = torch.where(z_too_high, max_height, new_pos[:, 2])
        
        # Check orientation constraints and reverse angular velocities if needed
        # Roll constraints [-0.78, 0.78]
        roll_too_low = new_roll < -0.78
        roll_too_high = new_roll > 0.78
        if torch.any(roll_too_low) or torch.any(roll_too_high):
            # Reverse roll velocity
            self.target_ang_vel[:, 0] = torch.where(
                (roll_too_low | roll_too_high), 
                -self.target_ang_vel[:, 0], 
                self.target_ang_vel[:, 0]
            )
            # Correct roll
            new_roll = torch.clamp(new_roll, -0.78, 0.78)
        
        # Pitch constraints [-1.04, 0.523]
        pitch_too_low = new_pitch < -1.04
        pitch_too_high = new_pitch > 0.523
        if torch.any(pitch_too_low) or torch.any(pitch_too_high):
            # Reverse pitch velocity
            self.target_ang_vel[:, 1] = torch.where(
                (pitch_too_low | pitch_too_high), 
                -self.target_ang_vel[:, 1], 
                self.target_ang_vel[:, 1]
            )
            # Correct pitch
            new_pitch = torch.clamp(new_pitch, -1.04, 0.523)
        
        # Yaw constraints [-3.14, 3.14]
        # For yaw, we need to handle the wrap-around at +/- π
        new_yaw = torch.remainder(new_yaw + torch.pi, 2 * torch.pi) - torch.pi
        
        # Convert updated euler angles to quaternions
        new_quat = torch.zeros_like(current_quat)
        for i in range(self.num_envs):
            new_quat[i] = quat_from_euler_xyz(new_roll[i], new_pitch[i], new_yaw[i]).squeeze()
        
        # Update robot states with new positions and orientations
        self.root_states[:, 0:3] = new_pos+self.env_origins
        self.root_states[:, 3:7] = new_quat
        
        # Apply changes to simulation for all environments at once
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states)
        )


    def _init_buffer(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)

            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)



        self.base_quat = self.root_states[:, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.up_axis_idx=2
        self.gravity_vec = to_torch([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_pose = self.root_states[:, 0:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        
        
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.last_projected_gravity = self.projected_gravity.clone()
        
        
        self.height_points = self._init_height_points()
        self.measured_heights=self._get_heights()
        
        
        
        self.sensor_translation = torch.tensor([0.0002835, 0.00003, 0.41818], device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor([3.14, 0., 0], device=self.device)
        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.terrain_cfg.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.terrain_cfg.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.terrain_cfg.measure_horizontal_noise, self.terrain_cfg.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.terrain_cfg.measure_horizontal_noise, self.terrain_cfg.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def create_warp_env(self):
        
        
        terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        #save terrain mesh
        transform = np.zeros((3,))
        transform[0] = -self.terrain_cfg.border_size 
        transform[1] = -self.terrain_cfg.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        
        two_levels_up = os.path.dirname(os.path.dirname(current_script_dir))
        
        
        obstacle_mesh_path = os.path.join(two_levels_up, "resources", "robots","g1_29", "robot_combined.stl")
        
        obstacle_mesh = trimesh.load(obstacle_mesh_path)


            #obstacle_mesh = trimesh.load(self.terrain_cfg.obstacle_config.obstacle_root_path+"/human/meshes/Male.OBJ")
        transaltion = np.zeros((3,))
        transaltion[0]=self.root_states[0,0]
        transaltion[1]=self.root_states[0,1]
        transaltion[2]=self.root_states[0,2]
        # quat = self.root_states[0,3:7].numpy()
        # rotation = trimesh.transformations.quaternion_matrix(quat)
        translation = trimesh.transformations.translation_matrix(transaltion)
            
        obstacle_mesh.apply_transform(translation)

        combine_mesh = trimesh.util.concatenate([terrain_mesh, obstacle_mesh])
        #save combined mesh
        #combine_mesh.export("robot_terrain_combined.stl")
        vertices = combine_mesh.vertices
        triangles = combine_mesh.faces
        vertex_tensor = torch.tensor( 
                vertices,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )
        
        #if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        self.mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)
        
    def create_sim(self):
        """Create a Genesis simulation."""
        # configure sim
        sim_params = gymapi.SimParams()
        
        
        dt =  0.02
        self.dt = dt
        sim_params.dt = dt
        if args.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.shape_collision_margin = 0.25
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 10
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 1
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = args.num_threads
            sim_params.physx.use_gpu = args.use_gpu
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        sim_params.use_gpu_pipeline = True    
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        
    def create_ground(self):
        """Create a ground plane."""
        self.terrain_cfg = Terrain_cfg()
        self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        self._create_trimesh()
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_cfg.static_friction
        tm_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        tm_params.restitution = self.terrain_cfg.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def create_viewer(self):
        # create viewer
        if args.headless:
            self.viewer = None
            print("Running in headless mode")
        else:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.terrain_cfg.max_init_terrain_level
            if not self.terrain_cfg.curriculum: max_init_level = self.terrain_cfg.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.terrain_cfg.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.terrain_cfg.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
        

    


    def create_env(self):
        """Create random obstacles in the environment using URDF files."""
        self.obstacles = []
        

        sensor_asset_root = f"{RESOURCES_DIR}"
        sensor_asset_file = "robots/g1_29/g1_29dof.urdf"
        asset_options = gymapi.AssetOptions()
        
        
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        sensor_asset = self.gym.load_asset(self.sim, sensor_asset_root, sensor_asset_file,asset_options)

        
        # Create obstacles
        self.object_handles = []
        self.warp_meshes_trasnformation =[]
        self.warp_meshes_list = []
    
        
        self.warp_mesh_per_env = []
        self.warp_mesh_id_list = []
        
        
        #self.num_envs = args.num_envs
        self._get_env_origins()
        num_per_row = int(sqrt(self.num_envs))
        env_spacing = 0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.envs=[]
        self.sensor_handles = []
        
        for i in range(self.num_envs):
            
            pos = self.env_origins[i].clone()
            
            env = self.gym.create_env(self.sim,env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # create sensor pose

            start_pose = gymapi.Transform()
            start_pose.r  = gymapi.Quat(0, 0, 0, 1)
            pos[2] = 0.85
            start_pose.p = gymapi.Vec3(*(pos))
            sensor_hanle = self.gym.create_actor(env, sensor_asset, start_pose, "sensor",i, 1, 1)
            self.sensor_handles.append(sensor_hanle)
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.sensor_handles[0])
        print("dof_names",dof_names)

                          
    def create_obstacles(self):
        """Create random obstacles in the environment using URDF files."""
        self.obstacles = []
        
        # Define obstacle paths
        obstacle_path_mesh = f"./resources/envs/plane/pillar_08x08x3.stl"
        obstacle_path_asset = f"./resources/envs/plane/pillar.urdf"
        
        # Define ranges for obstacle placement
        x_range = (-5.0, 5.0)
        y_range = (-5.0, 5.0)
        z_height = 0.0
        
        # Create obstacles
        self.object_handles = []
        self.warp_meshes_trasnformation =[]
        self.warp_meshes_list = []
    
        
        self.warp_mesh_per_env = []
        self.warp_mesh_id_list = []
        
        per_env_obstacle_transformations = []
        per_env_obstacle_meshes =[]  
        
        for _obj in range(self.num_obstacles):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            pos = np.array([x, y, z_height])
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Create obstacle using URDF
            obstacle_handle = self.scene.add_entity(
                gs.morphs.URDF(
                    file=obstacle_path_asset,
                    fixed=True,
                    pos=pos,
                    quat=quat,
                ),
            )

            self.object_handles.append(obstacle_handle)  

            self.warp_meshes_list.append(obstacle_path_mesh)
            self.warp_meshes_trasnformation.append(pos) 
            
        self.create_obstacles_warp_mesh(self.warp_meshes_list,self.warp_meshes_trasnformation)
        
        
    def create_obstacles_warp_mesh(self,obstacle_meshes,obstacle_transformations):
        # triangles = self.terrain.triangles
        # vertices = self.terrain.vertices
        
        
        self.warp_mesh_per_env =[]
        self.warp_mesh_id_list = []
        self.global_tensor_dict = {}
        self.obstacle_mesh_per_env =[]
        self.obstacle_vertex_indices = [] 
        self.obstacle_indices_per_vertex_list = []
        num_obstacles = self.num_obstacles

        self.single_num_vertices_list = []
        self.all_obstacles_points = []
        for i in range(num_obstacles):
            mesh_path = obstacle_meshes[i]
            obstacle_mesh = trimesh.load(mesh_path)
            #obstacle_mesh = trimesh.load(self.terrain_cfg.obstacle_config.obstacle_root_path+"/human/meshes/Male.OBJ")
            translation = trimesh.transformations.translation_matrix(obstacle_transformations[i])
            
            obstacle_mesh.apply_transform(translation)
            self.obstacle_mesh_per_env.append(obstacle_mesh)
            obstacle_points = obstacle_mesh.vertices
            self.all_obstacles_points.append(obstacle_points)
            # Record start and end indices of vertices for this obstacle
            
            # 计算障碍物的全局索引
            self.single_num_vertices_list.append(len(obstacle_mesh.vertices))

        terrain_mesh = trimesh.load(f"./resources/envs/plane/plane100.obj")
        # terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        transform = np.zeros((3,))
        transform[0] = -25
        transform[1] = -25 # TODO
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)
        
        
        combined_mesh = trimesh.util.concatenate(self.obstacle_mesh_per_env+[terrain_mesh])
        #combined_mesh = terrain_mesh

        vertices = combined_mesh.vertices
        triangles = combined_mesh.faces

        vertex_tensor = torch.tensor(  # 288,3
                vertices,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )           
        vertex_vec3_array = wp.from_torch(
            vertex_tensor,
            dtype=wp.vec3,
        )
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_mesh =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        
        old_points = wp.to_torch(self.wp_mesh.points)
        self.init_points = old_points.clone()


        self.warp_mesh_per_env.append(self.wp_mesh)
        self.warp_mesh_id_list.append(self.wp_mesh.id)


        
        # wrap to one tensor
        #self.obstacle_indices_per_vertex = torch.tensor(self.obstacle_indices_per_vertex_list, device=self.device)
        self.single_num_vertices = torch.tensor(self.single_num_vertices_list, device=self.device)
        self.all_obstacle_num_vertices = torch.sum(self.single_num_vertices)
        
        
        
    def crete_warp_tensor(self):
        self.warp_tensor_dict={}
        self.lidar_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                    3, #3
                ),
                device=self.device,
                requires_grad=False,
            )        
        self.sensor_dist_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                ),
                device=self.device,
                requires_grad=False,
            ) 
        #self.mesh_ids = self.mesh_ids_array = wp.array(self.warp_mesh_id_list, dtype=wp.uint64)
        self.sensor_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.sensor_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])
        
        
        self.sensor_translation = torch.tensor([0., 0.0, 0.436], device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor([3.14, 0, 0], device=self.device)

        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))
        # self.sensor_pos_tensor = self.root_states[:, 0:3]
        # self.sensor_quat_tensor = self.root_states[:, 3:7]
        
        self.warp_tensor_dict["sensor_dist_tensor"] = self.sensor_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.sensor_cfg.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids
        
        
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.terrain_cfg.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.terrain_cfg.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")


        
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)            

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale      
        
        
         
    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
 
    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        #self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.selected_env_idx
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    def keyboard_input(self):
        """Process keyboard input to move the robot"""
        # 在没有查看器的情况下直接返回，因为无法获取键盘输入
        if not self.viewer:
            return True
        
        # 处理自上次调用以来的所有事件
        for evt in self.gym.query_viewer_action_events(self.viewer):
            print(f"Key event: action={evt.action}, value={evt.value}")  # 调试信息
            
            # 处理按键事件 - 当值大于0时表示按下，等于0时表示释放
            if evt.action == "move_forward":
                self.key_pressed[KEY_W] = evt.value > 0
            elif evt.action == "move_backward":
                self.key_pressed[KEY_S] = evt.value > 0
            elif evt.action == "move_left":
                self.key_pressed[KEY_A] = evt.value > 0
            elif evt.action == "move_right":
                self.key_pressed[KEY_D] = evt.value > 0
            elif evt.action == "move_up":
                self.key_pressed[KEY_Q] = evt.value > 0
            elif evt.action == "move_down":
                self.key_pressed[KEY_E] = evt.value > 0
            elif evt.action == "rotate_up":
                self.key_pressed[KEY_UP] = evt.value > 0
            elif evt.action == "rotate_down":
                self.key_pressed[KEY_DOWN] = evt.value > 0
            elif evt.action == "rotate_left":
                self.key_pressed[KEY_LEFT] = evt.value > 0
            elif evt.action == "rotate_right":
                self.key_pressed[KEY_RIGHT] = evt.value > 0
            elif evt.action == "exit" and evt.value > 0:
                print("Exiting simulation")
                return False
        
        # 固定时间步长（与模拟一致）
        dt = 0.005
        
        # 获取选定机器人的当前状态
        env_idx = self.selected_env_idx
        current_pos = self.root_states[env_idx, 0:3].clone()
        current_quat = self.root_states[env_idx, 3:7].clone()
        
        # 设置速度 (每次调用时设置固定速度，而不是累加)
        self.linear_speed = 3.0  # 1 m/s
        self.angular_speed = 3.0  # 1 rad/s
        
        # 初始化速度向量 - 始终从零开始以响应当前按键状态
        linear_vel = torch.zeros(3, device=self.device)
        euler_rates = torch.zeros(3, device=self.device)
        
        # 处理按键状态 - 设置当前速度
        # 前后移动
        if self.key_pressed.get(KEY_W, False):
            linear_vel[0] = self.linear_speed
        if self.key_pressed.get(KEY_S, False):
            linear_vel[0] = -self.linear_speed
            
        # 左右移动
        if self.key_pressed.get(KEY_A, False):
            linear_vel[1] = -self.linear_speed
        if self.key_pressed.get(KEY_D, False):
            linear_vel[1] = self.linear_speed
            
        # 上下移动
        if self.key_pressed.get(KEY_Q, False):
            linear_vel[2] = self.linear_speed
        if self.key_pressed.get(KEY_E, False):
            linear_vel[2] = -self.linear_speed
            
        # 旋转控制（偏航）
        if self.key_pressed.get(KEY_LEFT, False):
            euler_rates[2] = self.angular_speed
        if self.key_pressed.get(KEY_RIGHT, False):
            euler_rates[2] = -self.angular_speed
            
        # 旋转控制（俯仰）
        if self.key_pressed.get(KEY_UP, False):
            euler_rates[1] = self.angular_speed
        if self.key_pressed.get(KEY_DOWN, False):
            euler_rates[1] = -self.angular_speed
        
        # 将局部线性速度转换为全局速度
        global_vel = quat_apply(current_quat, linear_vel)
        
        # 应用移动 - 根据当前速度和时间步长计算位移
        new_pos = current_pos + global_vel * dt
        
        # 应用旋转（将欧拉角速率转换为四元数变化）
        roll, pitch, yaw = euler_from_quaternion(current_quat.unsqueeze(0))
        
        # 更新欧拉角 - 根据当前角速度和时间步长计算角度变化
        roll = roll + euler_rates[0] * dt
        pitch = pitch + euler_rates[1] * dt
        yaw = yaw + euler_rates[2] * dt
        
        # 转换回四元数
        new_quat = quat_from_euler_xyz(roll, pitch, yaw)
        
        # 更新机器人状态
        self.root_states[env_idx, 0:3] = new_pos
        self.root_states[env_idx, 3:7] = new_quat
        
        # 应用更改到模拟
        env_ids_int32 = torch.tensor([env_idx], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        
        return True  # 继续模拟

    def _render_headless(self):
        self.gym.render_all_camera_sensors(self.sim)
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                        gymapi.Vec3(bx, by, bz))
        #camera_hanle=self.gym.get_viewer_camera_handle(self.viewer)
        self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                        gymapi.IMAGE_COLOR)
        self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
        self.video_frames.append(self.video_frame)
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(bx-3, by- 2.5, bz + 3.5), gymapi.Vec3(bx, by, bz))
        if len(self.video_frames)>250:
            save_video(self.video_frames,f"videos/Lidar_demo.mp4",fps=50)
            print("save video!!")
            self.video_frames=[]
        self.sequence_number = self.sequence_number + 1
        rgb_image_filename = "images/rgb_image_%03d.png" % (self.sequence_number)

        self.gym.write_viewer_image_to_file(self.viewer,rgb_image_filename)

    def step(self):
        """Step the simulation forward."""
        # Process keyboard input first
        if not self.enable_random_movement:
            # Use original keyboard control
            continue_sim = self.keyboard_input()
            if not continue_sim:
                return False
        else:
            # For random movement, check if any environment needs a new target
            if self.target_timer is not None:
                # Find environments that need new targets
                needs_new_target = self.state_update_time > self.target_timer
                if torch.any(needs_new_target):
                    # Generate new targets for all environments at once if any need updating
                    # This is simpler than generating targets for only specific environments
                    self.generate_random_targets()
                    self.state_update_time=0
                
                # Move all environments toward their targets
                self.move_to_targets()
        
        # Rest of the step function remains the same
        self.sim_time += self.dt
        self.sensor_update_time += self.dt
        self.state_update_time += self.dt
        self.save_time += self.dt
        # Update sensor data based on new positions
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        self.base_pose[:] = self.root_states[:, :7]  
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_base_lin_vel) / 0.005
        self.base_ang_vel = (self.root_states[:, 10:13] - self.last_base_ang_vel) / 0.005

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.measured_heights = self._get_heights()
        
        # Update sensor position/orientation
        # sensor_translation = torch.tensor([0.28945, 0.0, -0.046825], device=self.device)
        # rpy_offset = torch.tensor([0.0, -2.8782, 3.14], device=self.device)
        # sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2])
        
        sensor_quat = quat_mul(self.root_states[:, 3:7], self.sensor_offset_quat)
        sensor_pos = self.root_states[:, 0:3] + quat_apply(self.root_states[:, 3:7], self.sensor_translation)
        self.sensor_pos_tensor[:,:] = sensor_pos[:,:]
        self.sensor_quat_tensor[:,:] = sensor_quat[:,:]
        

        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()
        
        #points dowm sample
        #self.downsampled_cloud = self.lidar_tensor.view(self.num_envs,1,self.lidar_tensor.shape[2],3) #
        
        self.downsampled_cloud = farthest_point_sampling(self.lidar_tensor.view(self.num_envs,1,self.lidar_tensor.shape[2],3), sample_size=5000)
        self.sphere_points = cart2sphere(self.lidar_tensor.view(-1,3)).view(self.num_envs,-1,3)
        self.downsampled_sphere_points = downsample_spherical_points_vectorized(self.sphere_points, 10, 10)
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        if self.save_data and (self.save_time) >= self.save_interval:
            self.collect_and_save_data()
            self.save_time = 0
        # Update visualization
        
        self._render_headless()


        if self.sensor_update_time > 1/self.sensor_cfg.update_frequency:
            self.gym.clear_lines(self.viewer)
            self._draw_lidar_vis()
            #self._draw_sphere_vis() wait fix
            #self._draw_height_samples()
            self.sensor_update_time=0
            
            # if self.publish_ros:
            #     self.publish_point_cloud()
            #     # 处理ROS消息
            #     rclpy.spin_once(self.ros_node, timeout_sec=0)
        
        if self.show_viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

        # Synchronize with rendering rate
        self.gym.sync_frame_time(self.sim)
        
        return True  # Continue simulation
    def _draw_lidar_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines

        
        #self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))


        if self.sensor_cfg.pointcloud_in_world_frame:
            self.global_pixels =  self.downsampled_cloud
            for i in range(self.selected_env_idx,self.selected_env_idx+1):
                for j in range(int(self.global_pixels.shape[2])):
                    for k in range(self.global_pixels.shape[3]):
                        x = self.global_pixels[i, 0,j,k,0]#+self.root_states[:1, 0]
                        y = self.global_pixels[i, 0,j,k,1]
                        z = self.global_pixels[i, 0,j,k,2]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        else:
            self.local_pixels_downsampled = self.downsampled_cloud.reshape(-1, 3)
            self.sensor_axis= self.sensor_pos_tensor[:,:]       
            pixels = self.local_pixels_downsampled.view(self.num_envs,-1,3)
            pixels_num = pixels.shape[1]
            sensor_axis_shaped = self.sensor_axis.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 3)
            sensor_quat = self.sensor_quat_tensor.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 4)
            self.global_pixels = sensor_axis_shaped + quat_apply(sensor_quat, pixels)
            
            #def draw_line(p1, p2, color, gym, viewer, env):

            self.global_pixels.view(self.num_envs,-1, 3)
            for i in range(self.selected_env_idx,self.selected_env_idx+1):
                for j in range(0,self.global_pixels.shape[1]):
                        x = self.global_pixels[i, j,0]
                        y = self.global_pixels[i, j,1]
                        z = self.global_pixels[i, j,2]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _draw_sphere_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # def draw_line(p1, p2, color, gym, viewer, env):
        #     verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
        #     verts[0][0] = (p1.x, p1.y, p1.z)
        #     verts[0][1] = (p2.x, p2.y, p2.z)
        #     colors = np.empty(1, dtype=gymapi.Vec3.dtype)
        #     colors[0] = (color.x, color.y, color.z)
        #     gym.add_lines(viewer, env, 1, verts, colors)

        
        #self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        self.downsampled_sphere_points_worlds = self.downsampled_sphere_points.view(self.num_envs,-1, 3)+self.sensor_pos_tensor[:,:].view(self.num_envs,-1, 3).repeat(1, self.downsampled_sphere_points.shape[1], 1).view(-1, 3)
        for i in range(0,1):
            for j in range(0,self.downsampled_sphere_points_worlds.shape[1]):
                    end_x = self.downsampled_sphere_points_worlds[i, j,0]
                    end_y = self.downsampled_sphere_points_worlds[i, j,1]
                    end_z = self.downsampled_sphere_points_worlds[i, j,2]
                    
                    sphere_pose = gymapi.Transform(gymapi.Vec3(end_x, end_y, end_z), r=None)
                        
    def collect_and_save_data(self):
        """收集当前时刻的数据并添加到存储列表"""
        current_time = self.sim_time
        
        # 1. 收集激光雷达局部点云数据 - 在激光雷达坐标系中
        local_pixels = self.lidar_tensor.clone()  # [num_envs, num_sensors, vertical_lines, horizontal_lines, 3]
        
        # 2. 收集机器人位置 - 世界坐标系
        robot_positions = self.root_states[:, 0:3].clone()  # [num_envs, 3]
        
        # 3. 收集机器人方向 (四元数) - 世界坐标系
        robot_orientations = self.root_states[:, 3:7].clone()  # [num_envs, 4]
        
        # 4. 收集地形高度测量值 - 世界坐标系
        terrain_heights = self.measured_heights.clone()  # [num_envs, num_height_points]
        
        # 将当前数据添加到存储列表 (保持原始张量格式)
        self.stored_local_pixels.append(local_pixels)
        self.stored_robot_positions.append(robot_positions)
        self.stored_robot_orientations.append(robot_orientations)
        self.stored_terrain_heights.append(terrain_heights)
        self.stored_timestamps.append(current_time)
        
        # 如果列表变得太大，保存并清空
        if len(self.stored_timestamps) >= 10:  # 每1000帧保存一次
            self.save_data_to_files()

    def save_data_to_files(self):
        """将收集的数据保存到文件中并清空存储列表"""
        if not self.stored_timestamps:
            return  # 如果没有数据，直接返回
        
        # 生成时间戳字符串作为文件名的一部分
        timestamp_str = f"{self.stored_timestamps[0]:.2f}_{self.stored_timestamps[-1]:.2f}"
        
        # 将存储的列表转换为张量
        # 注意：这里我们堆叠张量以创建时间序列数据
        local_pixels_tensor = torch.stack(self.stored_local_pixels)
        robot_positions_tensor = torch.stack(self.stored_robot_positions)
        robot_orientations_tensor = torch.stack(self.stored_robot_orientations)
        terrain_heights_tensor = torch.stack(self.stored_terrain_heights)
        timestamps_tensor = torch.tensor(self.stored_timestamps, device=self.device)
        
        # 创建数据字典
        data_dict = {
            'local_pixels': local_pixels_tensor,
            'robot_positions': robot_positions_tensor,
            'robot_orientations': robot_orientations_tensor, 
            'terrain_heights': terrain_heights_tensor,
            'timestamps': timestamps_tensor
        }
        
        # 使用torch.save保存字典
        torch.save(data_dict, f"{self.data_dir}/sensor_data_{timestamp_str}.pt")
        
        print(f"Saved {len(self.stored_timestamps)} frames of data with timestamp {timestamp_str}")
        
        # 清空存储列表
        self.stored_local_pixels = []
        self.stored_robot_positions = []
        self.stored_robot_orientations = []
        self.stored_terrain_heights = []
        self.stored_timestamps = []

    # 添加析构函数确保数据保存
    def __del__(self):
        """确保在对象销毁前保存所有数据"""
        if hasattr(self, 'save_data') and self.save_data and hasattr(self, 'stored_timestamps') and self.stored_timestamps:
            print("Saving remaining data before exit...")
            self.save_data_to_files()
def print_lidar_pos():
    """Get and print the lidar position data."""
    # Update the sensor to get the latest data
   
    
    # Print lidar position
    # The env.lidar_tensor contains the lidar data we want to print
    print(f"Lidar Position at {time.time():.3f}:")
    
    # Example: Print a summary of the lidar position data
    # You can customize this to print specific parts of the data that are of interes
    print("-" * 50)
    timer = threading.Timer(0.02, print_lidar_pos)
    timer.daemon = True
    timer.start()
    
if __name__ == "__main__":
    # Create and run a simple lidar environment
    env = G1Env(
        num_envs=4,
        num_obstacles=10
    )
    
    
    print("LidarSensorEnv created!")
    print("Use WASD keys to move the lidar sensor horizontally")
    print("Use Q/E keys to move the lidar sensor up/down")

    # Run simulation loop
    try:
        while True:
            start = time.time()
            lidar_data = env.step()
            # Add a small sleep to prevent hogging the CPU
            cost_time= time.time()-start
            #time.sleep(0.01)
    except KeyboardInterrupt:
        print("Simulation stopped by user")
