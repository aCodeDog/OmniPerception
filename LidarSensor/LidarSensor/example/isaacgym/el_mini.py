import numpy as np
import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from math import sqrt
import torch
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig
import random
import time
import trimesh
import warp as wp
import threading
from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain
from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg
from LidarSensor import SENSOR_ROOT_DIR, RESOURCES_DIR
import os

# 视频保存函数同之前

def save_video(frame_stack, key, format=None, fps=20, **imageio_kwargs):
    if format:
        key += "." + format
    else:
        _, format = os.path.splitext(key)
        if format:
            format = format[1:]
        else:
            format = "mp4"
            key += "." + format

    import tempfile, imageio
    from skimage import img_as_ubyte
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        try:
            imageio.mimsave(key, img_as_ubyte(frame_stack), format=format, fps=fps, macro_block_size=1, **imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(key, img_as_ubyte(frame_stack), format=format, fps=fps, macro_block_size=1, **imageio_kwargs)
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

def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

def euler_from_quaternion(quat_angle):
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
    return roll_x, pitch_y, yaw_z

def farthest_point_sampling(point_cloud, sample_size):
    num_envs, _, num_points, _ = point_cloud.shape
    device = point_cloud.device
    result = []
    for env_idx in range(num_envs):
        points = point_cloud[env_idx, 0]  # (num_points, 3)
        sampled_indices = torch.zeros(sample_size, dtype=torch.long, device=device)
        sampled_indices[0] = torch.randint(0, num_points, (1,), device=device)
        distances = torch.norm(points - points[sampled_indices[0]], dim=1)
        for i in range(1, sample_size):
            sampled_indices[i] = torch.argmax(distances)
            if i < sample_size - 1:
                new_distances = torch.norm(points - points[sampled_indices[i]], dim=1)
                distances = torch.min(distances, new_distances)
        sampled_points = points[sampled_indices]
        result.append(sampled_points.unsqueeze(0))
    return torch.stack(result)

def downsample_spherical_points_vectorized(sphere_points, num_theta_bins=10, num_phi_bins=10):
    num_envs = sphere_points.shape[0]
    num_points = sphere_points.shape[1]
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.12, 0.907
    r = sphere_points[:, :, 0]
    theta = sphere_points[:, :, 1]
    phi = sphere_points[:, :, 2]
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
    bin_indices = theta_bin * num_phi_bins + phi_bin
    r_sum = torch.zeros(num_envs, num_bins, device=device)
    bin_count = torch.zeros(num_envs, num_bins, device=device)
    r_sum.scatter_add_(1, bin_indices, r)
    ones = torch.ones_like(r)
    bin_count.scatter_add_(1, bin_indices, ones)
    bin_count = torch.clamp(bin_count, min=1.0)
    avg_r = r_sum / bin_count
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
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)
    phi_centers_flat = phi_grid.reshape(-1)
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = avg_r
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)
    return downsampled

args = gymutil.parse_arguments(
    description="el_mini six-legged robot + Lidar demo",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 4, "help": "Number of environments"},
        {"name": "--headless", "type": bool, "default": False, "help": "Run in headless mode"}
    ]
)
headless = args.headless

class ElMiniEnv:
    def __init__(self, num_envs=1, num_obstacles=5, save_data=False, save_interval=0.1):
        self.gym = gymapi.acquire_gym()
        self.num_envs = num_envs
        self.num_obstacles = num_obstacles
        self.headless = args.headless
        self.show_viewer = not args.headless
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sensor_cfg = LidarConfig()
        self.sensor_cfg.sensor_type = "avia"
        self.sensor_cfg.update_frequency
        self.sim_time = 0
        self.sensor_update_time = 0
        self.state_update_time = 0
        self.save_data = save_data
        self.save_interval = save_interval
        self.save_time = 0
        self.sequence_number = 0
        wp.init()
        if self.save_data:
            self.data_dir = f"./sensor_data_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.data_dir, exist_ok=True)
            self.stored_local_pixels = []
            self.stored_robot_positions = []
            self.stored_robot_orientations = []
            self.stored_terrain_heights = []
            self.stored_timestamps = []

        # 雷达外参（el_mini机身中心上方0.4m，朝向前）
        self.sensor_translation_local = torch.tensor([0.0, 0.0, 0.4], device=self.device)
        self.sensor_rpy_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.sensor_offset_quat_local = quat_from_euler_xyz(
            self.sensor_rpy_offset[0], self.sensor_rpy_offset[1], self.sensor_rpy_offset[2]
        )

        self.create_sim()
        self.create_ground()
        self.create_viewer()
        self.create_env()
        self.gym.prepare_sim(self.sim)
        self._init_buffer()
        self.create_warp_env()
        self.crete_warp_tensor()
        self.sensor = LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()
        self.record_video = True
        if self.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
        self.video_frames = []
        self.key_pressed = {}
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.selected_env_idx = 0
        self.enable_random_movement = False

        if self.viewer:
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

        print("Keyboard controls:")
        print("  WASD: Move robot horizontally")
        print("  Q/E: Move robot up/down")
        print("  Arrow keys: Rotate robot")
        print("  ESC: Exit simulation")

    # 其余环境/仿真/采样/地形函数与unitree_g1.py基本一致，此处略
    # 只更换对应的urdf路径和雷达外参

    def create_sim(self):
        sim_params = gymapi.SimParams()
        dt = 0.02
        self.dt = dt
        sim_params.dt = dt
        if args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 1
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = args.num_threads if hasattr(args, 'num_threads') else 4
            sim_params.physx.use_gpu = getattr(args, "use_gpu", False)
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True
        self.sim = self.gym.create_sim(
            getattr(args, "compute_device_id", 0),
            getattr(args, "graphics_device_id", 0),
            args.physics_engine, sim_params
        )
        if self.sim is None:
            raise RuntimeError("*** Failed to create sim")

    def create_ground(self):
        self.terrain_cfg = Terrain_cfg()
        self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        self._create_trimesh()

    def _create_trimesh(self):
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
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def create_viewer(self):
        if args.headless:
            self.viewer = None
            print("Running in headless mode")
        else:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("*** Failed to create viewer")

    def _get_env_origins(self):
        if self.terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            max_init_level = self.terrain_cfg.max_init_terrain_level
            if not self.terrain_cfg.curriculum:
                max_init_level = self.terrain_cfg.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.terrain_cfg.num_cols),
                                           rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.terrain_cfg.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

    def create_env(self):
        sensor_asset_root = f"{RESOURCES_DIR}"
        sensor_asset_file = "robots/el_mini/urdf/el_mini.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False

        sensor_asset = self.gym.load_asset(self.sim, sensor_asset_root, sensor_asset_file, asset_options)
        self._get_env_origins()
        num_per_row = int(sqrt(self.num_envs))
        env_spacing = 0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.envs = []
        self.sensor_handles = []
        for i in range(self.num_envs):
            pos = self.env_origins[i].clone()
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            start_pose = gymapi.Transform()
            start_pose.r = gymapi.Quat(0, 0, 0, 1)
            pos[2] = 0.4
            start_pose.p = gymapi.Vec3(*(pos))
            sensor_handle = self.gym.create_actor(env, sensor_asset, start_pose, "el_mini", i, 1, 1)
            self.sensor_handles.append(sensor_handle)
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.sensor_handles[0])
        print("el_mini dof_names", dof_names)

    def _init_buffer(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_quat = self.root_states[:, 3:7]
        self.common_step_counter = 0
        self.up_axis_idx = 2
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
        self.measured_heights = self._get_heights()
        self.sensor_translation = self.sensor_translation_local.repeat((self.num_envs, 1))
        self.sensor_offset_quat = self.sensor_offset_quat_local.repeat((self.num_envs, 1))

    def _init_height_points(self):
        y = torch.tensor(self.terrain_cfg.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.terrain_cfg.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.terrain_cfg.measure_horizontal_noise,
                                      self.terrain_cfg.measure_horizontal_noise,
                                      (self.num_height_points, 2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.terrain_cfg.measure_horizontal_noise,
                                        self.terrain_cfg.measure_horizontal_noise,
                                        (self.num_height_points, 2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def create_warp_env(self):
        # 如果有整体mesh可补充下面el_mini_combined.stl部分，否则暂时只用terrain
        terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        transform = np.zeros((3,))
        transform[0] = -self.terrain_cfg.border_size
        transform[1] = -self.terrain_cfg.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)
        # current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # two_levels_up = os.path.dirname(os.path.dirname(current_script_dir))
        # obstacle_mesh_path = os.path.join(two_levels_up, "resources", "robots", "el_mini", "el_mini_combined.stl")
        # obstacle_mesh = trimesh.load(obstacle_mesh_path)
        # transaltion = np.zeros((3,))
        # transaltion[0]=self.root_states[0,0]
        # transaltion[1]=self.root_states[0,1]
        # transaltion[2]=self.root_states[0,2]
        # translation = trimesh.transformations.translation_matrix(transaltion)
        # obstacle_mesh.apply_transform(translation)
        # combine_mesh = trimesh.util.concatenate([terrain_mesh, obstacle_mesh])
        combine_mesh = terrain_mesh
        vertices = combine_mesh.vertices
        triangles = combine_mesh.faces
        vertex_tensor = torch.tensor(
            vertices, device=self.device, requires_grad=False, dtype=torch.float32)
        vertex_vec3_array = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32, device=self.device)
        self.wp_meshes = wp.Mesh(points=vertex_vec3_array, indices=faces_wp_int32_array)
        self.mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)

    def crete_warp_tensor(self):
        self.warp_tensor_dict = {}
        self.lidar_tensor = torch.zeros(
            (
                self.num_envs,
                self.sensor_cfg.num_sensors,
                self.sensor_cfg.vertical_line_num,
                self.sensor_cfg.horizontal_line_num,
                3,
            ),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_dist_tensor = torch.zeros(
            (
                self.num_envs,
                self.sensor_cfg.num_sensors,
                self.sensor_cfg.vertical_line_num,
                self.sensor_cfg.horizontal_line_num,
            ),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.sensor_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])
        self.sensor_translation = self.sensor_translation_local.repeat((self.num_envs, 1))
        self.sensor_offset_quat = self.sensor_offset_quat_local.repeat((self.num_envs, 1))
        self.warp_tensor_dict["sensor_dist_tensor"] = self.sensor_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.sensor_cfg.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids

    # 其余如 step/keyboard_input/render 逻辑同unitree_g1.py 直接复用
    # 只要用self.sensor_translation和self.sensor_offset_quat都来自上面定义的统一外参即可
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

    def _draw_lidar_vis(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        # 点云下采样后 shape = (num_envs, 1, N, 3)
        if not hasattr(self, 'downsampled_cloud'):
            return
        local_cloud = self.downsampled_cloud  # (num_envs, 1, N, 3)
        pixels = local_cloud.view(self.num_envs, -1, 3)
        sensor_axis = self.sensor_pos_tensor  # (num_envs, 3)
        pixels_num = pixels.shape[1]
        sensor_axis_shaped = sensor_axis.unsqueeze(1).repeat(1, pixels_num, 1)
        sensor_quat = self.sensor_quat_tensor.unsqueeze(1).repeat(1, pixels_num, 1)
        global_pixels = sensor_axis_shaped + quat_apply(sensor_quat.view(-1, 4), pixels.view(-1, 3)).view(self.num_envs, -1, 3)
        i = self.selected_env_idx
        for j in range(global_pixels.shape[1]):
            x = global_pixels[i, j, 0]
            y = global_pixels[i, j, 1]
            z = global_pixels[i, j, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def step(self):
        # 键盘或随机控制，仿真步进及Lidar采样同unitree_g1.py
        continue_sim = self.keyboard_input()
        if not continue_sim:
            return False
        self.sim_time += self.dt
        self.sensor_update_time += self.dt
        self.state_update_time += self.dt
        self.save_time += self.dt
        self.last_base_lin_vel = getattr(self, "base_lin_vel", torch.zeros_like(self.root_states[:,0:3])).clone()
        self.last_base_ang_vel = getattr(self, "base_ang_vel", torch.zeros_like(self.root_states[:,0:3])).clone()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pose = self.root_states[:, :7]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.measured_heights = self._get_heights()
        sensor_quat = quat_mul(self.root_states[:, 3:7], self.sensor_offset_quat)
        sensor_pos = self.root_states[:, 0:3] + quat_apply(self.root_states[:, 3:7], self.sensor_translation)
        self.sensor_pos_tensor[:] = sensor_pos[:]
        self.sensor_quat_tensor[:] = sensor_quat[:]
        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()
        self.downsampled_cloud = farthest_point_sampling(
            self.lidar_tensor.view(self.num_envs,1,self.lidar_tensor.shape[2],3), sample_size=5000)
        self.sphere_points = cart2sphere(self.lidar_tensor.view(-1,3)).view(self.num_envs,-1,3)
        self.downsampled_sphere_points = downsample_spherical_points_vectorized(self.sphere_points, 10, 10)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.record_video:
            # 仅简单保存render帧
            self.gym.render_all_camera_sensors(self.sim)
            frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
            frame = frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self.video_frames.append(frame)
            if len(self.video_frames) > 250:
                save_video(self.video_frames, "videos/el_mini_lidar.mp4", fps=50)
                print("save video!!")
                self.video_frames = []
        if self.show_viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        # === 点云可视化 ===
        if self.sensor_update_time > 1 / self.sensor_cfg.update_frequency:
            self.gym.clear_lines(self.viewer)
            self._draw_lidar_vis()
            self.sensor_update_time = 0

        if self.show_viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        # FIX
        print("lidar_tensor min", self.lidar_tensor.min(), "max", self.lidar_tensor.max(), "mean", self.lidar_tensor.mean())
        return True

    def keyboard_input(self):
        if not self.viewer:
            return True
        for evt in self.gym.query_viewer_action_events(self.viewer):
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
        dt = 0.005
        env_idx = self.selected_env_idx
        current_pos = self.root_states[env_idx, 0:3].clone()
        current_quat = self.root_states[env_idx, 3:7].clone()
        self.linear_speed = 3.0
        self.angular_speed = 3.0
        linear_vel = torch.zeros(3, device=self.device)
        euler_rates = torch.zeros(3, device=self.device)
        if self.key_pressed.get(KEY_W, False):
            linear_vel[0] = self.linear_speed
        if self.key_pressed.get(KEY_S, False):
            linear_vel[0] = -self.linear_speed
        if self.key_pressed.get(KEY_A, False):
            linear_vel[1] = -self.linear_speed
        if self.key_pressed.get(KEY_D, False):
            linear_vel[1] = self.linear_speed
        if self.key_pressed.get(KEY_Q, False):
            linear_vel[2] = self.linear_speed
        if self.key_pressed.get(KEY_E, False):
            linear_vel[2] = -self.linear_speed
        if self.key_pressed.get(KEY_LEFT, False):
            euler_rates[2] = self.angular_speed
        if self.key_pressed.get(KEY_RIGHT, False):
            euler_rates[2] = -self.angular_speed
        if self.key_pressed.get(KEY_UP, False):
            euler_rates[1] = self.angular_speed
        if self.key_pressed.get(KEY_DOWN, False):
            euler_rates[1] = -self.angular_speed
        global_vel = quat_apply(current_quat, linear_vel)
        new_pos = current_pos + global_vel * dt
        roll, pitch, yaw = euler_from_quaternion(current_quat.unsqueeze(0))
        roll = roll + euler_rates[0] * dt
        pitch = pitch + euler_rates[1] * dt
        yaw = yaw + euler_rates[2] * dt
        new_quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[env_idx, 0:3] = new_pos
        self.root_states[env_idx, 3:7] = new_quat
        env_ids_int32 = torch.tensor([env_idx], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        return True

if __name__ == "__main__":
    env = ElMiniEnv(num_envs=args.num_envs)
    print("el_mini LidarSensorEnv created!\nUse WASD keys to move the base horizontally\nUse Q/E keys for up/down")
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("Simulation stopped by user")