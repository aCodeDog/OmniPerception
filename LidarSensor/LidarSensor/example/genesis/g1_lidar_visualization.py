#!/usr/bin/env python3
"""
Genesis G1 Robot Environment with LidarSensor Visualization

This script demonstrates proper lidar point visualization in Genesis with G1 robot.
Key features:
1. Proper coordinate transformation handling Warp (xyzw) vs Genesis (wxyz) quaternions
2. Real-time lidar point cloud visualization
3. Multiple sensor types support
4. Clean visualization with color-coded distance information
"""

import torch
import numpy as np
import genesis as gs
import warp as wp
from typing import Optional, Tuple, List
import os
import sys
import math

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


def quat_genesis_to_warp(genesis_quat):
    """Convert Genesis quaternion (wxyz) to Warp quaternion (xyzw)"""
    # Genesis: (w, x, y, z) -> Warp: (x, y, z, w)
    if len(genesis_quat.shape) == 1:
        return torch.tensor([genesis_quat[1], genesis_quat[2], genesis_quat[3], genesis_quat[0]], 
                          device=genesis_quat.device, dtype=genesis_quat.dtype)
    else:
        return torch.stack([genesis_quat[:, 1], genesis_quat[:, 2], genesis_quat[:, 3], genesis_quat[:, 0]], dim=1)


def quat_warp_to_genesis(warp_quat):
    """Convert Warp quaternion (xyzw) to Genesis quaternion (wxyz)"""
    # Warp: (x, y, z, w) -> Genesis: (w, x, y, z)
    if len(warp_quat.shape) == 1:
        return torch.tensor([warp_quat[3], warp_quat[0], warp_quat[1], warp_quat[2]], 
                          device=warp_quat.device, dtype=warp_quat.dtype)
    else:
        return torch.stack([warp_quat[:, 3], warp_quat[:, 0], warp_quat[:, 1], warp_quat[:, 2]], dim=1)


def quat_apply_genesis(quat, vec):
    """Apply quaternion rotation using Genesis quaternion format (wxyz)"""
    # quat: (w, x, y, z), vec: (x, y, z)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Extract vector components
    vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
    
    # Quaternion rotation formula
    # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qxyz = torch.stack([x, y, z], dim=-1)
    cross1 = torch.cross(qxyz, vec, dim=-1) + w.unsqueeze(-1) * vec
    cross2 = torch.cross(qxyz, cross1, dim=-1)
    result = vec + 2 * cross2
    
    return result


def quat_mul_genesis(q1, q2):
    """Multiply two quaternions in Genesis format (wxyz)"""
    # q1, q2: (w, x, y, z)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


class GenesisG1LidarVisualizer:
    """Genesis G1 Environment with advanced lidar visualization"""
    
    def __init__(self, 
                 num_envs: int = 1,
                 device: str = 'cuda:0',
                 headless: bool = False,
                 sensor_type: LidarType = LidarType.MID360,
                 visualization_mode: str = 'spheres'):  # 'spheres' or 'lines'
        
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.sensor_type = sensor_type
        self.visualization_mode = visualization_mode
        self.dt = 0.02
        
        print(f"Initializing Genesis G1 Lidar Visualizer:")
        print(f"  - Environments: {num_envs}")
        print(f"  - Device: {device}")
        print(f"  - Sensor: {sensor_type.value}")
        print(f"  - Visualization: {visualization_mode}")
        
        # Initialize state
        self.episode_length = 0
        self.sim_time = 0.0
        self.lidar_update_counter = 0
        
        # Initialize components
        self.scene = None
        self.robot = None
        self.terrain = None
        self.obstacles = []
        self.lidar_sensor = None
        
        # Robot state
        self.base_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.base_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        
        # Initialize base quaternion (Genesis format: wxyz)
        self.base_quat[:, 0] = 1.0  # w = 1
        
        # Lidar visualization state
        self.current_points = None
        self.current_distances = None
        self.point_colors = None
        
        # Sensor offset parameters (similar to IsaacGym version)# 10cm forward, 30cm up
        self.sensor_offset_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)  # No rotation offset (wxyz)
        self.sensor_translation = torch.tensor([0., 0.0, 0.436], device=self.device)
        
        # Create environment
        self._create_genesis_scene()
        self._setup_environment()
        self._setup_robot()
        self._init_robot_state()
        
        # Initialize lidar after Genesis is ready
        print("Initializing lidar sensor...")
        self._setup_lidar_sensor()
        
        print("Initialization complete!")
    
    def _create_genesis_scene(self):
        """Create Genesis scene"""
        print("Creating Genesis scene...")
        
        try:
            gs.init(logging_level="warning")
        except Exception as e:
            if "already initialized" in str(e):
                print("Genesis already initialized")
            else:
                raise e
        
        # Create scene
        if self.headless:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=self.dt, 
                    substeps=2,
                    gravity=(0.0, 0.0, -9.81)
                ),
                vis_options=gs.options.VisOptions(n_rendered_envs=min(self.num_envs, 4)),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                ),
                show_viewer=False,
            )
        else:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=self.dt, 
                    substeps=2,
                    gravity=(0.0, 0.0, -9.81)
                ),
                viewer_options=gs.options.ViewerOptions(
                    max_FPS=int(0.5 / self.dt),
                    camera_pos=(4.0, 2.0, 3.0),
                    camera_lookat=(0.0, 0.0, 1.0),
                    camera_fov=50,
                ),
                vis_options=gs.options.VisOptions(n_rendered_envs=min(self.num_envs, 4)),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                ),
                show_viewer=True,
            )
    
    def _setup_environment(self):
        """Setup environment with interesting geometry for lidar scanning"""
        print("Setting up environment...")
        
        # Ground plane
        self.terrain = self.scene.add_entity(
            gs.morphs.Plane()
        )
        
        # Add more interesting obstacles for lidar visualization
        self._add_visualization_obstacles()
    
    def _add_visualization_obstacles(self):
        """Add various obstacles for interesting lidar visualization"""
        # Walls to create interesting lidar patterns
        wall_configs = [
            # Front wall
            {'pos': (3.0, 0.0, 1.0), 'size': (0.1, 4.0, 2.0)},
            # Side walls
            {'pos': (0.0, 3.0, 1.0), 'size': (6.0, 0.1, 2.0)},
            {'pos': (0.0, -3.0, 1.0), 'size': (6.0, 0.1, 2.0)},
            # Some pillars
            {'pos': (2.0, 1.5, 1.0), 'size': (0.3, 0.3, 2.0)},
            {'pos': (2.0, -1.5, 1.0), 'size': (0.3, 0.3, 2.0)},
            {'pos': (-1.0, 0.0, 0.5), 'size': (0.5, 0.5, 1.0)},
        ]
        
        # Spheres for variety
        sphere_configs = [
            {'pos': (1.5, 0.8, 0.3), 'radius': 0.3},
            {'pos': (1.5, -0.8, 0.3), 'radius': 0.3},
            {'pos': (-2.0, 1.0, 0.4), 'radius': 0.4},
        ]
        
        self.obstacles = []
        
        # Add walls/boxes
        for config in wall_configs:
            obstacle = self.scene.add_entity(
                gs.morphs.Box(
                    size=config['size'],
                    pos=config['pos'],
                    fixed=True
                )
            )
            self.obstacles.append(obstacle)
        
        # Add spheres
        # for config in sphere_configs:
        #     obstacle = self.scene.add_entity(
        #         gs.morphs.Sphere(
        #             radius=config['radius'],
        #             pos=config['pos'],
        #             fixed=True
        #         )
        #     )
        #     self.obstacles.append(obstacle)
        
        print(f"Added {len(self.obstacles)} visualization obstacles")
    
    def _setup_robot(self):
        """Setup G1 robot or fallback"""
        print("Setting up robot...")
        
        urdf_path = self._find_g1_urdf()
        
        if urdf_path:
            print(f"Loading G1 URDF: {urdf_path}")
            try:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=urdf_path,
                        pos=(0.0, 0.0, 1.0),
                        fixed=True
                    )
                )
            except Exception as e:
                print(f"G1 URDF loading failed: {e}")
                self.robot = self._create_simple_robot()
        else:
            print("G1 URDF not found, using simple robot")
            self.robot = self._create_simple_robot()
    
    def _find_g1_urdf(self) -> Optional[str]:
        """Find G1 URDF file"""
        paths = [
            "LidarSensor/resources/robots/g1_29/g1_29dof.urdf",
            "../resources/robots/g1_29/g1_29dof.urdf", 
            "../../resources/robots/g1_29/g1_29dof.urdf",
        ]   
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    
    def _create_simple_robot(self):
        """Create simple robot as fallback"""
        return self.scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.2, 0.8),
                pos=(0.0, 0.0, 1.0),
                fixed=False
            )
        )
    
    def _init_robot_state(self):
        """Initialize robot state"""
        print("Initializing robot state...")
        
        # Build scene
        self.scene.build(n_envs=self.num_envs)
        
        # Update state
        self._update_robot_state()
    
    def _update_robot_state(self):
        """Update robot state from Genesis"""
        if self.robot is not None:
            try:
                positions = self.robot.get_pos()
                quaternions = self.robot.get_quat()  # Genesis format: wxyz
                
                # Handle single env case
                if positions.dim() == 1:
                    positions = positions.unsqueeze(0)
                if quaternions.dim() == 1:
                    quaternions = quaternions.unsqueeze(0)
                
                self.base_pos[:] = positions
                self.base_quat[:] = quaternions  # Keep in Genesis format
                
                # Get velocities if available
                try:
                    lin_vel = self.robot.get_vel()
                    ang_vel = self.robot.get_ang()
                    
                    if lin_vel.dim() == 1:
                        lin_vel = lin_vel.unsqueeze(0)
                    if ang_vel.dim() == 1:
                        ang_vel = ang_vel.unsqueeze(0)
                    
                    self.base_lin_vel[:] = lin_vel
                    self.base_ang_vel[:] = ang_vel
                except:
                    self.base_lin_vel.zero_()
                    self.base_ang_vel.zero_()
                    
            except Exception as e:
                print(f"Warning: Robot state update failed: {e}")
                self.base_pos[:, 2] = 1.0
                self.base_quat[:, 0] = 1.0
    
    def _setup_lidar_sensor(self):
        """Setup LidarSensor with proper parameters"""
        print(f"Setting up LidarSensor: {self.sensor_type.value}")
        
        try:
            # Initialize Warp
            wp.init()
            
            # Create lidar config
            sensor_config = LidarConfig(
                sensor_type=self.sensor_type,  # Choose your sensor type
                dt=0.02,  # CRITICAL: Must match simulation dt
                max_range=20.0,
                update_frequency=1.0/0.02,  # Update every simulation step
                return_pointcloud=True,
                pointcloud_in_world_frame=False,  # Get local coordinates first
                enable_sensor_noise=False,  # Disable for faster processing
            )

            
            # For grid-based sensors, set reasonable resolution
            if self.sensor_type == LidarType.SIMPLE_GRID:
                sensor_config.horizontal_line_num = 64
                sensor_config.vertical_line_num = 32
                sensor_config.horizontal_fov_deg_min = -90
                sensor_config.horizontal_fov_deg_max = 90
                sensor_config.vertical_fov_deg_min = -30
                sensor_config.vertical_fov_deg_max = 30
            elif self.sensor_type == LidarType.HEIGHT_SCANNER:
                # Configure height scanner parameters
                sensor_config.height_scanner_size = [8.0, 2.0]  # 4m x 2m grid (reasonable size for terrain mapping)
                sensor_config.height_scanner_resolution = 0.05   # 20cm spacing (less dense, more reasonable)
                sensor_config.height_scanner_direction = [0.0, 0.0, -1.0]  # downward rays
                sensor_config.height_scanner_ordering = "xy"
                sensor_config.height_scanner_offset = [0.0, 0.0]  # 1m forward offset (scan ahead of robot)
                sensor_config.height_scanner_height_above_ground = 12.0  # start rays 2m above robot base
                print(f"Height scanner configured: size={sensor_config.height_scanner_size}, resolution={sensor_config.height_scanner_resolution}")
                print(f"Height scanner offset: {sensor_config.height_scanner_offset}, height: {sensor_config.height_scanner_height_above_ground}m")
            
            # Create environment data for LidarSensor
            env_data = self._create_lidar_env_data()
            
            # Debug: Check environment data
            print(f"Environment data:")
            print(f"  sensor_pos_tensor: {env_data['sensor_pos_tensor']}")
            print(f"  sensor_quat_tensor: {env_data['sensor_quat_tensor']}")
            print(f"  vertices shape: {env_data['vertices'].shape}")
            print(f"  faces shape: {env_data['faces'].shape}")
            print(f"  mesh_ids: {env_data['mesh_ids']}")
            
            # Create LidarSensor with correct parameters
            self.lidar_sensor = LidarSensor(
                env=env_data,
                env_cfg={'sensor_noise': False},
                sensor_config=sensor_config,
                num_sensors=1,
                device=self.device
            )
            
            print("LidarSensor initialized successfully!")
            
        except Exception as e:
            print(f"LidarSensor setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.lidar_sensor = None
    
    def _create_lidar_env_data(self) -> dict:
        """Create environment data for LidarSensor"""
        # Create mock mesh data - in real usage, extract from Genesis scene
        vertices, faces = self._generate_scene_mesh()
        import trimesh
        
        save_stl_path = os.path.join(current_dir, "scene_mesh.stl")
        # Save vertices and faces to trimesh STL
        if not os.path.exists(os.path.dirname(save_stl_path)):
            os.makedirs(os.path.dirname(save_stl_path))
        # Save vertices and faces to trimesh STL
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(save_stl_path)
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
        faces_wp_int32_array = wp.from_numpy(faces.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)
        # Calculate sensor position and orientation with proper offsets (like IsaacGym)
        sensor_quat = quat_mul_genesis(self.base_quat, self.sensor_offset_quat.unsqueeze(0).expand(self.num_envs, -1))
        sensor_pos = self.base_pos + quat_apply_genesis(self.base_quat, self.sensor_translation.unsqueeze(0).expand(self.num_envs, -1))
        
        # Convert Genesis quaternion to Warp format for sensor
        sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
        
        # # Create mesh IDs (simple for now)
        # mesh_ids = torch.zeros(len(faces), dtype=torch.int32, device=self.device)
        
        return {
            'num_envs': self.num_envs,
            'sensor_pos_tensor': sensor_pos,
            'sensor_quat_tensor': sensor_quat_warp,  # Warp format
            'vertices': vertices,
            'faces': faces,
            'mesh_ids': mesh_ids
        }
    
    def _generate_scene_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simplified mesh representation of the scene"""
        vertices = []
        faces = []
        face_idx = 0
        
        # Ground plane (large)
        ground_verts = np.array([
            [-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]
        ], dtype=np.float32)
        vertices.extend(ground_verts)
        faces.extend([[0, 1, 2], [0, 2, 3]])
        face_idx += 4
        
        #print(f"DEBUG: Ground plane added - vertices: {len(ground_verts)}, faces: 2")
        
        # Add obstacle meshes (simplified boxes and spheres)
        # Walls
        wall_configs = [
            {'pos': (3.0, 0.0, 1.0), 'size': (0.1, 4.0, 2.0)},
            {'pos': (0.0, 3.0, 1.0), 'size': (6.0, 0.1, 2.0)},
            {'pos': (0.0, -3.0, 1.0), 'size': (6.0, 0.1, 2.0)},
            {'pos': (2.0, 1.5, 1.0), 'size': (0.3, 0.3, 2.0)},
            {'pos': (2.0, -1.5, 1.0), 'size': (0.3, 0.3, 2.0)},
            {'pos': (-1.0, 0.0, 0.5), 'size': (0.5, 0.5, 1.0)},
        ]
        
        # Add box vertices and faces
        for config in wall_configs:
            x, y, z = config['pos']
            sx, sy, sz = config['size']
            
            # Box vertices
            box_verts = np.array([
                [x-sx/2, y-sy/2, z-sz/2], [x+sx/2, y-sy/2, z-sz/2],
                [x+sx/2, y+sy/2, z-sz/2], [x-sx/2, y+sy/2, z-sz/2],
                [x-sx/2, y-sy/2, z+sz/2], [x+sx/2, y-sy/2, z+sz/2],
                [x+sx/2, y+sy/2, z+sz/2], [x-sx/2, y+sy/2, z+sz/2],
            ], dtype=np.float32)
            
            vertices.extend(box_verts)
            
            # Box faces
            box_faces = [
                # Bottom face
                [face_idx, face_idx+1, face_idx+2], [face_idx, face_idx+2, face_idx+3],
                # Top face
                [face_idx+4, face_idx+7, face_idx+6], [face_idx+4, face_idx+6, face_idx+5],
                # Side faces
                [face_idx, face_idx+4, face_idx+5], [face_idx, face_idx+5, face_idx+1],
                [face_idx+1, face_idx+5, face_idx+6], [face_idx+1, face_idx+6, face_idx+2],
                [face_idx+2, face_idx+6, face_idx+7], [face_idx+2, face_idx+7, face_idx+3],
                [face_idx+3, face_idx+7, face_idx+4], [face_idx+3, face_idx+4, face_idx],
            ]
            faces.extend(box_faces)
            face_idx += 8
        
        final_vertices = np.array(vertices, dtype=np.float32)
        final_faces = np.array(faces, dtype=np.int32)
        
        # print(f"DEBUG: Final mesh - vertices: {final_vertices.shape}, faces: {final_faces.shape}")
        # print(f"DEBUG: Vertex range - X: [{final_vertices[:, 0].min():.2f}, {final_vertices[:, 0].max():.2f}]")
        # print(f"DEBUG: Vertex range - Y: [{final_vertices[:, 1].min():.2f}, {final_vertices[:, 1].max():.2f}]") 
        # print(f"DEBUG: Vertex range - Z: [{final_vertices[:, 2].min():.2f}, {final_vertices[:, 2].max():.2f}]")
        
        return final_vertices, final_faces
    
    def step(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Step simulation and update lidar"""
        # Update robot state
        self._update_robot_state()
        
        # Update lidar sensor
        point_cloud, distances = None, None
        if self.lidar_sensor is not None:
            try:
                # Update sensor pose with proper offset calculation (like IsaacGym)
                sensor_quat = quat_mul_genesis(self.base_quat, self.sensor_offset_quat.unsqueeze(0).expand(self.num_envs, -1))
                sensor_pos = self.base_pos + quat_apply_genesis(self.base_quat, self.sensor_translation.unsqueeze(0).expand(self.num_envs, -1))
                sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
                
                self.lidar_sensor.lidar_positions_tensor[:] = sensor_pos
                self.lidar_sensor.lidar_quat_tensor[:] = sensor_quat_warp
                
                # Get lidar data
                point_cloud, distances = self.lidar_sensor.update()
                
                # Store for visualization
                if point_cloud is not None:
                    self.current_points = point_cloud.clone()
                    self.current_distances = distances.clone()
                    self.lidar_update_counter += 1
                
            except Exception as e:
                print(f"Lidar update failed: {e}")
        
        # Step physics
        if self.scene is not None:
            self.scene.step()
            self.sim_time += self.dt
            self.episode_length += 1
        
        # Visualize lidar points
        if self.current_points is not None and self.episode_length % 5 == 0:
            self._visualize_lidar_points()
        
        return point_cloud, distances
    
    def _visualize_lidar_points(self):
        """Visualize lidar points in Genesis scene"""
        if self.current_points is None or self.scene is None:
            return
        
        try:
            # Get points from first environment for visualization
            points = self.current_points[0]  # Shape: (num_points, 3)
            distances = self.current_distances[0].view(-1,1).squeeze()  # Shape: (num_points,)
            
            # Transform points to world coordinates using Genesis quaternion format
            # Current points are in sensor local frame, need to transform to world
            # Use proper sensor offset calculation
            base_quat_single = self.base_quat[0]
            sensor_quat = quat_mul_genesis(base_quat_single.unsqueeze(0), self.sensor_offset_quat.unsqueeze(0))[0]
            sensor_pos = self.base_pos[0] + quat_apply_genesis(base_quat_single.unsqueeze(0), self.sensor_translation.unsqueeze(0))[0]
            
            points = points.view(-1,3)
            sensor_quat = sensor_quat.repeat(points.shape[0],1)
            # Apply rotation and translation
            world_points = quat_apply_genesis(sensor_quat, points) + sensor_pos
            
            # Filter points by distance (remove invalid/far points)
            valid_mask = (distances > 0.1) & (distances < 20.0)
            if valid_mask.sum() == 0:
                return
            
            world_points = world_points[valid_mask]
            valid_distances = distances[valid_mask]
            
            # Sample points for visualization (too many points can slow down rendering)
            max_points = 20000
            if len(world_points) > max_points:
                indices = torch.randperm(len(world_points))[:max_points]
                world_points = world_points[indices]
                valid_distances = valid_distances[indices]
            
            # Generate colors based on distance
            colors = self._generate_distance_colors(valid_distances)
            # Visualize based on mode
            if self.visualization_mode == 'spheres':
                self._draw_point_spheres(world_points, colors)
            elif self.visualization_mode == 'lines':
                self._draw_point_lines(world_points, sensor_pos)
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _generate_distance_colors(self, distances: torch.Tensor) -> List[Tuple[float, float, float, float]]:
        """Generate colors based on distance values"""
        # Normalize distances to 0-1
        min_dist, max_dist = 0.5, 8.0
        normalized = torch.clamp((distances - min_dist) / (max_dist - min_dist), 0, 1)
        
        colors = []
        for dist in normalized:
            # Color from red (close) to blue (far)
            r = 1.0 - dist.item()
            g = 0.5
            b = dist.item()
            a = 0.8
            colors.append((r, g, b, a))
        
        return colors
    
    def _draw_point_spheres(self, points: torch.Tensor, colors: List[Tuple[float, float, float, float]]):
        """Draw lidar points as colored spheres"""
        self.scene.clear_debug_objects()
        
        self.scene.draw_debug_spheres(
            poss=points,
            radius=0.02,
            color=colors[0]
        )

    def _draw_point_lines(self, points: torch.Tensor, sensor_pos: torch.Tensor):
        """Draw lidar points as lines from sensor"""
        self.scene.clear_debug_objects()
        
        # Draw lines from sensor to hit points
        for point in points:
            line_points = torch.stack([sensor_pos, point])
            self.scene.draw_debug_lines(
                poss=line_points.unsqueeze(0),
                color=(0.0, 1.0, 0.0, 0.6)
            )
    
    def reset(self):
        """Reset environment"""
        print("Resetting environment...")
        
        self.episode_length = 0
        self.sim_time = 0.0
        self.lidar_update_counter = 0
        
        # Reset robot pose
        self.base_pos[:, :2] = torch.randn((self.num_envs, 2), device=self.device) * 0.3
        self.base_pos[:, 2] = 1.0
        self.base_quat.zero_()
        self.base_quat[:, 0] = 1.0  # w=1 (Genesis format)
        
        # Apply to robot
        if self.robot is not None:
            try:
                self.robot.set_pos(self.base_pos, zero_velocity=True)
                self.robot.set_quat(self.base_quat, zero_velocity=True)
            except Exception as e:
                print(f"Robot reset failed: {e}")
        
        self._update_robot_state()
        
        # Clear visualization
        if self.scene is not None:
            self.scene.clear_debug_objects()
        
        print("Reset complete")
    
    def close(self):
        """Clean up"""
        print("Closing environment...")
        
        if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
            try:
                del self.lidar_sensor
            except:
                pass
        
        try:
            wp.synchronize()
        except:
            pass
        
        self.scene = None
        print("Environment closed")


def main():
    """Main demonstration"""
    print("=== Genesis G1 Lidar Visualization Demo ===")
    
    # Test configurations
    configs = [
        {
            'sensor_type': LidarType.HEIGHT_SCANNER,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'visualization_mode': 'spheres'
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Demo {i+1}: {config['sensor_type'].value} ---")
        
        try:
            # Create environment
            env = GenesisG1LidarVisualizer(
                num_envs=1,
                device=config['device'],
                headless=False,
                sensor_type=config['sensor_type'],
                visualization_mode=config['visualization_mode']
            )
            
            # Reset and run
            env.reset()
            
            print(f"Running visualization demo with {config['sensor_type'].value}...")
            print("Watch the colored spheres representing lidar points!")
            print("Red = close, Blue = far")
            
            # Simulation loop
            for step in range(1000):
                point_cloud, distances = env.step()
                
                # Move robot in circle for interesting visualization
                if step % 50 == 0 and env.robot is not None:
                    try:
                        angle = step * 0.05
                        angle_tensor = torch.tensor(angle, device=env.device)
                        
                        new_pos = env.base_pos.clone()
                        new_pos[:, 0] = 0.5 * torch.cos(angle_tensor)
                        new_pos[:, 1] = 0.5 * torch.sin(angle_tensor)
                        env.robot.set_pos(new_pos)
                        
                        # Also rotate robot
                        yaw_tensor = torch.tensor(angle/2, device=env.device)
                        new_quat = env.base_quat.clone()
                        new_quat[:, 0] = torch.cos(yaw_tensor)  # w
                        new_quat[:, 3] = torch.sin(yaw_tensor)  # z
                        env.robot.set_quat(new_quat)
                        
                    except Exception as e:
                        print(f"Robot movement failed: {e}")
                
                # Print status
                if point_cloud is not None and step % 50 == 0:
                    num_points = point_cloud.shape[1] if len(point_cloud.shape) > 1 else len(point_cloud)
                    dist_range = f"[{distances.min().item():.2f}, {distances.max().item():.2f}]"
                    print(f"  Step {step}: {num_points} points, distance range: {dist_range}")
                    
                    # Debug: print point cloud shape and sample values
                    # print(f"    DEBUG: point_cloud.shape = {point_cloud.shape}")
                    # print(f"    DEBUG: distances.shape = {distances.shape}")
                    # if len(point_cloud.shape) > 2:
                    #     print(f"    DEBUG: sample points = {point_cloud[0, :3, :3]}")  # First env, first 3x3 points
                    #     print(f"    DEBUG: sample distances = {distances[0, :3, :3]}")  # First env, first 3x3 distances
            
            print(f"✓ Demo {i+1} completed!")
            
        except KeyboardInterrupt:
            print("Demo interrupted by user")
            break
        except Exception as e:
            print(f"✗ Demo {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                env.close()
            except:
                pass
    
    print("\n=== Visualization demo completed ===")


if __name__ == "__main__":
    main() 