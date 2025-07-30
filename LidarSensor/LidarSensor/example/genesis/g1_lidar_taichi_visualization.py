#!/usr/bin/env python3
"""
Genesis G1 Robot Environment with Taichi LidarSensor Visualization

This script demonstrates lidar point visualization in Genesis with G1 robot using Taichi kernels.
Key features:
1. Uses Taichi-based ray casting instead of Warp
2. Proper coordinate transformation handling 
3. Real-time lidar point cloud visualization
4. Integration with Genesis physics simulation
5. Clean visualization with color-coded distance information
"""

import torch
import numpy as np
import genesis as gs
import taichi as ti
from typing import Optional, Tuple, List
import os
import sys
import math
import time

# # Add parent directories to path for imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.insert(0, parent_dir)

# Import Taichi lidar components
from LidarSensor.sensor_kernels.lidar_kernels_taichi import LidarTaichiKernels, create_lidar_taichi_kernels


from LidarSensor.sensor_kernels.lidar_example_taichi import LidarWrapper


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


def convert_genesis_quat_to_taichi(genesis_quat):
    """Convert Genesis quaternion (wxyz) to Taichi quaternion (xyzw)"""
    # Genesis: (w, x, y, z) -> Taichi: (x, y, z, w)
    if len(genesis_quat.shape) == 1:
        return torch.tensor([genesis_quat[1], genesis_quat[2], genesis_quat[3], genesis_quat[0]], 
                          device=genesis_quat.device, dtype=genesis_quat.dtype)
    else:
        return torch.stack([genesis_quat[:, 1], genesis_quat[:, 2], genesis_quat[:, 3], genesis_quat[:, 0]], dim=1)


class GenesisG1TaichiLidarVisualizer:
    """Genesis G1 Environment with Taichi-based lidar visualization"""
    
    def __init__(self, 
                 num_envs: int = 1,
                 device: str = 'cuda:0',
                 headless: bool = False,
                 visualization_mode: str = 'spheres',  # 'spheres' or 'lines'
                 lidar_config: dict = None):
        
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.visualization_mode = visualization_mode
        self.dt = 0.02
        
        # Default lidar configuration
        self.lidar_config = lidar_config or {
            'n_scan_lines': 32,
            'n_points_per_line': 64,
            'fov_vertical': 30.0,  # degrees
            'fov_horizontal': 360.0,  # degrees
            'max_range': 20.0,
            'min_range': 0.1
        }
        
        print(f"Initializing Genesis G1 Taichi Lidar Visualizer:")
        print(f"  - Environments: {num_envs}")
        print(f"  - Device: {device}")
        print(f"  - Backend: Taichi")
        print(f"  - Visualization: {visualization_mode}")
        print(f"  - LiDAR Config: {self.lidar_config}")
        
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
        
        # Sensor offset parameters (sensor relative to robot base)
        self.sensor_offset_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)  # No rotation offset (wxyz)
        self.sensor_translation = torch.tensor([0.1, 0.0, 0.3], device=self.device)  # 10cm forward, 30cm up
        
        # Create environment
        self._create_genesis_scene()
        self._setup_environment()
        self._setup_robot()
        self._init_robot_state()
        
        # Initialize Taichi lidar after Genesis is ready
        print("Initializing Taichi lidar sensor...")
        self._setup_taichi_lidar_sensor()
        
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
        
        # Add spheres (commented out for now to keep mesh simple)
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
    
    def _setup_taichi_lidar_sensor(self):
        """Setup Taichi-based LiDAR sensor"""
        print("Setting up Taichi LiDAR sensor...")
        
        try:
            # Create Taichi lidar wrapper
            self.lidar_sensor = LidarWrapper(backend='taichi')
            
            # Extract mesh data from Genesis scene
            vertices, triangles = self._extract_scene_mesh()
            
            # Register mesh with lidar sensor
            self.lidar_sensor.register_mesh(
                mesh_id=0,
                vertices=vertices,
                triangles=triangles
            )
            
            # Create ray pattern
            self.ray_vectors = self._create_lidar_ray_pattern()
            
            print(f"Taichi LiDAR sensor initialized:")
            print(f"  - Mesh: {len(vertices)} vertices, {len(triangles)} triangles")
            print(f"  - Rays: {self.ray_vectors.shape}")
            
        except Exception as e:
            print(f"Taichi LiDAR setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.lidar_sensor = None
    
    def _extract_scene_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mesh data from Genesis scene for ray casting"""
        vertices = []
        triangles = []
        face_idx = 0
        
        # Ground plane (large)
        ground_size = 50.0
        ground_verts = np.array([
            [-ground_size, -ground_size, 0], 
            [ground_size, -ground_size, 0], 
            [ground_size, ground_size, 0], 
            [-ground_size, ground_size, 0]
        ], dtype=np.float32)
        vertices.extend(ground_verts)
        triangles.extend([[0, 1, 2], [0, 2, 3]])
        face_idx += 4
        
        # Add obstacle meshes (simplified boxes)
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
            
            # Box faces (12 triangles per box)
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
            triangles.extend(box_faces)
            face_idx += 8
        
        return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)
    
    def _create_lidar_ray_pattern(self) -> np.ndarray:
        """Create LiDAR ray pattern based on configuration"""
        n_scan_lines = self.lidar_config['n_scan_lines']
        n_points_per_line = self.lidar_config['n_points_per_line']
        fov_v = np.radians(self.lidar_config['fov_vertical'])
        fov_h = np.radians(self.lidar_config['fov_horizontal'])
        
        # Create angular grids
        vertical_angles = np.linspace(-fov_v/2, fov_v/2, n_scan_lines)
        horizontal_angles = np.linspace(-fov_h/2, fov_h/2, n_points_per_line)
        
        # Generate ray vectors in spherical coordinates
        ray_vectors = np.zeros((n_scan_lines, n_points_per_line, 3), dtype=np.float32)
        
        for i, v_angle in enumerate(vertical_angles):
            for j, h_angle in enumerate(horizontal_angles):
                # Convert spherical to cartesian
                # x = forward, y = left, z = up
                x = np.cos(v_angle) * np.cos(h_angle)
                y = np.cos(v_angle) * np.sin(h_angle)
                z = np.sin(v_angle)
                
                ray_vectors[i, j] = [x, y, z]
        
        return ray_vectors
    
    def step(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Step simulation and update lidar"""
        # Update robot state
        self._update_robot_state()
        
        # Update lidar sensor
        point_cloud, distances = None, None
        if self.lidar_sensor is not None:
            try:
                # Calculate sensor pose with proper offset
                sensor_quat = quat_mul_genesis(
                    self.base_quat, 
                    self.sensor_offset_quat.unsqueeze(0).expand(self.num_envs, -1)
                )
                sensor_pos = self.base_pos + quat_apply_genesis(
                    self.base_quat, 
                    self.sensor_translation.unsqueeze(0).expand(self.num_envs, -1)
                )
                
                # Convert to Taichi format (xyzw)
                sensor_quat_taichi = convert_genesis_quat_to_taichi(sensor_quat)
                
                # Perform ray casting
                hit_points, hit_distances = self.lidar_sensor.cast_rays(
                    lidar_positions=sensor_pos.cpu().numpy().reshape(1, 1, -1),  # Shape: [1, 1, 3]
                    lidar_quaternions=sensor_quat_taichi.cpu().numpy().reshape(1, 1, -1),  # Shape: [1, 1, 4]
                    ray_vectors=self.ray_vectors,
                    far_plane=self.lidar_config['max_range'],
                    pointcloud_in_world_frame=True
                )
                
                # Convert to torch tensors
                if hit_points is not None:
                    point_cloud = torch.from_numpy(hit_points).to(self.device)
                    distances = torch.from_numpy(hit_distances).to(self.device)
                    
                    # Store for visualization
                    self.current_points = point_cloud.clone()
                    self.current_distances = distances.clone()
                    self.lidar_update_counter += 1
                
            except Exception as e:
                print(f"Taichi lidar update failed: {e}")
                import traceback
                traceback.print_exc()
        
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
            if len(self.current_points.shape) == 4:  # [env, scan_line, point, 3]
                points = self.current_points[0].view(-1, 3)  # Flatten scan lines and points
                distances = self.current_distances[0].view(-1)
            else:
                points = self.current_points[0] if len(self.current_points.shape) > 2 else self.current_points
                distances = self.current_distances[0] if len(self.current_distances.shape) > 1 else self.current_distances
            
            # Filter points by distance (remove invalid/far points)
            valid_mask = (distances > self.lidar_config['min_range']) & (distances < self.lidar_config['max_range'])
            if valid_mask.sum() == 0:
                return
            
            world_points = points[valid_mask]
            valid_distances = distances[valid_mask]
            
            # Sample points for visualization (too many points can slow down rendering)
            max_points = 5000
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
                # Calculate sensor position for line drawing
                sensor_quat = quat_mul_genesis(self.base_quat[0:1], self.sensor_offset_quat.unsqueeze(0))
                sensor_pos = self.base_pos[0:1] + quat_apply_genesis(self.base_quat[0:1], self.sensor_translation.unsqueeze(0))
                self._draw_point_lines(world_points, sensor_pos[0])
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_distance_colors(self, distances: torch.Tensor) -> List[Tuple[float, float, float, float]]:
        """Generate colors based on distance values"""
        # Normalize distances to 0-1
        min_dist, max_dist = self.lidar_config['min_range'], self.lidar_config['max_range'] * 0.8
        normalized = torch.clamp((distances - min_dist) / (max_dist - min_dist), 0, 1)
        
        colors = []
        for dist in normalized:
            # Color from red (close) to blue (far)
            r = 1.0 - dist.item()
            g = 0.3
            b = dist.item()
            a = 0.8
            colors.append((r, g, b, a))
        
        return colors
    
    def _draw_point_spheres(self, points: torch.Tensor, colors: List[Tuple[float, float, float, float]]):
        """Draw lidar points as colored spheres"""
        self.scene.clear_debug_objects()
        
        if len(points) > 0:
            # Use the first color for all points (Genesis limitation)
            self.scene.draw_debug_spheres(
                poss=points,
                radius=0.02,
                color=colors[0] if colors else (1.0, 0.0, 0.0, 0.8)
            )
    
    def _draw_point_lines(self, points: torch.Tensor, sensor_pos: torch.Tensor):
        """Draw lidar points as lines from sensor"""
        self.scene.clear_debug_objects()
        
        # Draw lines from sensor to hit points (sample a subset)
        max_lines = 100
        if len(points) > max_lines:
            indices = torch.randperm(len(points))[:max_lines]
            points = points[indices]
        
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
        
        self.scene = None
        print("Environment closed")


def main():
    """Main demonstration"""
    print("=== Genesis G1 Taichi Lidar Visualization Demo ===")
    
    # Test configurations
    configs = [
        {
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'visualization_mode': 'spheres',
            'lidar_config': {
                'n_scan_lines': 16,
                'n_points_per_line': 32,
                'fov_vertical': 30.0,
                'fov_horizontal': 120.0,
                'max_range': 15.0,
                'min_range': 0.1
            }
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Demo {i+1}: Taichi LiDAR ---")
        
        try:
            # Create environment
            env = GenesisG1TaichiLidarVisualizer(
                num_envs=1,
                device=config['device'],
                headless=False,
                visualization_mode=config['visualization_mode'],
                lidar_config=config['lidar_config']
            )
            
            # Reset and run
            env.reset()
            
            print("Running Taichi LiDAR visualization demo...")
            print("Watch the colored spheres representing lidar points!")
            print("Red = close, Blue = far")
            
            # Simulation loop
            for step in range(500):
                start_time = time.time()
                point_cloud, distances = env.step()
                step_time = time.time() - start_time
                
                # Move robot in circle for interesting visualization
                if step % 50 == 0 and env.robot is not None:
                    try:
                        angle = step * 0.05
                        angle_tensor = torch.tensor(angle, device=env.device)
                        
                        new_pos = env.base_pos.clone()
                        new_pos[:, 0] = 1.0 * torch.cos(angle_tensor)
                        new_pos[:, 1] = 1.0 * torch.sin(angle_tensor)
                        env.robot.set_pos(new_pos)
                        
                        # Also rotate robot
                        yaw_tensor = torch.tensor(angle/2, device=env.device)
                        new_quat = env.base_quat.clone()
                        new_quat[:, 0] = torch.cos(yaw_tensor/2)  # w
                        new_quat[:, 3] = torch.sin(yaw_tensor/2)  # z
                        env.robot.set_quat(new_quat)
                        
                    except Exception as e:
                        print(f"Robot movement failed: {e}")
                
                # Print status
                if point_cloud is not None and step % 50 == 0:
                    if len(point_cloud.shape) == 4:  # [env, scan_line, point, 3]
                        total_points = point_cloud.shape[1] * point_cloud.shape[2]
                    else:
                        total_points = point_cloud.shape[0] if len(point_cloud.shape) > 1 else len(point_cloud)
                    
                    dist_min = distances.min().item() if distances is not None else 0
                    dist_max = distances.max().item() if distances is not None else 0
                    dist_range = f"[{dist_min:.2f}, {dist_max:.2f}]"
                    
                    print(f"  Step {step}: {total_points} points, range: {dist_range}, step time: {step_time*1000:.1f}ms")
            
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
    
    print("\n=== Taichi LiDAR visualization demo completed ===")


if __name__ == "__main__":
    main()
