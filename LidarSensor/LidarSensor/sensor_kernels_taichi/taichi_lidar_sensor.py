import taichi as ti
import numpy as np
import torch
import math
from abc import ABC, abstractmethod

class BaseSensor(ABC):
    def __init__(self, env,env_cfg,sensor_config, num_sensor, device):
        self.env =env
        self.env_cfg =env_cfg
        self.sensor_cfg = sensor_config
        self.device = device
        self.num_sensor = num_sensor
        self.robot_position = None
        self.robot_orientation = None
        self.robot_linvel = None
        self.robot_angvel = None

    @abstractmethod
    def init_tensors(self):
        raise NotImplementedError("update func not implemented")

    @abstractmethod
    def update(self):
        raise NotImplementedError("update func not implemented")
# Initialize Taichi with GPU
ti.init(arch=ti.gpu)

@ti.data_oriented
class AABB:
    """Axis-Aligned Bounding Box for BVH construction"""
    def __init__(self):
        self.min_bound = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.max_bound = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.min_bound[None] = ti.Vector([float('inf'), float('inf'), float('inf')])
        self.max_bound[None] = ti.Vector([float('-inf'), float('-inf'), float('-inf')])
    
    @ti.func
    def expand(self, point):
        """Expand the AABB to include the point"""
        ti.atomic_min(self.min_bound[None][0], point[0])
        ti.atomic_min(self.min_bound[None][1], point[1])
        ti.atomic_min(self.min_bound[None][2], point[2])
        
        ti.atomic_max(self.max_bound[None][0], point[0])
        ti.atomic_max(self.max_bound[None][1], point[1])
        ti.atomic_max(self.max_bound[None][2], point[2])

    @ti.func
    def intersect(self, ray_origin, ray_dir, t_min=0.001, t_max=float('inf')):
        """Test if ray intersects this AABB"""
        inv_dir = ti.Vector([1.0 / ray_dir[0], 1.0 / ray_dir[1], 1.0 / ray_dir[2]])
        
        t1 = (self.min_bound[None][0] - ray_origin[0]) * inv_dir[0]
        t2 = (self.max_bound[None][0] - ray_origin[0]) * inv_dir[0]
        
        t3 = (self.min_bound[None][1] - ray_origin[1]) * inv_dir[1]
        t4 = (self.max_bound[None][1] - ray_origin[1]) * inv_dir[1]
        
        t5 = (self.min_bound[None][2] - ray_origin[2]) * inv_dir[2]
        t6 = (self.max_bound[None][2] - ray_origin[2]) * inv_dir[2]
        
        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
        
        # If tmax < 0, ray is intersecting AABB, but the whole AABB is behind us
        if tmax < 0:
            return False, t_max
        
        # If tmin > tmax, ray doesn't intersect AABB
        if tmin > tmax:
            return False, t_max
        
        # If tmin < 0, ray origin is inside the AABB
        if tmin < 0:
            return True, tmax
        
        return True, tmin

@ti.data_oriented
class BVHNode:
    """BVH node for accelerating ray-triangle intersection tests"""
    def __init__(self, max_triangles=1000):
        # BVH tree structure
        self.max_nodes = 2 * max_triangles  # Worst case: one leaf per triangle
        
        # Node properties
        self.aabb_min = ti.Vector.field(3, dtype=ti.f32, shape=self.max_nodes)
        self.aabb_max = ti.Vector.field(3, dtype=ti.f32, shape=self.max_nodes)
        self.left_child = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self.right_child = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self.triangle_count = ti.field(dtype=ti.i32, shape=self.max_nodes)
        self.first_triangle_idx = ti.field(dtype=ti.i32, shape=self.max_nodes)
        
        # Initialize values
        for i in range(self.max_nodes):
            self.left_child[i] = -1
            self.right_child[i] = -1
            self.triangle_count[i] = 0
            self.first_triangle_idx[i] = -1
            
            # Initialize AABB to "empty"
            self.aabb_min[i] = ti.Vector([float('inf'), float('inf'), float('inf')])
            self.aabb_max[i] = ti.Vector([float('-inf'), float('-inf'), float('-inf')])

@ti.data_oriented
class TaichiLidarSensor(BaseSensor):
    def __init__(self, env, env_cfg, sensor_config, num_sensor=1, device='cuda:0'):
        super().__init__(env, env_cfg, sensor_config, num_sensor, device)
        
        # Extract configuration from sensor_config
        self.num_envs = self.env['num_envs']
        self.num_vertical_lines = self.sensor_cfg.vertical_line_num
        self.num_horizontal_lines = self.sensor_cfg.horizontal_line_num
        self.pointcloud_in_world_frame = self.sensor_cfg.pointcloud_in_world_frame
        
        # Convert FOV from degrees to radians
        self.horizontal_fov_min = math.radians(self.sensor_cfg.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.sensor_cfg.horizontal_fov_deg_max)
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2
        
        self.vertical_fov_min = math.radians(self.sensor_cfg.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.sensor_cfg.vertical_fov_deg_max)
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2
        
        self.far_plane = self.sensor_cfg.max_range
        
        # Check if we have valid sensor position and orientation
        assert self.env['sensor_pos_tensor'] is not None
        assert self.env['sensor_quat_tensor'] is not None
        self.lidar_positions_tensor = self.env['sensor_pos_tensor']
        self.lidar_quat_tensor = self.env['sensor_quat_tensor']
        
        # Initialize Taichi fields
        self.init_taichi_fields()
        self.initialize_ray_vectors()
        self.init_tensors()
        
        # BVH acceleration structure
        self.bvh = None
        self.triangle_indices_sorted = None
        
    def init_taichi_fields(self):
        """Initialize Taichi fields for efficient computation"""
        # Fields for mesh data
        self.mesh_vertices = None  # Will be set when mesh data is available
        self.mesh_indices = None   # Will be set when mesh data is available
        
        # Fields for ray data
        self.ray_origins = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_envs, self.num_sensor))
        self.ray_directions = ti.Vector.field(3, dtype=ti.f32, 
                                            shape=(self.num_vertical_lines, self.num_horizontal_lines))
        
        # Fields for results
        self.hit_distances = ti.field(dtype=ti.f32, 
                                    shape=(self.num_envs, self.num_sensor, 
                                           self.num_vertical_lines, self.num_horizontal_lines))
        self.hit_points = ti.Vector.field(3, dtype=ti.f32, 
                                         shape=(self.num_envs, self.num_sensor, 
                                                self.num_vertical_lines, self.num_horizontal_lines))
        self.hit_normals = ti.Vector.field(3, dtype=ti.f32, 
                                          shape=(self.num_envs, self.num_sensor, 
                                                 self.num_vertical_lines, self.num_horizontal_lines))
        
        # Constants
        self.NO_HIT_VALUE = 1000.0
        
    def initialize_ray_vectors(self):
        """Initialize the ray direction vectors based on LiDAR configuration"""
        @ti.kernel
        def compute_ray_directions():
            for i, j in self.ray_directions:
                # Compute azimuth and elevation angles
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_horizontal_lines - 1))
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_vertical_lines - 1))
                
                # Convert to Cartesian coordinates (x, y, z)
                x = ti.cos(azimuth_angle) * ti.cos(elevation_angle)
                y = ti.sin(azimuth_angle) * ti.cos(elevation_angle)
                z = ti.sin(elevation_angle)
                
                # Normalize and store the ray direction
                dir = ti.Vector([x, y, z])
                self.ray_directions[i, j] = dir.normalized()
        
        # Execute the Taichi kernel to compute ray directions
        compute_ray_directions()

    def init_tensors(self):
        """Initialize tensors for storing results"""
        # Convert PyTorch tensors to Taichi fields
        lidar_pos_np = self.lidar_positions_tensor.cpu().numpy().reshape(self.num_envs, self.num_sensor, 3)
        lidar_quat_np = self.lidar_quat_tensor.cpu().numpy().reshape(self.num_envs, self.num_sensor, 4)
        
        # Create output tensors
        self.lidar_tensor = torch.zeros(
            (self.num_envs, self.num_sensor, self.num_vertical_lines, self.num_horizontal_lines, 3),
            device=self.device,
            requires_grad=False
        )
        
        self.lidar_dist_tensor = torch.zeros(
            (self.num_envs, self.num_sensor, self.num_vertical_lines, self.num_horizontal_lines),
            device=self.device,
            requires_grad=False
        )
        
        # Set up the initial values for ray origins (LiDAR positions)
        @ti.kernel
        def set_ray_origins(lidar_pos: ti.types.ndarray()):
            for i, j in ti.ndrange(self.num_envs, self.num_sensor):
                self.ray_origins[i, j] = ti.Vector([lidar_pos[i, j, 0], lidar_pos[i, j, 1], lidar_pos[i, j, 2]])
        
        # Execute the kernel to set ray origins
        set_ray_origins(lidar_pos_np)
    
    def set_mesh_data(self, vertices, indices):
        """Set mesh data for ray-triangle intersection and build BVH"""
        num_vertices = vertices.shape[0]
        num_triangles = indices.shape[0]
        
        # Create Taichi fields for mesh data
        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.mesh_indices = ti.Vector.field(3, dtype=ti.i32, shape=num_triangles)
        
        # Create field for triangle centroids (used for BVH sorting)
        triangle_centroids = ti.Vector.field(3, dtype=ti.f32, shape=num_triangles)
        
        # Convert numpy arrays to Taichi fields
        @ti.kernel
        def set_mesh_data_kernel(v: ti.types.ndarray(), idx: ti.types.ndarray()):
            for i in range(num_vertices):
                self.mesh_vertices[i] = ti.Vector([v[i, 0], v[i, 1], v[i, 2]])
            
            # Set triangle indices and compute centroids
            for i in range(num_triangles):
                i0, i1, i2 = idx[i, 0], idx[i, 1], idx[i, 2]
                self.mesh_indices[i] = ti.Vector([i0, i1, i2])
                
                # Compute triangle centroid
                v0 = self.mesh_vertices[i0]
                v1 = self.mesh_vertices[i1]
                v2 = self.mesh_vertices[i2]
                triangle_centroids[i] = (v0 + v1 + v2) / 3.0
        
        # Execute the kernel to set mesh data
        set_mesh_data_kernel(vertices, indices)
        
        # Build BVH for acceleration
        self.build_bvh(num_triangles, triangle_centroids)
    
    def build_bvh(self, num_triangles, triangle_centroids):
        """Build a Bounding Volume Hierarchy for fast ray-triangle intersection"""
        # Create BVH structure
        self.bvh = BVHNode(num_triangles)
        
        # Create a sorted array of triangle indices
        self.triangle_indices_sorted = ti.field(dtype=ti.i32, shape=num_triangles)
        
        # Initialize triangle indices
        @ti.kernel
        def init_triangle_indices():
            for i in range(num_triangles):
                self.triangle_indices_sorted[i] = i
        
        init_triangle_indices()
        
        # Compute global AABB for all triangles
        global_aabb = AABB()
        
        @ti.kernel
        def compute_global_aabb():
            for i in range(num_triangles):
                i0, i1, i2 = self.mesh_indices[i][0], self.mesh_indices[i][1], self.mesh_indices[i][2]
                v0, v1, v2 = self.mesh_vertices[i0], self.mesh_vertices[i1], self.mesh_vertices[i2]
                
                global_aabb.expand(v0)
                global_aabb.expand(v1)
                global_aabb.expand(v2)
        
        compute_global_aabb()
        
        # Initialize root node
        self.bvh.aabb_min[0] = global_aabb.min_bound[None]
        self.bvh.aabb_max[0] = global_aabb.max_bound[None]
        self.bvh.triangle_count[0] = num_triangles
        self.bvh.first_triangle_idx[0] = 0
        
        # Recursively build BVH (simplified version using median splits)
        def build_bvh_recursive(node_idx, start_idx, end_idx, depth=0):
            if end_idx - start_idx <= 8 or depth >= 20:  # Leaf node
                return
            
            # Find the axis with the largest extent
            node_min = self.bvh.aabb_min[node_idx]
            node_max = self.bvh.aabb_max[node_idx]
            extent = node_max - node_min
            axis = 0
            if extent[1] > extent[0] and extent[1] > extent[2]:
                axis = 1
            elif extent[2] > extent[0] and extent[2] > extent[1]:
                axis = 2
            
            # Sort triangles based on centroid along the selected axis
            # Note: For simplicity, we'll sort on CPU
            centroids_np = triangle_centroids.to_numpy()
            indices_np = self.triangle_indices_sorted.to_numpy()
            
            # Get centroids for triangles in this node
            node_centroids = centroids_np[indices_np[start_idx:end_idx]]
            
            # Sort the indices based on centroid coordinate along the selected axis
            sorted_indices = np.argsort(node_centroids[:, axis])
            indices_np[start_idx:end_idx] = indices_np[start_idx:end_idx][sorted_indices]
            
            # Update Taichi field with sorted indices
            self.triangle_indices_sorted.from_numpy(indices_np)
            
            # Find the middle index for splitting
            mid_idx = start_idx + (end_idx - start_idx) // 2
            
            # Create child nodes
            left_idx = node_idx * 2 + 1
            right_idx = node_idx * 2 + 2
            
            # Initialize child nodes
            self.bvh.left_child[node_idx] = left_idx
            self.bvh.right_child[node_idx] = right_idx
            
            self.bvh.triangle_count[left_idx] = mid_idx - start_idx
            self.bvh.first_triangle_idx[left_idx] = start_idx
            
            self.bvh.triangle_count[right_idx] = end_idx - mid_idx
            self.bvh.first_triangle_idx[right_idx] = mid_idx
            
            # Compute AABBs for child nodes
            @ti.kernel
            def compute_node_aabb(node_idx: ti.i32, start_idx: ti.i32, end_idx: ti.i32):
                node_min = ti.Vector([float('inf'), float('inf'), float('inf')])
                node_max = ti.Vector([float('-inf'), float('-inf'), float('-inf')])
                
                for i in range(start_idx, end_idx):
                    tri_idx = self.triangle_indices_sorted[i]
                    i0, i1, i2 = self.mesh_indices[tri_idx][0], self.mesh_indices[tri_idx][1], self.mesh_indices[tri_idx][2]
                    v0, v1, v2 = self.mesh_vertices[i0], self.mesh_vertices[i1], self.mesh_vertices[i2]
                    
                    # Expand AABB
                    for v in (v0, v1, v2):
                        for j in ti.static(range(3)):
                            ti.atomic_min(node_min[j], v[j])
                            ti.atomic_max(node_max[j], v[j])
                
                self.bvh.aabb_min[node_idx] = node_min
                self.bvh.aabb_max[node_idx] = node_max
            
            # Compute AABBs for both children
            compute_node_aabb(left_idx, start_idx, mid_idx)
            compute_node_aabb(right_idx, mid_idx, end_idx)
            
            # Recursively build sub-trees
            build_bvh_recursive(left_idx, start_idx, mid_idx, depth + 1)
            build_bvh_recursive(right_idx, mid_idx, end_idx, depth + 1)
        
        # Start building the BVH
        build_bvh_recursive(0, 0, num_triangles)
        
        print(f"BVH built with {num_triangles} triangles")
    
    @ti.func
    def ray_triangle_intersection(self, ray_origin, ray_dir, triangle_idx):
        """
        Möller-Trumbore algorithm for ray-triangle intersection
        Returns hit distance, hit point, normal, and whether there was a hit
        """
        # Get triangle vertices
        v0 = self.mesh_vertices[self.mesh_indices[triangle_idx][0]]
        v1 = self.mesh_vertices[self.mesh_indices[triangle_idx][1]]
        v2 = self.mesh_vertices[self.mesh_indices[triangle_idx][2]]
        
        # Compute triangle edges
        e1 = v1 - v0
        e2 = v2 - v0
        
        # Calculate determinant
        h = ray_dir.cross(e2)
        a = e1.dot(h)
        
        # Check if ray is parallel to the triangle
        EPSILON = 1e-8
        if ti.abs(a) < EPSILON:
            return False, 0.0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        
        f = 1.0 / a
        s = ray_origin - v0
        u = f * s.dot(h)
        
        # Check if hit point is outside the triangle
        if u < 0.0 or u > 1.0:
            return False, 0.0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        
        q = s.cross(e1)
        v = f * ray_dir.dot(q)
        
        # Check if hit point is outside the triangle
        if v < 0.0 or u + v > 1.0:
            return False, 0.0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        
        # Compute the distance to the intersection point
        t = f * e2.dot(q)
        
        # Check if intersection is behind the ray origin
        if t <= EPSILON:
            return False, 0.0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        
        # Compute hit point
        hit_point = ray_origin + t * ray_dir
        
        # Compute normal (normalized cross product of the triangle edges)
        normal = e1.cross(e2).normalized()
        
        return True, t, hit_point, normal
    
    @ti.func
    def intersect_bvh(self, ray_origin, ray_dir, t_min, t_max):
        """
        Traverse BVH to find the closest triangle intersection
        Returns hit information for the closest triangle hit
        """
        hit = False
        closest_t = t_max
        closest_hit_point = ti.Vector([0.0, 0.0, 0.0])
        closest_normal = ti.Vector([0.0, 0.0, 0.0])
        
        # Stack for traversal (avoids recursion)
        stack = ti.field(dtype=ti.i32, shape=64)  # Max depth
        stack_ptr = 0
        
        # Start with the root node
        stack[0] = 0  # Root node index
        stack_ptr = 1
        
        while stack_ptr > 0:
            # Pop a node
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            
            # Check if ray intersects node AABB
            node_min = ti.Vector([self.bvh.aabb_min[node_idx][0], 
                                 self.bvh.aabb_min[node_idx][1], 
                                 self.bvh.aabb_min[node_idx][2]])
            node_max = ti.Vector([self.bvh.aabb_max[node_idx][0], 
                                 self.bvh.aabb_max[node_idx][1], 
                                 self.bvh.aabb_max[node_idx][2]])
            
            # Ray-AABB intersection test
            inv_dir = ti.Vector([1.0 / ray_dir[0], 1.0 / ray_dir[1], 1.0 / ray_dir[2]])
            
            t1 = (node_min[0] - ray_origin[0]) * inv_dir[0]
            t2 = (node_max[0] - ray_origin[0]) * inv_dir[0]
            
            t3 = (node_min[1] - ray_origin[1]) * inv_dir[1]
            t4 = (node_max[1] - ray_origin[1]) * inv_dir[1]
            
            t5 = (node_min[2] - ray_origin[2]) * inv_dir[2]
            t6 = (node_max[2] - ray_origin[2]) * inv_dir[2]
            
            tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
            tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
            
            # If there's no intersection with the node, skip it
            if tmax < 0 or tmin > tmax or tmin > closest_t:
                continue
            
            # If it's a leaf node, test triangles
            left_child = self.bvh.left_child[node_idx]
            right_child = self.bvh.right_child[node_idx]
            
            if left_child < 0:  # Leaf node
                start_idx = self.bvh.first_triangle_idx[node_idx]
                end_idx = start_idx + self.bvh.triangle_count[node_idx]
                
                # Test all triangles in the leaf
                for i in range(start_idx, end_idx):
                    tri_idx = self.triangle_indices_sorted[i]
                    tri_hit, tri_t, tri_hit_point, tri_normal = self.ray_triangle_intersection(
                        ray_origin, ray_dir, tri_idx)
                    
                    if tri_hit and tri_t < closest_t and tri_t > t_min:
                        hit = True
                        closest_t = tri_t
                        closest_hit_point = tri_hit_point
                        closest_normal = tri_normal
            else:
                # Add children to the stack
                # Add the closest child first (front-to-back traversal)
                if ray_dir[0] > 0.0 or (ray_dir[0] == 0.0 and ray_origin[0] > (node_min[0] + node_max[0]) / 2):
                    # Push right then left (so left gets processed first)
                    stack[stack_ptr] = right_child
                    stack_ptr += 1
                    stack[stack_ptr] = left_child
                    stack_ptr += 1
                else:
                    # Push left then right
                    stack[stack_ptr] = left_child
                    stack_ptr += 1
                    stack[stack_ptr] = right_child
                    stack_ptr += 1
        
        return hit, closest_t, closest_hit_point, closest_normal
    
    @ti.kernel
    def trace_rays(self):
        """
        Main ray tracing kernel - highly optimized for parallel execution
        Traces rays against the triangle mesh using BVH acceleration
        """
        # Initialize all hits to no-hit values
        for env_id, sensor_id, i, j in ti.ndrange(self.num_envs, self.num_sensor, 
                                              self.num_vertical_lines, self.num_horizontal_lines):
            self.hit_distances[env_id, sensor_id, i, j] = self.NO_HIT_VALUE
            self.hit_points[env_id, sensor_id, i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.hit_normals[env_id, sensor_id, i, j] = ti.Vector([0.0, 0.0, 0.0])
        
        # For each ray
        for env_id, sensor_id, i, j in ti.ndrange(self.num_envs, self.num_sensor, 
                                              self.num_vertical_lines, self.num_horizontal_lines):
            ray_origin = self.ray_origins[env_id, sensor_id]
            ray_dir = self.ray_directions[i, j]
            
            # Intersect ray with BVH
            hit, dist, hit_point, normal = self.intersect_bvh(ray_origin, ray_dir, 0.001, self.far_plane)
            
            # Store the hit information
            if hit:
                self.hit_distances[env_id, sensor_id, i, j] = dist
                self.hit_points[env_id, sensor_id, i, j] = hit_point
                self.hit_normals[env_id, sensor_id, i, j] = normal
    
    @ti.kernel
    def transform_to_world_frame(self):
        """
        Transform hit points to world frame if needed
        This is separate from the ray tracing to allow for better memory access patterns
        """
        if self.pointcloud_in_world_frame:
            for env_id, sensor_id, i, j in ti.ndrange(self.num_envs, self.num_sensor, 
                                                  self.num_vertical_lines, self.num_horizontal_lines):
                if self.hit_distances[env_id, sensor_id, i, j] < self.NO_HIT_VALUE:
                    # For world frame, we keep the actual hit point
                    pass
                else:
                    # For no-hit rays, set point to origin + far_plane * direction
                    self.hit_points[env_id, sensor_id, i, j] = (
                        self.ray_origins[env_id, sensor_id] + 
                        self.far_plane * self.ray_directions[i, j]
                    )
        else:
            # For sensor frame, convert hit points to local coordinates
            for env_id, sensor_id, i, j in ti.ndrange(self.num_envs, self.num_sensor, 
                                                  self.num_vertical_lines, self.num_horizontal_lines):
                if self.hit_distances[env_id, sensor_id, i, j] < self.NO_HIT_VALUE:
                    # For sensor frame, we use distance * direction
                    self.hit_points[env_id, sensor_id, i, j] = (
                        self.hit_distances[env_id, sensor_id, i, j] * self.ray_directions[i, j]
                    )
    
    def update_lidar_positions(self):
        """Update LiDAR positions from the environment tensors"""
        lidar_pos_np = self.lidar_positions_tensor.cpu().numpy().reshape(self.num_envs, self.num_sensor, 3)
        
        @ti.kernel
        def update_ray_origins(lidar_pos: ti.types.ndarray()):
            for i, j in ti.ndrange(self.num_envs, self.num_sensor):
                self.ray_origins[i, j] = ti.Vector([lidar_pos[i, j, 0], lidar_pos[i, j, 1], lidar_pos[i, j, 2]])
        
        update_ray_origins(lidar_pos_np)
    
    def update(self):
        """
        Main update method called by the environment
        Traces rays and returns hit information
        """
        # Update LiDAR positions
        self.update_lidar_positions()
        
        # Trace rays against the mesh
        self.trace_rays()
        
        # Transform hit points to world or sensor frame
        self.transform_to_world_frame()
        
        # Copy results back to PyTorch tensors
        hit_points_np = self.hit_points.to_numpy()
        hit_distances_np = self.hit_distances.to_numpy()
        
        # Convert to PyTorch tensors on the correct device
        self.lidar_tensor = torch.tensor(hit_points_np, device=self.device)
        self.lidar_dist_tensor = torch.tensor(hit_distances_np, device=self.device)
        
        return self.lidar_tensor, self.lidar_dist_tensor 