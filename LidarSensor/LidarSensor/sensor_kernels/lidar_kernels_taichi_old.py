import taichi as ti
import numpy as np
import torch

# Initialize Taichi with GPU support
ti.init(arch=ti.gpu)

NO_HIT_RAY_VAL = 1000.0
NO_HIT_SEGMENTATION_VAL = -2

@ti.data_oriented
class LidarTaichiKernels:
    def __init__(self, max_num_vertices=1000000, max_num_triangles=100000, max_nodes=2000000):
        # Mesh data structures
        self.points = ti.Vector.field(3, dtype=ti.f32, shape=max_num_vertices)
        self.indices = ti.field(ti.i32, shape=max_num_triangles * 3)
        self.num_vertices = ti.field(ti.i32, shape=())
        self.num_triangles = ti.field(ti.i32, shape=())
        
        # BVH data structures
        self.bvh_nodes_min = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.bvh_nodes_max = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.bvh_node_left = ti.field(ti.i32, shape=max_nodes)
        self.bvh_node_right = ti.field(ti.i32, shape=max_nodes)
        self.bvh_node_parent = ti.field(ti.i32, shape=max_nodes)
        self.bvh_node_is_leaf = ti.field(ti.i32, shape=max_nodes)
        self.bvh_node_triangle_idx = ti.field(ti.i32, shape=max_nodes)
        self.bvh_root = ti.field(ti.i32, shape=())
        self.bvh_num_nodes = ti.field(ti.i32, shape=())
        
        # Temporary fields for BVH construction
        self.centroids = ti.Vector.field(3, dtype=ti.f32, shape=max_num_triangles)
        self.tri_aabbs_min = ti.Vector.field(3, dtype=ti.f32, shape=max_num_triangles)
        self.tri_aabbs_max = ti.Vector.field(3, dtype=ti.f32, shape=max_num_triangles)
    
    def load_mesh_from_numpy(self, vertices, triangles):
        """Load mesh data from numpy arrays and build BVH"""
        num_vertices = vertices.shape[0]
        num_triangles = triangles.shape[0]
        
        # Set counts
        self.num_vertices[None] = num_vertices
        self.num_triangles[None] = num_triangles
        
        # Copy data to Taichi fields
        for i in range(num_vertices):
            self.points[i] = vertices[i]
        
        for i in range(num_triangles):
            self.indices[i*3] = triangles[i][0]
            self.indices[i*3+1] = triangles[i][1]
            self.indices[i*3+2] = triangles[i][2]
        
        # Compute triangle data for BVH
        self._compute_triangle_data()
        
        # Build BVH
        self._build_bvh()
    
    @ti.kernel
    def _compute_triangle_data(self):
        """Compute triangle centroids and AABBs"""
        for i in range(self.num_triangles[None]):
            i0, i1, i2 = self.indices[i*3], self.indices[i*3+1], self.indices[i*3+2]
            p0, p1, p2 = self.points[i0], self.points[i1], self.points[i2]
            
            # Compute centroid
            self.centroids[i] = (p0 + p1 + p2) / 3.0
            
            # Compute AABB
            self.tri_aabbs_min[i] = ti.min(ti.min(p0, p1), p2)
            self.tri_aabbs_max[i] = ti.max(ti.max(p0, p1), p2)
            
            # Add epsilon to avoid numerical issues
            eps = 1e-4
            self.tri_aabbs_min[i] -= ti.Vector([eps, eps, eps])
            self.tri_aabbs_max[i] += ti.Vector([eps, eps, eps])
    
    @ti.kernel
    def _build_bvh(self):
        """Build a simple BVH for the mesh (simplified implementation)"""
        # Create root node (contains all triangles)
        self.bvh_root[None] = 0
        self.bvh_num_nodes[None] = 1
        
        # Set leaf node with all triangles for now
        node_idx = 0
        self.bvh_node_is_leaf[node_idx] = 1
        self.bvh_node_triangle_idx[node_idx] = 0  # Start triangle index
        
        # Compute scene AABB
        scene_min = ti.Vector([1e10, 1e10, 1e10])  # Using large values instead of float('inf')
        scene_max = ti.Vector([-1e10, -1e10, -1e10])
        
        for i in range(self.num_triangles[None]):
            scene_min = ti.min(scene_min, self.tri_aabbs_min[i])
            scene_max = ti.max(scene_max, self.tri_aabbs_max[i])
        
        self.bvh_nodes_min[node_idx] = scene_min
        self.bvh_nodes_max[node_idx] = scene_max
    
    @ti.func
    def ray_aabb_intersection(self, ray_origin, ray_dir, aabb_min, aabb_max):
        """Check if a ray intersects an AABB using slab method"""
        rcp_dir = ti.Vector([1.0, 1.0, 1.0])
        # Avoid division by zero
        for i in ti.static(range(3)):
            if ray_dir[i] != 0:
                rcp_dir[i] = 1.0 / ray_dir[i]
        
        t1 = (aabb_min[0] - ray_origin[0]) * rcp_dir[0]
        t2 = (aabb_max[0] - ray_origin[0]) * rcp_dir[0]
        tmin = ti.min(t1, t2)
        tmax = ti.max(t1, t2)
        
        t1 = (aabb_min[1] - ray_origin[1]) * rcp_dir[1]
        t2 = (aabb_max[1] - ray_origin[1]) * rcp_dir[1]
        tmin = ti.max(tmin, ti.min(t1, t2))
        tmax = ti.min(tmax, ti.max(t1, t2))
        
        t1 = (aabb_min[2] - ray_origin[2]) * rcp_dir[2]
        t2 = (aabb_max[2] - ray_origin[2]) * rcp_dir[2]
        tmin = ti.max(tmin, ti.min(t1, t2))
        tmax = ti.min(tmax, ti.max(t1, t2))
        
        hit = (tmax >= tmin) and (tmax >= 0.0)
        return hit, tmin
    
    @ti.func
    def ray_triangle_intersection_moller(self, ray_origin, ray_dir, v0, v1, v2):
        """Improved Möller–Trumbore algorithm with better numerical stability"""
        hit = False
        t = 0.0
        u = 0.0
        v = 0.0
        sign = 0.0
        normal = ti.Vector([0.0, 0.0, 0.0])
        
        # Define a smaller epsilon for better precision
        EPSILON = 1e-8
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = ray_dir.cross(edge2)
        a = edge1.dot(h)
        
        # Check if ray is not parallel to triangle (with smaller epsilon)
        if ti.abs(a) >= EPSILON:
            f = 1.0 / a
            s = ray_origin - v0
            u = f * s.dot(h)
            
            # Check if u is within bounds with tolerance
            if u >= -EPSILON and u <= 1.0 + EPSILON:
                q = s.cross(edge1)
                v = f * ray_dir.dot(q)
                
                # Check if v is within bounds and u+v <= 1 with tolerance
                if v >= -EPSILON and u + v <= 1.0 + EPSILON:
                    # Compute t
                    t = f * edge2.dot(q)
                    
                    # Check if t is positive (intersection in ray direction)
                    if t >= EPSILON:
                        hit = True
                        normal = edge1.cross(edge2)
                        normal_length = normal.norm()
                        if normal_length > EPSILON:
                            normal = normal / normal_length
                        sign = 1.0 if a > 0 else -1.0
        
        return hit, t, u, v, sign, normal
    
    @ti.func
    def mesh_query_ray(self, ray_origin, ray_dir, max_t):
        """Enhanced ray query with debug counters"""
        # Initialize min values
        min_t = max_t
        min_face = -1
        min_u = 0.0
        min_v = 0.0
        min_sign = 1.0
        min_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_found = 0
        
        # Normalize ray direction to ensure consistent behavior
        ray_dir_normalized = ray_dir
        ray_dir_length = ray_dir.norm()
        if ray_dir_length > 1e-8:
            ray_dir_normalized = ray_dir / ray_dir_length
        
        # Brute force approach for now
        for face_idx in range(self.num_triangles[None]):
            i0 = self.indices[face_idx*3]
            i1 = self.indices[face_idx*3+1]
            i2 = self.indices[face_idx*3+2]
            
            v0 = self.points[i0]
            v1 = self.points[i1]
            v2 = self.points[i2]
            
            hit, t, u, v, sign, normal = self.ray_triangle_intersection_moller(
                ray_origin, ray_dir_normalized, v0, v1, v2)
            
            if hit and t < min_t and t >= 0.0:
                min_t = t
                min_face = face_idx
                min_u = u
                min_v = v
                min_sign = sign
                min_normal = normal
                hit_found = 1
        
        return hit_found, min_t, min_u, min_v, min_sign, min_normal, min_face
    @ti.kernel
    def draw_lidar_pointcloud(self, 
                            ray_origins: ti.template(),
                            ray_directions: ti.template(),
                            far_plane: ti.f32,
                            hit_points: ti.template(),
                            hit_distances: ti.template(),
                            pointcloud_in_world_frame: ti.i32):
        """Lidar ray tracing kernel"""
        # Use block_dim for better GPU performance
        ti.loop_config(block_dim=128)
        
        num_envs = hit_points.shape[0]
        num_cams = hit_points.shape[1]
        num_vertical = hit_points.shape[2]
        num_horizontal = hit_points.shape[3]
        
        for env_id, cam_id, scan_line, point_index in ti.ndrange(num_envs, num_cams, num_vertical, num_horizontal):
            ray_origin = ray_origins[env_id, cam_id]
            ray_dir = ray_directions[scan_line, point_index]
            
            # Query ray intersection with mesh (function call, not kernel call)
            hit_found, t, u, v, sign, normal, face = self.mesh_query_ray(ray_origin, ray_dir, far_plane)
            
            # Set hit distance
            dist = NO_HIT_RAY_VAL
            if hit_found == 1:
                dist = t
            
            hit_distances[env_id, cam_id, scan_line, point_index] = dist
            
            # Set hit point - handle the vector separately
            hit_pos = ti.Vector([0.0, 0.0, 0.0])
            if pointcloud_in_world_frame == 1:
                hit_pos = ray_origin + dist * ray_dir
            else:
                hit_pos = dist * ray_dir
                
            # Assign values to the 5D tensor correctly - component by component
            hit_points[env_id, cam_id, scan_line, point_index, 0] = hit_pos[0]
            hit_points[env_id, cam_id, scan_line, point_index, 1] = hit_pos[1]
            hit_points[env_id, cam_id, scan_line, point_index, 2] = hit_pos[2]
    
    def create_lidar_pointcloud(self, ray_origins, ray_directions, far_plane, hit_points, hit_distances, pointcloud_in_world_frame):
        """Launch the lidar ray tracing for all environments and cameras"""
        # Convert torch tensors to numpy for Taichi
        if isinstance(ray_origins, torch.Tensor):
            ray_origins_np = ray_origins.cpu().numpy()
        else:
            ray_origins_np = ray_origins
            
        if isinstance(ray_directions, torch.Tensor):
            ray_directions_np = ray_directions.cpu().numpy()
        else:
            ray_directions_np = ray_directions
        
        # Create Taichi fields with the right shapes
        ray_origins_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_origins_np.shape[:2])
        ray_directions_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_directions_np.shape[:2])
        
        # Create fields with correct dimensions
        hit_points_field = ti.field(ti.f32, shape=hit_points.shape)  # 5D tensor
        hit_distances_field = ti.field(ti.f32, shape=hit_distances.shape)
        
        # Copy data to Taichi fields
        ray_origins_field.from_numpy(ray_origins_np)
        ray_directions_field.from_numpy(ray_directions_np)
        
        # Launch the raytracing kernel (single kernel call for all rays)
        self.draw_lidar_pointcloud(
            ray_origins_field, 
            ray_directions_field,
            far_plane,
            hit_points_field, 
            hit_distances_field,
            1 if pointcloud_in_world_frame else 0
        )
        
        # Copy results back to torch tensors
        hit_points_np = hit_points_field.to_numpy()
        hit_distances_np = hit_distances_field.to_numpy()
        
        # Update the original torch tensors
        hit_points.copy_(torch.from_numpy(hit_points_np))
        hit_distances.copy_(torch.from_numpy(hit_distances_np))