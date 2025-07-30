import taichi as ti
import numpy as np
from typing import Optional

# Import and initialize genesis first
import sys
import genesis as gs

# Initialize Genesis (this will also initialize Taichi properly)
# Only initialize if not already initialized
if not hasattr(gs, '_initialized') or not gs._initialized:
    try:
        gs.init(backend=gs.gpu, precision="32", debug=False, seed=0)
    except Exception as e:
        print(f"Failed to initialize Genesis with GPU backend: {e}")
        print("Falling back to CPU backend...")
        gs.init(backend=gs.cpu, precision="64", debug=True, seed=0)

from genesis.engine.bvh import AABB, LBVH

NO_HIT_RAY_VAL = ti.field(dtype=ti.f32, shape=())
NO_HIT_SEGMENTATION_VAL = ti.field(dtype=ti.i32, shape=())

# Initialize constants
NO_HIT_RAY_VAL[None] = 1000.0
NO_HIT_SEGMENTATION_VAL[None] = -2

@ti.data_oriented
class MeshData:
    """
    Taichi mesh data structure that holds vertices, triangles, and BVH for ray intersection.
    """
    
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        Initialize mesh data.
        
        Args:
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        self.n_vertices = vertices.shape[0]
        self.n_triangles = triangles.shape[0]
        
        # Store mesh geometry
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=self.n_triangles)
        
        # Copy data to Taichi fields
        self.vertices.from_numpy(vertices.astype(np.float32))
        self.triangles.from_numpy(triangles.astype(np.int32))
        
        # BVH acceleration structure (will be built when needed)
        self._build_triangle_aabbs()
    
    def _build_triangle_aabbs(self):
        """Build AABBs for each triangle for BVH construction."""
        # Ensure we have at least one triangle
        if self.n_triangles == 0:
            raise ValueError("Cannot create BVH for mesh with 0 triangles")
            
        # Create AABB for each triangle
        self.triangle_aabbs = AABB(n_batches=1, n_aabbs=self.n_triangles)
        self._compute_triangle_aabbs()
    
    @ti.kernel
    def _compute_triangle_aabbs(self):
        """Compute AABB for each triangle."""
        for i in range(self.n_triangles):
            # Get triangle vertices
            v0_idx = self.triangles[i][0]
            v1_idx = self.triangles[i][1] 
            v2_idx = self.triangles[i][2]
            
            v0 = self.vertices[v0_idx]
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            
            # Compute AABB
            min_pos = ti.min(ti.min(v0, v1), v2)
            max_pos = ti.max(ti.max(v0, v1), v2)
            
            # Store in AABB structure
            self.triangle_aabbs.aabbs[0, i].min = min_pos
            self.triangle_aabbs.aabbs[0, i].max = max_pos


@ti.data_oriented
class LidarTaichiKernels:
    """
    Taichi-based LiDAR ray casting kernels with BVH acceleration.
    Provides the same interface as the Warp version.
    """
    
    def __init__(self, max_stack_depth: int = 64):
        self.max_stack_depth = max_stack_depth
        self.meshes = {}  # Store registered meshes
        self.current_mesh_vertices = None
        self.current_mesh_triangles = None
        self.current_mesh_bvh = None
        
    def register_mesh(self, mesh_id: int, vertices: np.ndarray, triangles: np.ndarray):
        """
        Register a mesh for ray casting.
        
        Args:
            mesh_id: Unique identifier for the mesh
            vertices: Nx3 array of vertex positions
            triangles: Mx3 array of triangle indices
        """
        mesh_data = MeshData(vertices, triangles)
        
        # Build BVH for this mesh
        mesh_data.bvh = LBVH(mesh_data.triangle_aabbs)
        mesh_data.bvh.build()
        
        self.meshes[mesh_id] = mesh_data
        
        # Set as current mesh for kernel access (simplified approach)
        # In a full implementation, we'd support multiple meshes
        self.current_mesh_vertices = mesh_data.vertices
        self.current_mesh_triangles = mesh_data.triangles
    
    @ti.func
    def ray_triangle_intersection_moller_trumbore(self, ray_start: ti.math.vec3, ray_dir: ti.math.vec3, 
                                                  v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3) -> ti.math.vec4:
        """
        Möller-Trumbore ray-triangle intersection algorithm.
        More straightforward than Woop and easier to implement correctly in Taichi.
        
        Returns: vec4(t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
        """
        result = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        
        # Compute edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Begin calculating determinant - also used to calculate u parameter
        h = ray_dir.cross(edge2)
        a = edge1.dot(h)
        
        # Check all conditions in sequence without early returns
        valid = True
        
        # Declare all variables at the top to avoid scope issues
        t = 0.0
        u = 0.0
        v = 0.0
        f = 0.0
        s = ti.math.vec3(0.0, 0.0, 0.0)
        q = ti.math.vec3(0.0, 0.0, 0.0)
        
        # If determinant is near zero, ray lies in plane of triangle
        if ti.abs(a) < 1e-8:
            valid = False
            
        if valid:
            f = 1.0 / a
            s = ray_start - v0
            u = f * s.dot(h)
            
            # Check u parameter bounds
            if u < 0.0 or u > 1.0:
                valid = False
                
        if valid:
            q = s.cross(edge1)
            v = f * ray_dir.dot(q)
            
            # Check v parameter bounds
            if v < 0.0 or u + v > 1.0:
                valid = False
                
        if valid:
            # At this stage we can compute t to find out where the intersection point is on the line
            t = f * edge2.dot(q)
            
            # Ray intersection
            if t <= 1e-8:  # Invalid intersection
                valid = False
        
        # Set result only if valid
        if valid:
            result = ti.math.vec4(t, u, v, 1.0)
            
        return result
    
    @ti.func
    def ray_aabb_intersection(self, ray_start: ti.math.vec3, ray_dir: ti.math.vec3, 
                              aabb_min: ti.math.vec3, aabb_max: ti.math.vec3) -> ti.f32:
        """
        Ray-AABB intersection test.
        Returns the t value of intersection, or -1.0 if no intersection.
        """
        # Compute intersection t value for each axis
        rcp_dir = ti.math.vec3(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z)
        
        # Handle potential division by zero
        if ti.abs(ray_dir.x) < 1e-10:
            rcp_dir.x = 1e10 if ray_dir.x >= 0.0 else -1e10
        if ti.abs(ray_dir.y) < 1e-10:
            rcp_dir.y = 1e10 if ray_dir.y >= 0.0 else -1e10
        if ti.abs(ray_dir.z) < 1e-10:
            rcp_dir.z = 1e10 if ray_dir.z >= 0.0 else -1e10
        
        t1 = (aabb_min - ray_start) * rcp_dir
        t2 = (aabb_max - ray_start) * rcp_dir
        
        tmin = ti.min(t1, t2)
        tmax = ti.max(t1, t2)
        
        t_near = ti.max(ti.max(tmin.x, tmin.y), tmin.z)
        t_far = ti.min(ti.min(tmax.x, tmax.y), tmax.z)
        
        # Check if ray intersects AABB
        if t_near <= t_far and t_far >= 0.0:
            return ti.max(t_near, 0.0)
        else:
            return -1.0
    
    @ti.func 
    def mesh_query_ray_simple(self, vertices: ti.template(), triangles: ti.template(), 
                              ray_start: ti.math.vec3, ray_dir: ti.math.vec3, max_t: ti.f32) -> ti.math.vec4:
        """
        Simple ray-mesh intersection without BVH (for initial implementation).
        Returns vec4(t, u, v, face_id) where t is distance, u,v are barycentric coords, face_id is triangle index.
        Returns vec4(max_t+1, 0, 0, -1) if no intersection.
        """
        result = ti.math.vec4(max_t + 1.0, 0.0, 0.0, -1.0)
        
        min_t = max_t
        hit_face = -1
        hit_u = 0.0
        hit_v = 0.0
        
        # Iterate through all triangles (brute force approach)
        n_triangles = triangles.shape[0]
        for i in range(n_triangles):
            # Get triangle vertices
            v0_idx = triangles[i][0]
            v1_idx = triangles[i][1]
            v2_idx = triangles[i][2]
            
            v0 = vertices[v0_idx]
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            
            # Perform ray-triangle intersection
            hit_result = self.ray_triangle_intersection_moller_trumbore(ray_start, ray_dir, v0, v1, v2)
            
            # Check if we have a valid hit closer than previous hits
            if hit_result.w > 0.0 and hit_result.x < min_t and hit_result.x >= 0.0:
                min_t = hit_result.x
                hit_face = i
                hit_u = hit_result.y
                hit_v = hit_result.z
        
        # Return the closest hit
        if hit_face >= 0:
            result = ti.math.vec4(min_t, hit_u, hit_v, ti.cast(hit_face, ti.f32))
        
        return result
    
    @ti.kernel
    def draw_optimized_kernel_pointcloud(
        self,
        mesh_ids: ti.types.ndarray(dtype=ti.u64, ndim=1),
        lidar_pos_array: ti.types.ndarray(dtype=ti.f32, ndim=3),  # [env, cam, 3]
        lidar_quat_array: ti.types.ndarray(dtype=ti.f32, ndim=3), # [env, cam, 4] 
        ray_vectors: ti.types.ndarray(dtype=ti.f32, ndim=3),      # [scan_line, point, 3]
        far_plane: ti.f32,
        pixels: ti.types.ndarray(dtype=ti.f32, ndim=5),           # [env, cam, scan_line, point, 3]
        local_dist: ti.types.ndarray(dtype=ti.f32, ndim=4),       # [env, cam, scan_line, point]
        pointcloud_in_world_frame: ti.i32,
        vertices: ti.template(),                                   # Mesh vertices
        triangles: ti.template(),                                 # Mesh triangles
    ):
        """
        Taichi kernel for LiDAR ray casting - matches the Warp interface.
        """
        # Get kernel execution dimensions
        n_envs = pixels.shape[0]
        n_cams = pixels.shape[1] 
        n_scan_lines = pixels.shape[2]
        n_points = pixels.shape[3]
        
        # Parallel execution over all rays
        for env_id, cam_id, scan_line, point_index in ti.ndrange(n_envs, n_cams, n_scan_lines, n_points):
            # Get LiDAR position and orientation
            lidar_position = ti.math.vec3(
                lidar_pos_array[env_id, cam_id, 0],
                lidar_pos_array[env_id, cam_id, 1], 
                lidar_pos_array[env_id, cam_id, 2]
            )
            
            lidar_quaternion = ti.math.vec4(
                lidar_quat_array[env_id, cam_id, 0],
                lidar_quat_array[env_id, cam_id, 1],
                lidar_quat_array[env_id, cam_id, 2],
                lidar_quat_array[env_id, cam_id, 3]
            )
            
            # Get ray direction in local coordinates
            ray_dir_local = ti.math.vec3(
                ray_vectors[scan_line, point_index, 0],
                ray_vectors[scan_line, point_index, 1],
                ray_vectors[scan_line, point_index, 2]
            )
            ray_dir_local = ray_dir_local.normalized()
            
            # Transform ray direction to world coordinates using quaternion
            ray_direction_world = self.quat_rotate(lidar_quaternion, ray_dir_local).normalized()
            
            # Initialize hit result
            dist = NO_HIT_RAY_VAL[None]
            
            # Perform ray casting using the mesh
            hit_result = self.mesh_query_ray_simple(vertices, triangles, lidar_position, ray_direction_world, far_plane)
            
            # Process hit result
            if hit_result.x < far_plane:
                dist = hit_result.x
                local_dist[env_id, cam_id, scan_line, point_index] = dist
                
                if pointcloud_in_world_frame:
                    hit_point = lidar_position + dist * ray_direction_world
                    pixels[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                    pixels[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                    pixels[env_id, cam_id, scan_line, point_index, 2] = hit_point.z
                else:
                    hit_point = dist * ray_dir_local
                    pixels[env_id, cam_id, scan_line, point_index, 0] = hit_point.x
                    pixels[env_id, cam_id, scan_line, point_index, 1] = hit_point.y
                    pixels[env_id, cam_id, scan_line, point_index, 2] = hit_point.z
            else:
                # No hit - set to default miss values
                local_dist[env_id, cam_id, scan_line, point_index] = NO_HIT_RAY_VAL[None]
                pixels[env_id, cam_id, scan_line, point_index, 0] = 0.0
                pixels[env_id, cam_id, scan_line, point_index, 1] = 0.0
                pixels[env_id, cam_id, scan_line, point_index, 2] = 0.0
    
    @staticmethod
    @ti.func
    def quat_rotate(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
        """
        Rotate vector v by quaternion q.
        Quaternion format: (x, y, z, w)
        """
        # Extract quaternion components
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
        # Quaternion multiplication: q * (0, v) * q_conjugate
        # First: q * (0, v)
        temp_w = -qx * v.x - qy * v.y - qz * v.z
        temp_x = qw * v.x + qy * v.z - qz * v.y
        temp_y = qw * v.y + qz * v.x - qx * v.z
        temp_z = qw * v.z + qx * v.y - qy * v.x
        
        # Second: result * q_conjugate
        result_x = temp_w * (-qx) + temp_x * qw + temp_y * (-qz) - temp_z * (-qy)
        result_y = temp_w * (-qy) + temp_y * qw + temp_z * (-qx) - temp_x * (-qz)
        result_z = temp_w * (-qz) + temp_z * qw + temp_x * (-qy) - temp_y * (-qx)
        
        return ti.math.vec3(result_x, result_y, result_z)


# Helper function to use the kernels (similar to Warp interface)
def create_lidar_taichi_kernels():
    """Create and return a LidarTaichiKernels instance."""
    return LidarTaichiKernels() 