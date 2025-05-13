import taichi as ti
import numpy as np
import torch
import time
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
            
            # Work queue fields for BVH construction
            self.work_queue_start = ti.field(dtype=ti.i32, shape=max_num_triangles)
            self.work_queue_end = ti.field(dtype=ti.i32, shape=max_num_triangles)
            self.work_queue_node = ti.field(dtype=ti.i32, shape=max_num_triangles)
            self.queue_size = ti.field(dtype=ti.i32, shape=())
            self.new_queue_size = ti.field(dtype=ti.i32, shape=())
            
            # BVH traversal structures
            self.bvh_stack = ti.field(dtype=ti.i32, shape=64)  # Stack for traversal (64 depth is plenty)
            self.node_stack = ti.field(dtype=ti.i32, shape=64)
            self.bvh_stack_size = ti.field(dtype=ti.i32, shape=())  # Current stack size
            # 添加额外字段来跟踪每个叶节点中的三角形数量
            self.bvh_node_triangle_count = ti.field(ti.i32, shape=max_nodes)
            
            # 添加排序后的三角形索引数组
            self.sorted_triangle_indices = ti.field(ti.i32, shape=max_num_triangles)
            
            # 三角形临时工作空间
            self.triangle_temp_space = ti.field(ti.i32, shape=max_num_triangles)
    def load_mesh_from_numpy(self, vertices, triangles):
        """Load mesh data from numpy arrays and build BVH"""
        num_vertices = vertices.shape[0]
        num_triangles = triangles.shape[0]
        
        # 设置计数
        self.num_vertices[None] = num_vertices
        self.num_triangles[None] = num_triangles
        
        # 复制数据到Taichi字段
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
        
        self.validate_bvh()
        self.test_ray_intersection()
        #self._build_bvh_debug()
        print(f"BVH built with {self.bvh_num_nodes[None]} nodes for {num_triangles} triangles")
        
        # Test basic ray intersection
        self.test_ray_intersection()  
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
        """改进的BVH构建方法，包括三角形重排"""
        # 创建根节点（包含所有三角形）
        self.bvh_root[None] = 0
        self.bvh_num_nodes[None] = 1
        
        # 初始化三角形索引 - 确保所有三角形都有唯一索引
        for i in range(self.num_triangles[None]):
            self.sorted_triangle_indices[i] = i
        
        # 计算场景AABB
        scene_min = ti.Vector([1e10, 1e10, 1e10])
        scene_max = ti.Vector([-1e10, -1e10, -1e10])
        
        for i in range(self.num_triangles[None]):
            scene_min = ti.min(scene_min, self.tri_aabbs_min[i])
            scene_max = ti.max(scene_max, self.tri_aabbs_max[i])
        
        # 设置根节点边界
        self.bvh_nodes_min[0] = scene_min
        self.bvh_nodes_max[0] = scene_max
        self.bvh_node_is_leaf[0] = 0  # 初始不是叶节点
        self.bvh_node_triangle_count[0] = self.num_triangles[None]  # Add this line
        
        # 从根节点开始
        self.work_queue_start[0] = 0
        self.work_queue_end[0] = self.num_triangles[None]
        self.work_queue_node[0] = 0
        self.queue_size[None] = 1
        
        # 最大递归深度
        max_depth = 20
        
        # 节点计数器
        node_counter = 1  # 根节点为0
        
        # 处理队列
        depth = 0
        while depth < max_depth and self.queue_size[None] > 0:
            self.new_queue_size[None] = 0
            
            # 处理当前层级
            for q in range(self.queue_size[None]):
                start_idx = self.work_queue_start[q]
                end_idx = self.work_queue_end[q]
                node_idx = self.work_queue_node[q]
                
                # 获取三角形数量
                tri_count = end_idx - start_idx
                
                # 如果三角形很少或达到最大深度，创建叶节点
                if tri_count <= 4 or depth >= max_depth - 1:
                    self.bvh_node_is_leaf[node_idx] = 1
                    self.bvh_node_triangle_idx[node_idx] = start_idx
                    self.bvh_node_triangle_count[node_idx] = tri_count
                    continue
                
                # 找到节点AABB最长轴
                node_min = self.bvh_nodes_min[node_idx]
                node_max = self.bvh_nodes_max[node_idx]
                node_extent = node_max - node_min
                
                # 确定分割轴 - 选择最长的轴
                split_axis = 0
                if node_extent[1] > node_extent[0]:
                    split_axis = 1
                if node_extent[2] > node_extent[split_axis]:
                    split_axis = 2
                    
                # 分割坐标 - 使用节点中心而不是简单的中点
                split_coord = (node_min[split_axis] + node_max[split_axis]) * 0.5
                
                # 分区三角形 - 基于质心位置
                mid = start_idx  # 分区点
                
                # 第一遍：将质心在分割平面左侧的三角形移到左侧
                for i in range(start_idx, end_idx):
                    tri_idx = self.sorted_triangle_indices[i]
                    # 检查三角形质心是否在分割平面左侧
                    if self.centroids[tri_idx][split_axis] < split_coord:
                        # 如果是，与 mid 位置交换
                        temp = self.sorted_triangle_indices[i]
                        self.sorted_triangle_indices[i] = self.sorted_triangle_indices[mid]
                        self.sorted_triangle_indices[mid] = temp
                        mid += 1
                
                # 如果分割不均衡 (所有在一边)，强制平分
                if mid == start_idx or mid == end_idx:
                    mid = start_idx + tri_count // 2  # 强制平分
                    
                    # 重置排序，确保连续
                    for i in range(start_idx, end_idx):
                        self.triangle_temp_space[i-start_idx] = self.sorted_triangle_indices[i]
                    
                    # 重新按分割轴排序
                    for i in range(tri_count):
                        self.sorted_triangle_indices[start_idx + i] = self.triangle_temp_space[i]
                
                # 创建子节点
                left_idx = node_counter
                right_idx = node_counter + 1
                node_counter += 2
                
                # 更新父节点
                self.bvh_node_left[node_idx] = left_idx
                self.bvh_node_right[node_idx] = right_idx
                self.bvh_node_parent[left_idx] = node_idx
                self.bvh_node_parent[right_idx] = node_idx
                
                # 初始化子节点
                self.bvh_node_is_leaf[left_idx] = 0
                self.bvh_node_is_leaf[right_idx] = 0
                
                # 计算子节点AABB
                left_min = ti.Vector([1e10, 1e10, 1e10])
                left_max = ti.Vector([-1e10, -1e10, -1e10])
                right_min = ti.Vector([1e10, 1e10, 1e10])
                right_max = ti.Vector([-1e10, -1e10, -1e10])
                
                # 计算左子树AABB - 包含所有左侧三角形的边界
                for i in range(start_idx, mid):
                    tri_idx = self.sorted_triangle_indices[i]
                    left_min = ti.min(left_min, self.tri_aabbs_min[tri_idx])
                    left_max = ti.max(left_max, self.tri_aabbs_max[tri_idx])
                
                # 计算右子树AABB
                for i in range(mid, end_idx):
                    tri_idx = self.sorted_triangle_indices[i]
                    right_min = ti.min(right_min, self.tri_aabbs_min[tri_idx])
                    right_max = ti.max(right_max, self.tri_aabbs_max[tri_idx])
                
                # 设置子节点AABB
                self.bvh_nodes_min[left_idx] = left_min
                self.bvh_nodes_max[left_idx] = left_max
                self.bvh_nodes_min[right_idx] = right_min
                self.bvh_nodes_max[right_idx] = right_max
                
                # 添加到工作队列
                # 左子节点
                new_q_idx = self.new_queue_size[None]
                self.work_queue_start[new_q_idx] = start_idx
                self.work_queue_end[new_q_idx] = mid
                self.work_queue_node[new_q_idx] = left_idx
                self.new_queue_size[None] += 1
                
                # 右子节点
                new_q_idx = self.new_queue_size[None]
                self.work_queue_start[new_q_idx] = mid
                self.work_queue_end[new_q_idx] = end_idx
                self.work_queue_node[new_q_idx] = right_idx
                self.new_queue_size[None] += 1
            
            # 更新下一层级的队列
            for i in range(self.new_queue_size[None]):
                self.work_queue_start[i] = self.work_queue_start[self.queue_size[None] + i]
                self.work_queue_end[i] = self.work_queue_end[self.queue_size[None] + i]
                self.work_queue_node[i] = self.work_queue_node[self.queue_size[None] + i]
            
            self.queue_size[None] = self.new_queue_size[None]
            
            # 层级递增
            depth += 1
                
        # 更新节点总数
        self.bvh_num_nodes[None] = node_counter 
    @ti.func
    def ray_aabb_intersection(self, ray_origin, rcp_dir, aabb_min, aabb_max):
        """Check if a ray intersects an AABB using slab method"""
        # Initialize t values
        t_min = -1e30
        t_max = 1e30
        hit = True  # Assume hit until proven otherwise
        
        # Check each dimension
        for i in ti.static(range(3)):
            if ti.abs(rcp_dir[i]) < 1e-8:
                # Ray is parallel to slab in this dimension
                if ray_origin[i] < aabb_min[i] or ray_origin[i] > aabb_max[i]:
                    # Ray origin is outside slab, no intersection
                    hit = False
            else:
                # Compute intersection t values with near and far planes
                t1 = (aabb_min[i] - ray_origin[i]) * rcp_dir[i]
                t2 = (aabb_max[i] - ray_origin[i]) * rcp_dir[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                    
                # Update intersection interval
                t_min = ti.max(t_min, t1)
                t_max = ti.min(t_max, t2)
                
                if t_max < t_min:
                    hit = False
        
        # Check if intersection is in positive ray direction
        hit = hit and t_max >= 0.0 and t_min <= t_max
        t_hit = t_min if t_min > 0.0 else t_max
        
        return hit, t_hit
    
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
        """Lidar ray tracing kernel with optimized BVH traversal"""
        # Use block_dim for better GPU performance
        ti.loop_config(block_dim=128)
        
        # Get array dimensions
        num_envs = ray_origins.shape[0]
        num_cams = ray_origins.shape[1]
        num_vertical = ray_directions.shape[0]
        num_horizontal = ray_directions.shape[1]
        
        for env_id, cam_id, scan_line, point_index in ti.ndrange(num_envs, num_cams, num_vertical, num_horizontal):
            # Get ray origin and direction
            ray_origin = ray_origins[env_id, cam_id]
            ray_dir = ray_directions[scan_line, point_index]
            
            # Normalize ray direction for consistent behavior
            ray_dir_length = ray_dir.norm()
            if ray_dir_length > 1e-8:
                ray_dir = ray_dir / ray_dir_length
            
            # Query ray intersection with mesh using BVH traversal
            hit_found, t, u, v, sign, normal, face = self.mesh_query_ray_bvh(ray_origin, ray_dir, far_plane)
            
            # Store hit distance
            if hit_found == 1:
                hit_distances[env_id, cam_id, scan_line, point_index] = t
                
                # Calculate hit point based on configuration
                if pointcloud_in_world_frame == 1:
                    # World space
                    hit_pos = ray_origin + t * ray_dir
                    hit_points[env_id, cam_id, scan_line, point_index] = hit_pos
                else:
                    # Sensor space
                    hit_points[env_id, cam_id, scan_line, point_index] = t * ray_dir
            else:
                # No hit - set to max range
                hit_distances[env_id, cam_id, scan_line, point_index] = far_plane
                
                # Set hit point based on configuration
                if pointcloud_in_world_frame == 1:
                    hit_points[env_id, cam_id, scan_line, point_index] = ray_origin + far_plane * ray_dir
                else:
                    hit_points[env_id, cam_id, scan_line, point_index] = far_plane * ray_dir
    @ti.kernel
    def draw_lidar_pointcloud_brute_force(self, 
                                        ray_origins: ti.template(),
                                        ray_directions: ti.template(),
                                        far_plane: ti.f32,
                                        hit_points: ti.template(),
                                        hit_distances: ti.template(),
                                        pointcloud_in_world_frame: ti.i32):
        """使用暴力方法的光线追踪内核，用于调试"""
        num_envs = ray_origins.shape[0]
        num_cams = ray_origins.shape[1] 
        num_vertical = ray_directions.shape[0]
        num_horizontal = ray_directions.shape[1]
        
        for env_id, cam_id, scan_line, point_index in ti.ndrange(num_envs, num_cams, num_vertical, num_horizontal):
            ray_origin = ray_origins[env_id, cam_id]
            ray_dir = ray_directions[scan_line, point_index]
            
            # 规范化光线方向
            ray_dir_length = ray_dir.norm()
            if ray_dir_length > 1e-8:
                ray_dir = ray_dir / ray_dir_length
            
            # 简单暴力搜索所有三角形
            hit_found, t, u, v, sign, normal, face = self.mesh_query_ray(ray_origin, ray_dir, far_plane)
            
            if hit_found == 1:
                hit_distances[env_id, cam_id, scan_line, point_index] = t
                if pointcloud_in_world_frame == 1:
                    hit_points[env_id, cam_id, scan_line, point_index] = ray_origin + t * ray_dir
                else:
                    hit_points[env_id, cam_id, scan_line, point_index] = t * ray_dir
            else:
                hit_distances[env_id, cam_id, scan_line, point_index] = far_plane
                if pointcloud_in_world_frame == 1:
                    hit_points[env_id, cam_id, scan_line, point_index] = ray_origin + far_plane * ray_dir
                else:
                    hit_points[env_id, cam_id, scan_line, point_index] = far_plane * ray_dir

    @ti.kernel
    def test_ray_intersection(self):
        """测试固定射线与固定三角形相交，用于验证算法"""
        # 创建一个简单的三角形
        v0 = ti.Vector([0.0, 0.0, 0.0])
        v1 = ti.Vector([1.0, 0.0, 0.0]) 
        v2 = ti.Vector([0.0, 1.0, 0.0])
        
        # 创建一个射向三角形的射线
        ray_origin = ti.Vector([0.2, 0.2, 1.0])
        ray_dir = ti.Vector([0.0, 0.0, -1.0])  # 直接指向三角形
        
        # 测试相交
        hit, t, u, v, sign, normal = self.ray_triangle_intersection_moller(
            ray_origin, ray_dir, v0, v1, v2)
        
        print("Test intersection:", hit, t)
        
        # 再测试一条不会相交的射线
        ray_dir2 = ti.Vector([0.0, 0.0, 1.0])  # 方向远离三角形
        hit2, t2, u2, v2, sign2, normal2 = self.ray_triangle_intersection_moller(
            ray_origin, ray_dir2, v0, v1, v2)
        
        print("Test non-intersection:", hit2, t2)
    def create_lidar_pointcloud(self, ray_origins, ray_directions, far_plane, hit_points, hit_distances, pointcloud_in_world_frame):
        """Launch the lidar ray tracing kernel with PyTorch tensor support"""
        import torch
        import numpy as np
        
        # Convert PyTorch tensors to numpy arrays
        if isinstance(ray_origins, torch.Tensor):
            ray_origins_np = ray_origins.cpu().numpy()
        else:
            ray_origins_np = ray_origins
            
        if isinstance(ray_directions, torch.Tensor):
            ray_directions_np = ray_directions.cpu().numpy()
        else:
            ray_directions_np = ray_directions
        
        # Create Taichi fields with the appropriate shapes
        ray_origins_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_origins_np.shape[:2])
        ray_directions_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_directions_np.shape[:2])
        
        # Output fields
        hit_points_shape = hit_points.shape
        hit_distances_shape = hit_distances.shape
        
        hit_points_field = ti.Vector.field(3, dtype=ti.f32, shape=hit_points_shape[:-1])
        hit_distances_field = ti.field(dtype=ti.f32, shape=hit_distances_shape)
        
        # Copy data to Taichi fields
        ray_origins_field.from_numpy(ray_origins_np)
        ray_directions_field.from_numpy(ray_directions_np)
        
        # Launch kernel
        self.draw_lidar_pointcloud(
            ray_origins_field,
            ray_directions_field,
            far_plane,
            hit_points_field,
            hit_distances_field,
            1 if pointcloud_in_world_frame else 0
        )
        
        # self.draw_lidar_pointcloud_brute_force(
        #     ray_origins_field,
        #     ray_directions_field,
        #     far_plane,
        #     hit_points_field,
        #     hit_distances_field,
        #     1 if pointcloud_in_world_frame else 0
        # )
        
        # Copy results back to PyTorch tensors
        hit_points_np = hit_points_field.to_numpy()
        hit_distances_np = hit_distances_field.to_numpy()

        hit_counts = np.sum(hit_distances_np < far_plane)
        total_rays = np.prod(hit_distances_np.shape)
        print(f"Hit rate: {hit_counts}/{total_rays} ({hit_counts/total_rays*100:.2f}%)")
        
        # 如果没有命中，打印一些额外信息
        if hit_counts == 0:
            print("No hits detected! Debugging information:")
            print(f"Ray origins shape: {ray_origins_np.shape}")
            print(f"Ray directions shape: {ray_directions_np.shape}")
            print(f"BVH nodes: {self.bvh_num_nodes[None]}")
            print(f"Number of triangles: {self.num_triangles[None]}")
            
            # 打印一个示例光线和一个示例三角形用于调试
            sample_origin = ray_origins_np[0, 0]
            sample_dir = ray_directions_np[0, 0]
            print(f"Sample ray origin: {sample_origin}")
            print(f"Sample ray direction: {sample_dir}")
            
            if self.num_triangles[None] > 0:
                i0 = self.indices[0]
                i1 = self.indices[1]
                i2 = self.indices[2]
                v0 = self.points[i0]
                v1 = self.points[i1]
                v2 = self.points[i2]
                print(f"Sample triangle: [{v0}, {v1}, {v2}]")
                
        # Convert numpy arrays to PyTorch tensors and update the input tensors
        hit_points_out = torch.from_numpy(hit_points_np)
        hit_distances_out = torch.from_numpy(hit_distances_np)
        
        # Copy to the original tensors (preserving device)
        if isinstance(hit_points, torch.Tensor):
            hit_points.copy_(hit_points_out.to(hit_points.device))
        if isinstance(hit_distances, torch.Tensor):
            hit_distances.copy_(hit_distances_out.to(hit_distances.device))
    def create_brute_lidar_pointcloud(self, ray_origins, ray_directions, far_plane, hit_points, hit_distances, pointcloud_in_world_frame):
            """Launch the lidar ray tracing kernel with PyTorch tensor support"""
            import torch
            import numpy as np
            
            # Convert PyTorch tensors to numpy arrays
            if isinstance(ray_origins, torch.Tensor):
                ray_origins_np = ray_origins.cpu().numpy()
            else:
                ray_origins_np = ray_origins
                
            if isinstance(ray_directions, torch.Tensor):
                ray_directions_np = ray_directions.cpu().numpy()
            else:
                ray_directions_np = ray_directions
            
            # Create Taichi fields with the appropriate shapes
            ray_origins_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_origins_np.shape[:2])
            ray_directions_field = ti.Vector.field(3, dtype=ti.f32, shape=ray_directions_np.shape[:2])
            
            # Output fields
            hit_points_shape = hit_points.shape
            hit_distances_shape = hit_distances.shape
            
            hit_points_field = ti.Vector.field(3, dtype=ti.f32, shape=hit_points_shape[:-1])
            hit_distances_field = ti.field(dtype=ti.f32, shape=hit_distances_shape)
            
            # Copy data to Taichi fields
            ray_origins_field.from_numpy(ray_origins_np)
            ray_directions_field.from_numpy(ray_directions_np)
            
            # Launch kernel
            # self.draw_lidar_pointcloud(
            #     ray_origins_field,
            #     ray_directions_field,
            #     far_plane,
            #     hit_points_field,
            #     hit_distances_field,
            #     1 if pointcloud_in_world_frame else 0
            # )
            
            self.draw_lidar_pointcloud_brute_force(
                ray_origins_field,
                ray_directions_field,
                far_plane,
                hit_points_field,
                hit_distances_field,
                1 if pointcloud_in_world_frame else 0
            )
            
            # Copy results back to PyTorch tensors
            hit_points_np = hit_points_field.to_numpy()
            hit_distances_np = hit_distances_field.to_numpy()

            hit_counts = np.sum(hit_distances_np < far_plane)
            total_rays = np.prod(hit_distances_np.shape)
            print(f"Hit rate: {hit_counts}/{total_rays} ({hit_counts/total_rays*100:.2f}%)")
            
            # 如果没有命中，打印一些额外信息
            if hit_counts == 0:
                print("No hits detected! Debugging information:")
                print(f"Ray origins shape: {ray_origins_np.shape}")
                print(f"Ray directions shape: {ray_directions_np.shape}")
                print(f"BVH nodes: {self.bvh_num_nodes[None]}")
                print(f"Number of triangles: {self.num_triangles[None]}")
                
                # 打印一个示例光线和一个示例三角形用于调试
                sample_origin = ray_origins_np[0, 0]
                sample_dir = ray_directions_np[0, 0]
                print(f"Sample ray origin: {sample_origin}")
                print(f"Sample ray direction: {sample_dir}")
                
                if self.num_triangles[None] > 0:
                    i0 = self.indices[0]
                    i1 = self.indices[1]
                    i2 = self.indices[2]
                    v0 = self.points[i0]
                    v1 = self.points[i1]
                    v2 = self.points[i2]
                    print(f"Sample triangle: [{v0}, {v1}, {v2}]")
                    
            # Convert numpy arrays to PyTorch tensors and update the input tensors
            hit_points_out = torch.from_numpy(hit_points_np)
            hit_distances_out = torch.from_numpy(hit_distances_np)
            
            # Copy to the original tensors (preserving device)
            if isinstance(hit_points, torch.Tensor):
                hit_points.copy_(hit_points_out.to(hit_points.device))
            if isinstance(hit_distances, torch.Tensor):
                hit_distances.copy_(hit_distances_out.to(hit_distances.device))
                
                
                
    @ti.func
    def ray_aabb_intersection_robust(self, ray_origin, ray_dir, aabb_min, aabb_max):
        """Improved ray-AABB intersection that correctly handles all edge cases"""
        # Initialization - assume no hit initially
        hit = False
        t_hit = 1e30
        
        # Check for degenerate boxes
        valid_box = False
        for i in ti.static(range(3)):
            if aabb_min[i] < aabb_max[i]:
                valid_box = True
        
        if valid_box:
            # Calculate inverse ray direction (with safe division)
            inv_dir = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                if ti.abs(ray_dir[i]) > 1e-8:
                    inv_dir[i] = 1.0 / ray_dir[i]
                else:
                    # For parallel rays, set to a very large value with correct sign
                    inv_dir[i] = 1e20 if ray_dir[i] >= 0 else -1e20
            
            # Calculate intersection distances with all slabs
            t_near = -1e30
            t_far = 1e30
            
            # Calculate entry and exit points for each dimension
            for i in ti.static(range(3)):
                t1 = (aabb_min[i] - ray_origin[i]) * inv_dir[i]
                t2 = (aabb_max[i] - ray_origin[i]) * inv_dir[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update intersection interval
                t_near = ti.max(t_near, t1)
                t_far = ti.min(t_far, t2)
            
            # We have a valid hit if the exit point is after the entry point and at least one is positive
            hit = (t_far >= t_near) and (t_far > 0.0)
            
            # Return the earliest positive intersection
            if hit:
                # If ray origin is outside the box, return entry point
                # Otherwise return exit point (from inside the box)
                if t_near > 0.0:
                    t_hit = t_near
                else:
                    t_hit = t_far
        
        return hit, t_hit
    

    @ti.func
    def mesh_query_ray_bvh(self, ray_origin, ray_dir, max_t):
        """Improved BVH traversal with better handling of edge cases"""
        # Initialize result variables
        min_t = max_t
        min_face = -1
        min_u = 0.0
        min_v = 0.0
        min_sign = 1.0
        min_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_found = 0

        # Normalize ray direction
        ray_dir_normalized = ray_dir
        ray_dir_length = ray_dir.norm()
        if ray_dir_length > 1e-8:
            ray_dir_normalized = ray_dir / ray_dir_length
        
        # Use existing bvh_stack field instead of creating a local stack
        stack_ptr = 0
        max_stack_size = 64
        
        # Start with root node
        self.bvh_stack[0] = self.bvh_root[None]
        stack_ptr = 1
        
        # Track visited nodes to avoid excessive traversal
        nodes_visited = 0
        max_nodes = 256  # Higher limit for complex scenes
        
        while stack_ptr > 0 and nodes_visited < max_nodes:
            # Pop node from stack
            stack_ptr -= 1
            node_idx = self.bvh_stack[stack_ptr]
            nodes_visited += 1
            
            # Get node bounds
            node_min = self.bvh_nodes_min[node_idx]
            node_max = self.bvh_nodes_max[node_idx]
            
            # Test intersection with this node's AABB
            box_hit, box_t = self.ray_aabb_intersection_robust(ray_origin, ray_dir_normalized, node_min, node_max)
            
            # Only process nodes that are hit and closer than our current closest hit
            if box_hit and box_t < min_t:
                # Check if this is a leaf node
                if self.bvh_node_is_leaf[node_idx] == 1:
                    # Process triangles in leaf node
                    start_idx = self.bvh_node_triangle_idx[node_idx]
                    tri_count = self.bvh_node_triangle_count[node_idx]
                    
                    # Only process valid triangle ranges
                    if tri_count > 0 and start_idx >= 0 and start_idx + tri_count <= self.num_triangles[None]:
                        # Process each triangle
                        for i in range(start_idx, start_idx + tri_count):
                            tri_idx = self.sorted_triangle_indices[i]
                            
                            # Safety check on triangle index
                            if tri_idx >= 0 and tri_idx < self.num_triangles[None]:
                                # Get triangle vertices
                                idx0 = self.indices[tri_idx * 3]
                                idx1 = self.indices[tri_idx * 3 + 1]
                                idx2 = self.indices[tri_idx * 3 + 2]
                                
                                # Safety check on vertex indices
                                if (idx0 >= 0 and idx0 < self.num_vertices[None] and
                                    idx1 >= 0 and idx1 < self.num_vertices[None] and
                                    idx2 >= 0 and idx2 < self.num_vertices[None]):
                                    
                                    v0 = self.points[idx0]
                                    v1 = self.points[idx1]
                                    v2 = self.points[idx2]
                                    
                                    # Test ray against triangle
                                    hit, t, u, v, sign, normal = self.ray_triangle_intersection_moller(
                                        ray_origin, ray_dir_normalized, v0, v1, v2)
                                    
                                    if hit and t > 0.0 and t < min_t:
                                        min_t = t
                                        min_face = tri_idx
                                        min_u = u
                                        min_v = v
                                        min_sign = sign
                                        min_normal = normal
                                        hit_found = 1
                else:
                    # Internal node - process children
                    left_idx = self.bvh_node_left[node_idx]
                    right_idx = self.bvh_node_right[node_idx]
                    
                    # Verify child indices are valid
                    valid_left = left_idx >= 0 and left_idx < self.bvh_num_nodes[None]
                    valid_right = right_idx >= 0 and right_idx < self.bvh_num_nodes[None]
                    
                    # Only process valid children
                    if valid_left and stack_ptr < max_stack_size:
                        self.node_stack[stack_ptr] = left_idx
                        stack_ptr += 1
                    
                    if valid_right and stack_ptr < max_stack_size:
                        self.node_stack[stack_ptr] = right_idx
                        stack_ptr += 1
        
        return hit_found, min_t, min_u, min_v, min_sign, min_normal, min_face

    @ti.kernel
    def validate_bvh(self):
        """Validate BVH structure and identify common issues"""
        # Check for degenerate boxes
        degenerate_boxes = 0
        for i in range(self.bvh_num_nodes[None]):
            node_min = self.bvh_nodes_min[i]
            node_max = self.bvh_nodes_max[i]
            
            # Check if box has zero volume
            is_degenerate = True
            for j in ti.static(range(3)):
                if node_max[j] > node_min[j]:
                    is_degenerate = False
            
            if is_degenerate:
                degenerate_boxes += 1
                print(f"Node {i} has degenerate box: min={node_min}, max={node_max}")
        
        print(f"BVH validation complete: {degenerate_boxes} degenerate boxes found")
        
        # Check leaf nodes contain valid triangle indices
        invalid_leaf_nodes = 0
        for i in range(self.bvh_num_nodes[None]):
            if self.bvh_node_is_leaf[i] == 1:
                start_idx = self.bvh_node_triangle_idx[i]
                tri_count = self.bvh_node_triangle_count[i]
                
                if start_idx < 0 or start_idx + tri_count > self.num_triangles[None] or tri_count <= 0:
                    invalid_leaf_nodes += 1
                    print(f"Node {i} has invalid triangle range: start={start_idx}, count={tri_count}")
        
        print(f"Leaf node validation: {invalid_leaf_nodes} invalid leaf nodes found")
        
    @ti.kernel
    def debug_ray_box(self, ray_origin: ti.template(), ray_dir: ti.template()):
        """Debug ray-box intersections with a specific ray against the BVH"""
        # Normalize direction
        ray_norm = ray_dir.norm()
        ray_dir_normalized = ray_dir
        if ray_norm > 1e-8:
            ray_dir_normalized = ray_dir / ray_norm
        
        # Test against the root node first
        root_idx = self.bvh_root[None]
        root_min = self.bvh_nodes_min[root_idx]
        root_max = self.bvh_nodes_max[root_idx]
        
        hit, t = self.ray_aabb_intersection_robust(ray_origin, ray_dir_normalized, root_min, root_max)
        print(f"Root node test: hit={hit}, t={t}, min={root_min}, max={root_max}")
        
        # Test against first few nodes
        for i in range(min(10, self.bvh_num_nodes[None])):
            node_min = self.bvh_nodes_min[i]
            node_max = self.bvh_nodes_max[i]
            is_leaf = self.bvh_node_is_leaf[i]
            
            hit, t = self.ray_aabb_intersection_robust(ray_origin, ray_dir_normalized, node_min, node_max)
            if is_leaf:
                tri_start = self.bvh_node_triangle_idx[i]
                tri_count = self.bvh_node_triangle_count[i]
                print(f"Node {i} (LEAF): hit={hit}, t={t}, triangles={tri_count}")
            else:
                left = self.bvh_node_left[i]
                right = self.bvh_node_right[i]
                print(f"Node {i} (INNER): hit={hit}, t={t}, left={left}, right={right}")