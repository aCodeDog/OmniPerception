import time
import mujoco
import numpy as np
import taichi as ti
import argparse

ti.init(
    arch=ti.gpu, 
    kernel_profiler=True,
    advanced_optimization=True,  # 启用高级优化
    offline_cache=True,          # 启用离线缓存
    default_fp=ti.f32,           # 设置默认浮点类型
    default_ip=ti.i32,           # 设置默认整数类型
    device_memory_GB=4.0,        # 限制设备内存使用
)

# mj_geom_type = {
#     0 : "mjGEOM_PLANE",
#     1 : "mjGEOM_HFIELD",
#     2 : "mjGEOM_SPHERE",
#     3 : "mjGEOM_CAPSULE",
#     4 : "mjGEOM_ELLIPSOID",
#     5 : "mjGEOM_CYLINDER",
#     6 : "mjGEOM_BOX"
# }
@ti.data_oriented
class MjLidarSensor:

    def __init__(self, mj_scene, enable_profiling=False, verbose=False):
        """
        初始化LiDAR传感器
        
        参数:
            mj_scene: MuJoCo场景对象
            enable_profiling: 是否启用性能分析（默认False）
            verbose: 是否打印详细信息（默认False）
        """
        self.n_geoms = mj_scene.ngeom
        self.enable_profiling = enable_profiling
        self.verbose = verbose
        
        if self.verbose:
            print(f"n_geoms: {self.n_geoms}")

        # 预分配所有Taichi字段，避免重复创建
        self.geom_types = ti.field(dtype=ti.i32, shape=(self.n_geoms))
        self.geom_sizes = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_geoms))
        self.geom_rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.n_geoms))  # 修改为矩阵字段
        
        # 初始化几何体静态数据
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            if geom.objtype != 5: # 5 is mjOBJ_GEOM
                self.geom_types[i] = -1
            else:
                self.geom_types[i] = geom.type
            self.geom_sizes[i] = ti.math.vec3(geom.size[0], geom.size[1], geom.size[2])
            self.geom_positions[i] = ti.math.vec3(geom.pos[0], geom.pos[1], geom.pos[2])
            # 保存旋转矩阵
            rot_mat = geom.mat.reshape(3, 3)
            for r in range(3):
                for c in range(3):
                    self.geom_rotations[i][r, c] = rot_mat[r, c]
        
        # 预先分配传感器位姿数组
        self.sensor_pose_ti = ti.ndarray(dtype=ti.f32, shape=(4, 4))
        
        # 缓存射线数据
        self.cached_n_rays = 0
        self.rays_phi_ti = None
        self.rays_theta_ti = None
        self.hit_points = None
        
        # 预分配临时数组（用于内核计算）
        self.hit_points_world = None  # 世界坐标系下的命中点
        self.hit_mask = None  # 射线命中标志
        
        # 性能统计
        self.kernel_time = 0
        self.prepare_time = 0
        self.total_time = 0
        
        # 详细性能统计
        self.update_geom_time = 0
        self.convert_sensor_pose_time = 0
        self.update_rays_time = 0
        self.memory_allocation_time = 0
        self.sync_time = 0

    def set_sensor_pose(self, sensor_pose):
        assert sensor_pose.shape == (4, 4) and sensor_pose.dtype == np.float32, f"sensor_pose must be a 4x4 numpy array, but got {sensor_pose.shape} and {sensor_pose.dtype}"
        self.sensor_pose_ti.from_numpy(sensor_pose)

    def update_geom_positions(self, mj_scene):
        """更新几何体位置和旋转数据"""
        start_time = time.time() if self.enable_profiling else 0
        
        # 预先分配NumPy数组以收集所有数据
        pos_data = np.zeros((self.n_geoms, 3), dtype=np.float32)
        rot_data = np.zeros((self.n_geoms, 3, 3), dtype=np.float32)
        
        # 在CPU上收集数据
        for i in range(self.n_geoms):
            geom = mj_scene.geoms[i]
            pos_data[i] = geom.pos
            rot_data[i] = geom.mat.reshape(3, 3)
        
        # 使用Taichi内核并行更新
        self.update_geom_positions_parallel(pos_data, rot_data)
        
        end_time = time.time() if self.enable_profiling else 0
        self.update_geom_time = (end_time - start_time) * 1000 if self.enable_profiling else 0

    @ti.kernel
    def update_geom_positions_parallel(self, 
                                       pos_data: ti.types.ndarray(dtype=ti.f32, ndim=2), 
                                       rot_data: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        """并行更新几何体位置和旋转数据"""
        # 并行遍历所有几何体
        for i in range(self.n_geoms):
            # 更新位置
            self.geom_positions[i] = ti.math.vec3(pos_data[i, 0], pos_data[i, 1], pos_data[i, 2])
            
            # 更新旋转矩阵
            for r in range(3):
                for c in range(3):
                    self.geom_rotations[i][r, c] = rot_data[i, r, c]

    @ti.func
    def transform_ray_to_local(self, ray_start, ray_direction, center, rotation):
        """将射线从世界坐标系转换到物体的局部坐标系"""
        # 先平移射线起点
        local_start = ray_start - center
        
        # 旋转矩阵的转置是其逆（假设正交矩阵）
        rot_transpose = ti.Matrix.zero(ti.f32, 3, 3)
        for i in range(3):
            for j in range(3):
                rot_transpose[i, j] = rotation[j, i]
        
        # 应用旋转
        local_start = rot_transpose @ local_start
        local_direction = rot_transpose @ ray_direction
        
        return local_start, local_direction
    
    @ti.func
    def transform_point_to_world(self, local_point, center, rotation):
        """将点从局部坐标系转换回世界坐标系"""
        # 应用旋转
        world_point = rotation @ local_point
        # 应用平移
        world_point = world_point + center
        return world_point
    
    @ti.func
    def ray_plane_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与平面的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到平面的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        # 在局部坐标系中，平面的法向量是z轴
        normal = ti.math.vec3(0.0, 0.0, 1.0)
        half_width = size[0]
        half_height = size[1]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        denom = local_direction.dot(normal)
        
        # 避免除以零，检查光线是否与平面平行
        if ti.abs(denom) >= 1e-6:
            # 局部坐标系中平面在原点，所以只需要计算到原点的距离
            t = -local_start.z / denom
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                local_hit = local_start + t * local_direction
                
                # 检查交点是否在平面范围内
                if ti.abs(local_hit.x) <= half_width and ti.abs(local_hit.y) <= half_height:
                    # 将交点转换回世界坐标系
                    world_hit = self.transform_point_to_world(local_hit, center, rotation)
                    hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_sphere_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # 注意：球体旋转不会改变其形状，所以可以简化计算
        radius = size[0]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        oc = ray_start - center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        # 计算交点
        if discriminant >= 0:
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为负，则使用较大的t值
            if t < 0:
                t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                hit_pos = ray_start + t * ray_direction
                hit_result = ti.math.vec4(hit_pos.x, hit_pos.y, hit_pos.z, t)
        
        return hit_result
    
    @ti.func
    def ray_box_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与盒子的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到盒子的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 处理局部坐标系中的射线方向为零的情况
        inv_dir = ti.math.vec3(
            1.0 / (local_direction.x if ti.abs(local_direction.x) > 1e-6 else 1e10),
            1.0 / (local_direction.y if ti.abs(local_direction.y) > 1e-6 else 1e10),
            1.0 / (local_direction.z if ti.abs(local_direction.z) > 1e-6 else 1e10)
        )
        
        t_min = -1e10  # 使用大数而不是无穷
        t_max = 1e10
        
        # 检查x轴
        t1 = (-size.x - local_start.x) * inv_dir.x
        t2 = (size.x - local_start.x) * inv_dir.x
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查y轴
        t1 = (-size.y - local_start.y) * inv_dir.y
        t2 = (size.y - local_start.y) * inv_dir.y
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 检查z轴
        t1 = (-size.z - local_start.z) * inv_dir.z
        t2 = (size.z - local_start.z) * inv_dir.z
        t_min = ti.max(t_min, ti.min(t1, t2))
        t_max = ti.min(t_max, ti.max(t1, t2))
        
        # 如果有有效的交点
        if t_max >= t_min and t_max >= 0:
            t = t_min if t_min >= 0 else t_max
            if t >= 0:
                # 计算局部坐标系中的交点
                local_hit = local_start + t * local_direction
                # 转换回世界坐标系
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_cylinder_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与圆柱体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # size[0]是半径，size[1]是半高
        
        # 转换射线到圆柱体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        radius = size[0]
        half_height = size[2]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 在局部坐标系中，圆柱体的中心轴与z轴平行
        # 仅考虑xy平面上的方向分量
        ray_dir_xy = ti.math.vec2(local_direction.x, local_direction.y)
        oc_xy = ti.math.vec2(local_start.x, local_start.y)
        
        # 解二次方程 at² + bt + c = 0
        a = ray_dir_xy.dot(ray_dir_xy)
        
        # 如果a很小，射线几乎与z轴平行
        if a < 1e-6:
            # 检查射线是否在圆柱体内部
            if oc_xy.norm() <= radius:
                # 计算与顶部或底部平面的交点
                t1 = (half_height - local_start.z) / local_direction.z
                t2 = (-half_height - local_start.z) / local_direction.z
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0 and (t2 < 0 or t1 < t2):
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                if t >= 0:
                    local_hit = local_start + t * local_direction
                    world_hit = self.transform_point_to_world(local_hit, center, rotation)
                    hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        else:
            # 标准的圆柱体-射线相交测试
            b = 2.0 * oc_xy.dot(ray_dir_xy)
            c = oc_xy.dot(oc_xy) - radius * radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant >= 0:
                # 计算圆柱侧面的两个可能交点
                sqrt_disc = ti.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                
                # 选择最小的正t值
                t = -1.0  # 默认为无效值
                if t1 >= 0:
                    t = t1
                elif t2 >= 0:
                    t = t2
                
                # 检查交点是否在圆柱体高度范围内
                if t >= 0:
                    local_hit = local_start + t * local_direction
                    
                    if ti.abs(local_hit.z) <= half_height:
                        # 交点在圆柱体侧面上
                        world_hit = self.transform_point_to_world(local_hit, center, rotation)
                        hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
                    else:
                        # 侧面交点不在圆柱体高度范围内，检查与顶部或底部平面的交点
                        cap_t = -1.0
                        
                        # 射线从上方射向底平面
                        if local_direction.z < 0 and local_start.z > half_height:
                            cap_t = (half_height - local_start.z) / local_direction.z
                        # 射线从下方射向顶平面
                        elif local_direction.z > 0 and local_start.z < -half_height:
                            cap_t = (-half_height - local_start.z) / local_direction.z
                        
                        if cap_t >= 0:
                            local_hit = local_start + cap_t * local_direction
                            cap_xy = ti.math.vec2(local_hit.x, local_hit.y)
                            
                            # 检查交点是否在圆盘内
                            if cap_xy.norm() <= radius:
                                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, cap_t)
        
        return hit_result
    
    @ti.func
    def ray_ellipsoid_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与椭球体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        
        # 转换射线到椭球体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 将问题转换为单位球相交，通过缩放空间
        inv_size = ti.math.vec3(1.0/size.x, 1.0/size.y, 1.0/size.z)
        
        # 缩放局部坐标系中的射线（不要归一化方向向量，这会改变t的意义）
        scaled_start = ti.math.vec3(
            local_start.x * inv_size.x,
            local_start.y * inv_size.y,
            local_start.z * inv_size.z
        )
        scaled_dir = ti.math.vec3(
            local_direction.x * inv_size.x,
            local_direction.y * inv_size.y,
            local_direction.z * inv_size.z
        )
        
        # 解二次方程 at² + bt + c = 0
        a = scaled_dir.dot(scaled_dir)
        b = 2.0 * scaled_start.dot(scaled_dir)
        c = scaled_start.dot(scaled_start) - 1.0  # 单位球半径为1
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # 计算两个可能的t值，取最小的正值
            t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            t = t1 if t1 >= 0 else t2
            
            # 如果t为正，表示有有效交点
            if t >= 0:
                # 使用原始射线方程计算交点
                local_hit = local_start + t * local_direction
                
                # 转换回世界坐标系
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, t)
        
        return hit_result
    
    @ti.func
    def ray_capsule_intersection(self, ray_start: ti.math.vec3, ray_direction: ti.math.vec3, center: ti.math.vec3, size: ti.math.vec3, rotation: ti.math.mat3) -> ti.math.vec4:
        """计算射线与胶囊体的交点"""
        # 返回格式: vec4(hit_x, hit_y, hit_z, t)，t为距离，t<0表示未击中
        # 在MuJoCo中: size[0]是半径，size[1]是圆柱部分的半高
        
        # 转换射线到胶囊体的局部坐标系
        local_start, local_direction = self.transform_ray_to_local(ray_start, ray_direction, center, rotation)
        
        radius = size[0]
        half_height = size[2]
        
        hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        # 计算胶囊体两个半球的中心（在局部坐标系中）
        # 半球中心点在距离胶囊体中心half_height处
        sphere1_center = ti.math.vec3(0.0, 0.0, half_height)
        sphere2_center = ti.math.vec3(0.0, 0.0, -half_height)
        
        # 为圆柱部分创建新的size
        cylinder_size = ti.math.vec3(radius, radius, half_height)
        identity_mat = ti.Matrix.identity(ti.f32, 3)  # 局部坐标系中用单位矩阵
        
        # 首先检查与圆柱体部分的交点（在局部坐标系中）
        cylinder_hit = self.ray_cylinder_intersection(local_start, local_direction, ti.math.vec3(0.0, 0.0, 0.0), cylinder_size, identity_mat)
        
        # 初始化最小距离为无穷大
        min_t = 1e10
        has_hit = False
        
        # 如果有圆柱体交点
        if cylinder_hit.w > 0 and cylinder_hit.w < min_t:
            min_t = cylinder_hit.w
            
            # 计算世界坐标系中的交点
            local_hit = local_start + cylinder_hit.w * local_direction
            world_hit = self.transform_point_to_world(local_hit, center, rotation)
            hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
            has_hit = True
        
        # 然后检查与两个半球的交点
        sphere_size = ti.math.vec3(radius, radius, radius)
        
        # 上半球
        sphere1_hit = self.ray_sphere_intersection(local_start, local_direction, sphere1_center, sphere_size, identity_mat)
        if sphere1_hit.w > 0 and sphere1_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的下半部分
            local_hit = local_start + sphere1_hit.w * local_direction
            local_z = local_hit.z - sphere1_center.z
            if local_z >= 0:  # 只取上半部分
                min_t = sphere1_hit.w
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
                has_hit = True
        
        # 下半球
        sphere2_hit = self.ray_sphere_intersection(local_start, local_direction, sphere2_center, sphere_size, identity_mat)
        if sphere2_hit.w > 0 and sphere2_hit.w < min_t:
            # 确保交点在半球内，而不是在完整球体的上半部分
            local_hit = local_start + sphere2_hit.w * local_direction
            local_z = local_hit.z - sphere2_center.z
            if local_z <= 0:  # 只取下半部分
                min_t = sphere2_hit.w
                world_hit = self.transform_point_to_world(local_hit, center, rotation)
                hit_result = ti.math.vec4(world_hit.x, world_hit.y, world_hit.z, min_t)
                has_hit = True
        
        # 如果没有任何交点，返回无效结果
        if not has_hit:
            hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
        
        return hit_result
    
    # 优化的Taichi核函数
    @ti.kernel
    def trace_rays(self, 
        sensor_origins: ti.types.ndarray(dtype=ti.f32, ndim=2), 
        rays_phi: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rays_theta: ti.types.ndarray(dtype=ti.f32, ndim=1),
        n_rays: ti.i32,
        hit_points: ti.template(),
    ):
        # 设置LiDAR传感器位置和姿态（只做一次）
        sensor_pose = ti.Matrix.identity(ti.f32, 4)
        for i in range(4):
            for j in range(4):
                sensor_pose[i, j] = sensor_origins[i, j]
        ray_start = sensor_pose[0:3, 3]  # 射线起点

        # 计算传感器位姿的逆矩阵（只计算一次）
        sensor_pose_inv = ti.Matrix.identity(ti.f32, 4)
        # 旋转部分的转置（因为正交矩阵的逆等于其转置）
        for i in range(3):
            for j in range(3):
                sensor_pose_inv[i, j] = sensor_pose[j, i]
        # 平移部分的变换
        for i in range(3):
            sensor_pose_inv[i, 3] = 0.0
            for j in range(3):
                sensor_pose_inv[i, 3] -= sensor_pose_inv[i, j] * sensor_pose[j, 3]

        # 使用预分配的字段，而不是在内核中创建新字段
        self.hit_points_world.fill(ti.Vector([0.0, 0.0, 0.0]))
        hit_points.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.hit_mask.fill(0)
                
        # 为每条射线并行计算
        ti.loop_config(block_dim=512)
        for i in range(n_rays):
            min_distance = 1e10

            # 计算射线方向（球坐标系转笛卡尔坐标系）
            phi = rays_phi[i]      # 垂直角度
            theta = rays_theta[i]  # 水平角度
            
            # 预计算三角函数值
            cos_theta = ti.cos(theta)
            sin_theta = ti.sin(theta)
            cos_phi = ti.cos(phi)
            sin_phi = ti.sin(phi)

            dir_local = ti.Vector([
                cos_phi * cos_theta,  # x分量
                cos_phi * sin_theta,  # y分量
                sin_phi               # z分量
            ]).normalized()  # 单位化方向向量

            ray_direction = (sensor_pose @ ti.Vector([dir_local.x, dir_local.y, dir_local.z, 0.0])).xyz.normalized()

            # 检查与每个几何体的交点
            for j in range(self.n_geoms):
                hit_result = ti.math.vec4(0.0, 0.0, 0.0, -1.0)
                
                # 获取几何体数据
                geom_type = self.geom_types[j]
                center = self.geom_positions[j]
                size = self.geom_sizes[j]
                rotation = self.geom_rotations[j]
                
                # 根据几何体类型调用相应的交点计算函数
                if geom_type == 0:  # PLANE
                    hit_result = self.ray_plane_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 2:  # SPHERE
                    hit_result = self.ray_sphere_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 3:  # CAPSULE
                    hit_result = self.ray_capsule_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 4:  # ELLIPSOID
                    hit_result = self.ray_ellipsoid_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 5:  # CYLINDER
                    hit_result = self.ray_cylinder_intersection(ray_start, ray_direction, center, size, rotation)
                elif geom_type == 6:  # BOX
                    hit_result = self.ray_box_intersection(ray_start, ray_direction, center, size, rotation)
                # 暂不支持HFIELD(1)

                # 检查是否有有效交点，并且是否是最近的
                if hit_result.w > 0 and hit_result.w < min_distance:
                    # 记录世界坐标系中的最近交点
                    self.hit_points_world[i] = ti.math.vec3(hit_result.x, hit_result.y, hit_result.z)
                    min_distance = hit_result.w
                    self.hit_mask[i] = 1  # 标记此射线有命中
                    # print(f"射线{i}命中几何体{j}，距离为{min_distance}")

            # 在遍历完当前射线的所有几何体后，进行坐标转换
            if self.hit_mask[i] == 1:  # 如果有命中
                world_hit = self.hit_points_world[i]
                # 转换为齐次坐标并应用传感器位姿的逆矩阵
                local_hit = sensor_pose_inv @ ti.Vector([world_hit.x, world_hit.y, world_hit.z, 1.0])
                hit_points[i] = local_hit.xyz

    def ray_cast_taichi(self, rays_phi, rays_theta, sensor_pose, mj_scene):
        """
        使用Taichi进行真正的并行光线追踪
        Params:
            rays_phi: 垂直角度数组
            rays_theta: 水平角度数组
            sensor_pose: 4x4 matrix, the pose of the sensor in the world frame
            mj_scene: mujoco.MjvScene object, the scene to cast rays into
        Return:
            Nx3 matrix, each row is the intersection point of the corresponding ray in the sensor frame
        """
        assert rays_phi.shape == rays_theta.shape, "rays_phi和rays_theta的形状必须相同"
        n_rays = rays_phi.shape[0]
        
        # 性能计时
        start_total = time.time()
        
        # 确保sensor_pose是float32类型
        convert_start = time.time() if self.enable_profiling else 0
        sensor_pose = sensor_pose.astype(np.float32)
        
        # 创建Taichi ndarray并从NumPy数组填充
        self.sensor_pose_ti.from_numpy(sensor_pose)
        convert_end = time.time() if self.enable_profiling else 0
        self.convert_sensor_pose_time = (convert_end - convert_start) * 1000 if self.enable_profiling else 0
        
        # 如果光线数量变化，重新分配内存
        memory_start = time.time() if self.enable_profiling else 0
        if self.cached_n_rays != n_rays:
            self.rays_phi_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.rays_theta_ti = ti.ndarray(dtype=ti.f32, shape=n_rays)
            self.hit_points = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            # 同时创建临时字段
            self.hit_points_world = ti.Vector.field(3, dtype=ti.f32, shape=n_rays)
            self.hit_mask = ti.field(dtype=ti.i32, shape=n_rays)
            self.cached_n_rays = n_rays
        memory_end = time.time() if self.enable_profiling else 0
        self.memory_allocation_time = (memory_end - memory_start) * 1000 if self.enable_profiling else 0
            
        # 更新光线数据
        rays_start = time.time() if self.enable_profiling else 0
        self.rays_phi_ti.from_numpy(rays_phi.astype(np.float32))
        self.rays_theta_ti.from_numpy(rays_theta.astype(np.float32))
        rays_end = time.time() if self.enable_profiling else 0
        self.update_rays_time = (rays_end - rays_start) * 1000 if self.enable_profiling else 0
        
        # 更新几何体位置
        self.update_geom_positions(mj_scene)
        
        # 准备阶段结束，记录时间
        prepare_end = time.time() if self.enable_profiling else 0
        self.prepare_time = (prepare_end - start_total) * 1000 if self.enable_profiling else 0
        
        # 开始Taichi内核计算
        sync_start = time.time() if self.enable_profiling else 0
        ti.sync()  # 确保之前的操作完成
        sync_end = time.time() if self.enable_profiling else 0
        self.sync_time = (sync_end - sync_start) * 1000 if self.enable_profiling else 0
        
        kernel_start = time.time() if self.enable_profiling else 0
        
        # 调用Taichi内核
        self.trace_rays(
            self.sensor_pose_ti,
            self.rays_phi_ti,
            self.rays_theta_ti,
            n_rays,
            self.hit_points
        )
        
        # 等待内核完成
        ti.sync()
        kernel_end = time.time() if self.enable_profiling else 0
        self.kernel_time = (kernel_end - kernel_start) * 1000 if self.enable_profiling else 0
        
        # 结果已经在内核中转换为局部坐标系
        result = self.hit_points.to_numpy()
        
        # 计算总时间
        end_total = time.time()
        self.total_time = (end_total - start_total) * 1000 if self.enable_profiling else 0
        
        # 打印详细性能信息
        if self.enable_profiling and self.verbose:
            print(f"准备阶段性能分析:")
            print(f"  - 传感器位姿转换时间: {self.convert_sensor_pose_time:.2f}ms")
            print(f"  - 内存分配时间: {self.memory_allocation_time:.2f}ms")
            print(f"  - 光线数据更新时间: {self.update_rays_time:.2f}ms")
            print(f"  - 几何体位置更新时间: {self.update_geom_time:.2f}ms")
            print(f"  - 同步操作时间: {self.sync_time:.2f}ms")
            print(f"总计: 准备时间: {self.prepare_time:.2f}ms, 内核时间: {self.kernel_time:.2f}ms, 总时间: {self.total_time:.2f}ms")
        
        return result


def create_lidar_rays(horizontal_resolution=360, vertical_resolution=32, horizontal_fov=2*np.pi, vertical_fov=np.pi/3):
    """创建激光雷达扫描线的角度数组"""
    h_angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, horizontal_resolution)
    v_angles = np.linspace(-vertical_fov/2, vertical_fov/2, vertical_resolution)

    theta_grid, phi_grid = np.meshgrid(h_angles, v_angles)
    
    # 展平网格为一维数组
    rays_theta = theta_grid.flatten()
    rays_phi = phi_grid.flatten()
    return rays_theta, rays_phi

def create_lidar_single_line(horizontal_resolution=360, horizontal_fov=2*np.pi):
    """创建激光雷达扫描线的角度数组，仅包含水平方向"""
    h_angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, horizontal_resolution)
    v_angles = np.zeros_like(h_angles)
    return h_angles, v_angles
    

def create_demo_scene():
    """创建一个用于测试的mujoco场景，包含所有支持的几何体类型"""
    xml = """
    <mujoco>
        <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <!-- 平面 -->
        <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
        <geom name="plane1" type="plane" size="5 5 0.1" pos="6 0 5" euler="0 -60 0" rgba="0.9 0.9 0.9 1"/>
        <geom name="plane2" type="plane" size="5 3 0.1" pos="-1 -4 3" euler="90 0 0" contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
        
        <!-- 盒子 -->
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" euler="45 -45 0" rgba="1 0 0 1"/>
        <geom name="box2" type="box" size="0.3 0.8 0.2" pos="-2 -1 0.2" rgba="1 0 0 0.7"/>
        <geom name="box3" type="box" size="0.2 0.2 0.3" pos="2.4 -2 0.3" rgba="1 0 0 0.7"/>
        <geom name="box4" type="box" size="0.2 0.2 0.4" pos="0 -2.2 0.4" rgba="1 0 0 0.7"/>
        
        <!-- 球体 -->
        <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
        <geom name="sphere2" type="sphere" size="0.3" pos="-1 2 0.3" rgba="0 1 0 0.7"/>
        
        <!-- 圆柱体 -->
        <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.4" euler="0 90 0" rgba="0 0 1 1"/>
        <geom name="cylinder2" type="cylinder" size="0.2 0.3" pos="2 -2 0.3" rgba="0 0 1 0.7"/>
        
        <!-- 椭球体 -->
        <geom name="ellipsoid1" type="ellipsoid" size="0.4 0.3 0.5" pos="3 2 0.5" rgba="1 1 0 1"/>
        <geom name="ellipsoid2" type="ellipsoid" size="0.2 0.4 0.3" pos="3 -1 0.3" rgba="1 1 0 0.7"/>
        
        <!-- 胶囊体 -->
        <geom name="capsule1" type="capsule" size="0.3 0.5" pos="-3 1 0.8" euler="45 0 0" rgba="1 0 1 1"/>
        <geom name="capsule2" type="capsule" size="0.2 0.4" pos="-3 -2 0.6" euler="45 0 45" rgba="1 0 1 0.7"/>
        
        <!-- 角落放置一组排列的几何体 -->
        <body pos="-3 3 0">
            <geom name="corner_box" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" rgba="0.5 0.5 0.5 1"/>
            <geom name="corner_sphere" type="sphere" size="0.2" pos="0.5 0 0.2" rgba="0.7 0.7 0.7 1"/>
            <geom name="corner_cylinder" type="cylinder" size="0.2 0.2" pos="0 0.5 0.2" rgba="0.6 0.6 0.6 1"/>
            <geom name="corner_capsule" type="capsule" size="0.1 0.3" pos="0.5 0.5 0.3" rgba="0.8 0.8 0.8 1" euler="0 1.57 0"/>
        </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # mujoco.mj_saveLastXML("test.xml", model)
    return model, data

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR传感器演示和性能测试')
    parser.add_argument('--profiling', action='store_true', help='启用性能分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--skip-test', action='store_true', help='跳过性能测试')
    parser.add_argument('--rays', type=int, default=11520, help='默认射线数量(水平x垂直)')
    args = parser.parse_args()
    
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = 'simhei'  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建MuJoCo场景
    mj_model, mj_data = create_demo_scene()
    
    # 创建场景渲染对象
    scene = mujoco.MjvScene(mj_model, maxgeom=100)
    # 更新模拟
    mujoco.mj_forward(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )

    # 创建激光雷达传感器
    lidar = MjLidarSensor(scene, enable_profiling=args.profiling, verbose=args.verbose)
    
    # 创建激光雷达射线 - 使用命令行参数指定的分辨率
    h_res = int(np.sqrt(args.rays) * 2)  # 近似水平分辨率
    v_res = max(1, args.rays // h_res)   # 计算得到的垂直分辨率
    rays_theta, rays_phi = create_lidar_rays(horizontal_resolution=h_res, vertical_resolution=v_res)
    if args.verbose:
        print(f"射线数量: {len(rays_phi)}, 水平分辨率: {h_res}, 垂直分辨率: {v_res}")

    # 设置激光雷达传感器位姿
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 构建激光雷达位姿矩阵
    lidar_pose = np.eye(4, dtype=np.float32)
    lidar_pose[:3, 3] = lidar_position
    
    # 更新模拟
    mujoco.mj_step(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )
    
    # 是否跳过性能测试
    if not args.skip_test and args.profiling:
        print("=" * 50)
        print("性能测试 - 不同射线数量")
        print("=" * 50)
        
        # 测试不同射线数量的性能
        ray_counts = [1000, 5000, 10000, 20000, 50000, 100000]
        prepare_times = []
        kernel_times = []
        update_geom_times = []
        # 添加准备阶段各操作的时间收集列表
        sensor_pose_times = []
        memory_alloc_times = []
        rays_update_times = []
        sync_times = []
        
        for count in ray_counts:
            print(f"\n测试射线数量: {count}")
            
            # 创建测试用的射线
            test_rays_phi, test_rays_theta = create_lidar_rays(
                horizontal_resolution=int(np.sqrt(count) * 10), 
                vertical_resolution=int(np.sqrt(count) / 10)
            )
            test_rays_phi = test_rays_phi[:count]
            test_rays_theta = test_rays_theta[:count]
            
            # 执行多次测试取平均值
            n_tests = 5
            prepare_time_sum = 0
            kernel_time_sum = 0
            update_geom_time_sum = 0
            # 新增准备阶段各操作时间累加变量
            sensor_pose_time_sum = 0
            memory_alloc_time_sum = 0
            rays_update_time_sum = 0
            sync_time_sum = 0
            
            for i in range(n_tests+1):
                # 执行光线追踪
                points = lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, scene)
                ti.sync()
                if i == 0:
                    continue
                
                # 累加时间
                prepare_time_sum += lidar.prepare_time
                kernel_time_sum += lidar.kernel_time
                update_geom_time_sum += lidar.update_geom_time
                # 累加准备阶段各操作时间
                sensor_pose_time_sum += lidar.convert_sensor_pose_time
                memory_alloc_time_sum += lidar.memory_allocation_time
                rays_update_time_sum += lidar.update_rays_time
                sync_time_sum += lidar.sync_time
            
            # 计算平均时间
            avg_prepare_time = prepare_time_sum / n_tests
            avg_kernel_time = kernel_time_sum / n_tests
            avg_update_geom_time = update_geom_time_sum / n_tests
            # 计算准备阶段各操作的平均时间
            avg_sensor_pose_time = sensor_pose_time_sum / n_tests
            avg_memory_alloc_time = memory_alloc_time_sum / n_tests
            avg_rays_update_time = rays_update_time_sum / n_tests
            avg_sync_time = sync_time_sum / n_tests
            
            prepare_times.append(avg_prepare_time)
            kernel_times.append(avg_kernel_time)
            update_geom_times.append(avg_update_geom_time)
            # 保存准备阶段各操作的平均时间
            sensor_pose_times.append(avg_sensor_pose_time)
            memory_alloc_times.append(avg_memory_alloc_time)
            rays_update_times.append(avg_rays_update_time)
            sync_times.append(avg_sync_time)
            
            print(f"平均准备时间: {avg_prepare_time:.2f}ms")
            print(f"平均内核时间: {avg_kernel_time:.2f}ms")
            print(f"平均几何体更新时间: {avg_update_geom_time:.2f}ms")
        
        print("=" * 50)
        print("性能测试 - 不同几何体数量")
        print("=" * 50)
        
        # 测试不同几何体数量的性能
        # 创建包含更多几何体的场景
        num_geoms = [10, 20, 50, 100, 200, 500]
        geom_prepare_times = []
        geom_kernel_times = []
        geom_update_times = []
        # 添加各操作时间的收集列表
        geom_sensor_pose_times = []
        geom_memory_alloc_times = []
        geom_rays_update_times = []
        geom_sync_times = []
        
        # 使用固定数量的射线
        test_rays_phi, test_rays_theta = create_lidar_rays(horizontal_resolution=1800, vertical_resolution=64)
        test_rays_count = len(test_rays_phi)  # 记录测试用的射线数量
        
        for num_geom in num_geoms:
            print(f"\n测试几何体数量: {num_geom}")
            
            # 创建一个包含指定数量几何体的场景
            xml_header = """
            <mujoco>
              <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
            """
            
            xml_footer = """
              </worldbody>
            </mujoco>
            """
            
            xml_content = xml_header
            
            # 添加指定数量的几何体
            for i in range(num_geom):
                geom_type = i % 5  # 0=box, 1=sphere, 2=capsule, 3=ellipsoid, 4=cylinder
                x = (i % 10) - 5
                y = (i // 10) - 5
                z = 0.5
                
                if geom_type == 0:  # box
                    xml_content += f'<geom name="box{i}" type="box" size="0.3 0.3 0.3" pos="{x} {y} {z}" rgba="1 0 0 1"/>\n'
                elif geom_type == 1:  # sphere
                    xml_content += f'<geom name="sphere{i}" type="sphere" size="0.3" pos="{x} {y} {z}" rgba="0 1 0 1"/>\n'
                elif geom_type == 2:  # capsule
                    xml_content += f'<geom name="capsule{i}" type="capsule" size="0.2 0.4" pos="{x} {y} {z}" rgba="1 0 1 1"/>\n'
                elif geom_type == 3:  # ellipsoid
                    xml_content += f'<geom name="ellipsoid{i}" type="ellipsoid" size="0.3 0.2 0.4" pos="{x} {y} {z}" rgba="1 1 0 1"/>\n'
                elif geom_type == 4:  # cylinder
                    xml_content += f'<geom name="cylinder{i}" type="cylinder" size="0.2 0.3" pos="{x} {y} {z}" rgba="0 0 1 1"/>\n'
            
            xml_content += xml_footer
            
            # 创建MuJoCo模型和场景
            test_model = mujoco.MjModel.from_xml_string(xml_content)
            test_data = mujoco.MjData(test_model)
            # mujoco.mj_saveLastXML(f"test_{num_geom}.xml", test_model)
            mujoco.mj_forward(test_model, test_data)
            
            test_scene = mujoco.MjvScene(test_model, maxgeom=max(100, num_geom + 10))
            mujoco.mjv_updateScene(
                test_model, test_data, mujoco.MjvOption(), 
                None, mujoco.MjvCamera(), 
                mujoco.mjtCatBit.mjCAT_ALL.value, test_scene
            )
            
            # 创建新的激光雷达传感器
            test_lidar = MjLidarSensor(test_scene, enable_profiling=args.profiling, verbose=args.verbose)
            
            # 执行多次测试取平均值
            n_tests = 5
            prepare_time_sum = 0
            kernel_time_sum = 0
            update_geom_time_sum = 0
            # 添加各操作时间的累加变量
            sensor_pose_time_sum = 0
            memory_alloc_time_sum = 0
            rays_update_time_sum = 0
            sync_time_sum = 0
            
            for i in range(n_tests+1):
                # 执行光线追踪
                points = test_lidar.ray_cast_taichi(test_rays_phi, test_rays_theta, lidar_pose, test_scene)
                ti.sync()
                if i == 0:
                    continue
                
                # 累加时间
                prepare_time_sum += test_lidar.prepare_time
                kernel_time_sum += test_lidar.kernel_time
                update_geom_time_sum += test_lidar.update_geom_time
                # 累加各操作时间
                sensor_pose_time_sum += test_lidar.convert_sensor_pose_time
                memory_alloc_time_sum += test_lidar.memory_allocation_time
                rays_update_time_sum += test_lidar.update_rays_time
                sync_time_sum += test_lidar.sync_time
            
            # 计算平均时间
            avg_prepare_time = prepare_time_sum / n_tests
            avg_kernel_time = kernel_time_sum / n_tests
            avg_update_geom_time = update_geom_time_sum / n_tests
            # 计算各操作的平均时间
            avg_sensor_pose_time = sensor_pose_time_sum / n_tests
            avg_memory_alloc_time = memory_alloc_time_sum / n_tests
            avg_rays_update_time = rays_update_time_sum / n_tests
            avg_sync_time = sync_time_sum / n_tests
            
            geom_prepare_times.append(avg_prepare_time)
            geom_kernel_times.append(avg_kernel_time)
            geom_update_times.append(avg_update_geom_time)
            # 保存各操作的平均时间
            geom_sensor_pose_times.append(avg_sensor_pose_time)
            geom_memory_alloc_times.append(avg_memory_alloc_time)
            geom_rays_update_times.append(avg_rays_update_time)
            geom_sync_times.append(avg_sync_time)
            
            print(f"平均准备时间: {avg_prepare_time:.2f}ms")
            print(f"平均内核时间: {avg_kernel_time:.2f}ms")
            print(f"平均几何体更新时间: {avg_update_geom_time:.2f}ms")

        
        # 绘制性能结果图表
        plt.figure(figsize=(12, 10))
        
        # 全局字体设置
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        label_font = {'fontsize': 12}
        tick_font = {'fontsize': 10}
        
        # 根据字体可用性选择标签语言
        if 'use_english_labels' in locals() and use_english_labels:
            # 使用英文标签
            ray_count_title = 'Effect of Ray Count on Performance'
            prep_time_title = 'Breakdown of Preparation Time'
            geom_count_title = 'Effect of Geometry Count on Performance'
            geom_update_title = 'Effect of Geometry Count on update_geom_positions'
            
            ray_label = 'Ray Count'
            geom_label = 'Geometry Count'
            time_label = 'Time (ms)'
            
            prep_legend = 'Preparation Time'
            kernel_legend = 'Kernel Time'
            update_legend = 'Geometry Update Time'
            total_legend = 'Total Preparation Time'
            total_time_legend = 'Total Time'
            # 添加新的图例标签
            sensor_pose_legend = 'Sensor Pose Conversion'
            memory_alloc_legend = 'Memory Allocation'
            rays_update_legend = 'Rays Data Update'
            sync_legend = 'Synchronization'
        else:
            # 使用中文标签
            ray_count_title = '射线数量对性能的影响'
            prep_time_title = '准备时间的细分'
            geom_count_title = '几何体数量对性能的影响'
            geom_update_title = '几何体数量对update_geom_positions的影响'
            
            ray_label = '射线数量'
            geom_label = '几何体数量'
            time_label = '时间 (ms)'
            
            prep_legend = '准备时间'
            kernel_legend = '内核时间'
            update_legend = '几何体更新时间'
            total_legend = '总准备时间'
            total_time_legend = '总时间'
            # 添加新的图例标签
            sensor_pose_legend = '传感器位姿转换'
            memory_alloc_legend = '内存分配'
            rays_update_legend = '光线数据更新'
            sync_legend = '同步操作'
        
        # 射线数量对性能的影响
        plt.subplot(2, 2, 1)
        plt.plot(ray_counts, prepare_times, 'o-', label=prep_legend)
        plt.plot(ray_counts, kernel_times, 's-', label=kernel_legend)
        # 添加总时间曲线
        total_times = [p + k for p, k in zip(prepare_times, kernel_times)]
        plt.plot(ray_counts, total_times, '^-', label=total_time_legend)
        plt.xlabel(ray_label, **label_font)
        plt.ylabel(time_label, **label_font)
        plt.title(ray_count_title, **title_font)
        plt.legend(prop={'size': 10})
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 准备时间的细分
        plt.subplot(2, 2, 2)
        # 绘制各个操作的时间曲线，使用不同的标记和颜色
        plt.plot(ray_counts, sensor_pose_times, 'o-', label=sensor_pose_legend, color='purple')
        plt.plot(ray_counts, memory_alloc_times, 's-', label=memory_alloc_legend, color='orange')
        plt.plot(ray_counts, rays_update_times, '^-', label=rays_update_legend, color='green')
        plt.plot(ray_counts, update_geom_times, 'D-', label=update_legend, color='red')
        plt.plot(ray_counts, sync_times, 'x-', label=sync_legend, color='brown')
        plt.plot(ray_counts, prepare_times, '*-', label=total_legend, color='blue', linewidth=2)
        plt.xlabel(ray_label, **label_font)
        plt.ylabel(time_label, **label_font)
        plt.title(prep_time_title, **title_font)
        plt.legend(prop={'size': 9}, loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 几何体数量对性能的影响
        plt.subplot(2, 2, 3)
        plt.plot(num_geoms, geom_prepare_times, 'o-', label=prep_legend)
        plt.plot(num_geoms, geom_kernel_times, 's-', label=kernel_legend)
        # 添加总时间曲线
        geom_total_times = [p + k for p, k in zip(geom_prepare_times, geom_kernel_times)]
        plt.plot(num_geoms, geom_total_times, '^-', label=total_time_legend)
        plt.xlabel(geom_label, **label_font)
        plt.ylabel(time_label, **label_font)
        # 修改标题，添加测试用的雷达点数
        plt.title(f"{geom_count_title} ({test_rays_count} 射线)", **title_font)
        plt.legend(prop={'size': 10})
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        # 几何体数量对update_geom_positions的影响
        plt.subplot(2, 2, 4)
        # 绘制各个操作的时间曲线，使用不同的标记和颜色
        plt.plot(num_geoms, geom_sensor_pose_times, 'o-', label=sensor_pose_legend, color='purple')
        plt.plot(num_geoms, geom_memory_alloc_times, 's-', label=memory_alloc_legend, color='orange')
        plt.plot(num_geoms, geom_rays_update_times, '^-', label=rays_update_legend, color='green')
        plt.plot(num_geoms, geom_update_times, 'D-', label=update_legend, color='red')
        plt.plot(num_geoms, geom_sync_times, 'x-', label=sync_legend, color='brown')
        plt.plot(num_geoms, geom_prepare_times, '*-', label=total_legend, color='blue', linewidth=2)
        plt.xlabel(geom_label, **label_font)
        plt.ylabel(time_label, **label_font)
        # 修改标题，添加测试用的雷达点数
        plt.title(f"{geom_update_title} ({test_rays_count} 射线)", **title_font)
        plt.legend(prop={'size': 9}, loc='upper left')
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig('lidar_performance_analysis.png', dpi=300)
        plt.show()
        
        # 将性能测试结果以表格形式打印
        print("\n" + "=" * 80)
        print("性能测试结果汇总")
        print("=" * 80)
        
        # 表格1：射线数量对性能的影响
        print("\n表格1: 射线数量对性能的影响")
        print("-" * 70)
        print(f"{'射线数量':^12} | {'准备时间 (ms)':^15} | {'内核时间 (ms)':^15} | {'总时间 (ms)':^15}")
        print("-" * 70)
        for i, count in enumerate(ray_counts):
            total_time = prepare_times[i] + kernel_times[i]
            print(f"{count:^12} | {prepare_times[i]:^15.2f} | {kernel_times[i]:^15.2f} | {total_time:^15.2f}")
        print("-" * 70)
        
        # 表格2：准备时间的细分
        print("\n表格2: 准备时间的细分")
        print("-" * 140)
        print(f"{'射线数量':^10} | {'传感器位姿转换':^15} | {'内存分配':^12} | {'光线数据更新':^15} | {'几何体更新':^15} | {'同步操作':^12} | {'总准备时间':^15}")
        print("-" * 140)
        for i, count in enumerate(ray_counts):
            print(f"{count:^10} | {sensor_pose_times[i]:^15.2f} | {memory_alloc_times[i]:^12.2f} | {rays_update_times[i]:^15.2f} | {update_geom_times[i]:^15.2f} | {sync_times[i]:^12.2f} | {prepare_times[i]:^15.2f}")
        print("-" * 140)
    
    # 执行标准光线追踪测试
    print("\n执行标准光线追踪测试:")
    old_enable_profiling = lidar.enable_profiling
    old_verbose = lidar.verbose
    
    # 临时开启性能分析和详细输出
    lidar.enable_profiling = True
    lidar.verbose = True
    
    for _ in range(3):
        start_time = time.time()
        points = lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, scene)
        ti.sync()
        end_time = time.time()
    
    # 恢复原始设置
    lidar.enable_profiling = old_enable_profiling
    lidar.verbose = old_verbose
    
    # 打印性能信息和当前位置
    print(f"耗时: {(end_time - start_time)*1000:.2f} ms, 射线数量: {len(rays_phi)}")
    
    # 三维点云可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')

    # 设置轴标签
    ax.set_xlabel('X轴', fontsize=12)
    ax.set_ylabel('Y轴', fontsize=12)
    ax.set_zlabel('Z轴', fontsize=12)
    ax.set_title('激光雷达三维点云可视化', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('高度 (Z值)', fontsize=12)

    # 设置三轴等比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置网格线
    ax.grid(True)
    
    # 设置背景色为淡灰色，以更好地显示点云
    ax.set_facecolor((0.95, 0.95, 0.95))
    
    fig.tight_layout()
    plt.savefig('lidar_pointcloud.png', dpi=300)
    plt.show()