import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))


class LidarWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        local_dist: wp.array(dtype=wp.float32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[0]
        lidar_position = lidar_pos_array[env_id, cam_id]
        # if env_id == 1 :
        #     wp.print(lidar_position)
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        # perturb ray_vectors with uniform noise
        ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        query = wp.mesh_query_ray(mesh,ray_origin, ray_direction_world, far_plane)
        if query.result:
            dist = query.t
            local_dist[env_id, cam_id, scan_line, point_index] = dist
            if pointcloud_in_world_frame:
                pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
            else:
                pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir

    @staticmethod
    @wp.kernel
    def draw_height_scanner_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_origins: wp.array(dtype=wp.vec3, ndim=4),  # Different origins per ray
        ray_directions: wp.array(dtype=wp.vec3, ndim=4),  # Direction per ray
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        local_dist: wp.array(dtype=wp.float32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):
        """Height scanner kernel with different ray origins"""
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[0]
        
        # Get sensor position and orientation
        sensor_position = lidar_pos_array[env_id, cam_id]
        sensor_quaternion = lidar_quat_array[env_id, cam_id]
        
        # Get ray origin and direction for this specific ray
        ray_origin_local = ray_origins[env_id, cam_id, scan_line, point_index]
        ray_dir_local = ray_directions[env_id, cam_id, scan_line, point_index]
        
        # For height scanner: Apply yaw-only rotation (following Isaac Lab's approach)
        # Extract yaw angle from quaternion and create new yaw-only quaternion
        
        # Read quaternion components (sensor_quaternion is in warp format: x,y,z,w)
        qw = sensor_quaternion[3]  # w component 
        qx = sensor_quaternion[0]  # x component  
        qy = sensor_quaternion[1]  # y component
        qz = sensor_quaternion[2]  # z component

        # Extract yaw angle using atan2 (following Isaac Lab's yaw_quat function)
        yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Create yaw-only quaternion from yaw angle
        yaw_half = yaw / 2.0
        yaw_only_quat = wp.quat(0.0, 0.0, wp.sin(yaw_half), wp.cos(yaw_half))
        
        # Apply yaw-only transformation (following Isaac Lab's ray_alignment="yaw" approach)
        # Transform ray origins: ray_starts_w = quat_apply_yaw(quat_w.repeat(1, num_rays), ray_starts) + pos_w
        ray_origin_rotated = wp.quat_rotate(yaw_only_quat, ray_origin_local)
        ray_origin_world = ray_origin_rotated + sensor_position

        # Ray directions remain unchanged for height scanner (downward direction)
        ray_direction_world = ray_dir_local
        
        # Perform ray casting
        dist = NO_HIT_RAY_VAL
        query = wp.mesh_query_ray(mesh, ray_origin_world, ray_direction_world, far_plane)
        if query.result:
            dist = query.t
            local_dist[env_id, cam_id, scan_line, point_index] = dist
            if pointcloud_in_world_frame:
                # Return the actual hit point in world coordinates
                pixels[env_id, cam_id, scan_line, point_index] = ray_origin_world + dist * ray_direction_world
            else:
                # For local coordinates, return hit point relative to sensor
                hit_point_world = ray_origin_world + dist * ray_direction_world
                pixels[env_id, cam_id, scan_line, point_index] = wp.quat_rotate(wp.quat_inverse(sensor_quaternion), hit_point_world - sensor_position)

