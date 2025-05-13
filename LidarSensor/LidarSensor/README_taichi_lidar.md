# Taichi-based LiDAR Sensor

This implementation provides a high-performance LiDAR sensor using Taichi, a high-performance programming language designed for parallel computing. The implementation focuses on efficiently handling ray-triangle intersections for accurate LiDAR simulation while maintaining high performance through parallel GPU execution.

## Features

- **High-performance ray-triangle intersection**: Leverages Taichi's parallel computing capabilities for efficient LiDAR simulation
- **BVH acceleration structure**: Uses a Bounding Volume Hierarchy to accelerate ray-triangle intersection tests
- **GPU acceleration**: Takes advantage of GPU parallelism to handle millions of rays per second
- **Batch processing**: Supports processing multiple environments in parallel
- **World/local frame output**: Can return point clouds in either world or sensor local frame
- **Integration with existing infrastructure**: Compatible with the existing sensor framework

## Implementation Details

### Möller-Trumbore Algorithm

The core of our LiDAR sensor uses the Möller-Trumbore algorithm for ray-triangle intersection, which is mathematically elegant and efficient. This algorithm directly computes the barycentric coordinates of the intersection point without explicitly computing the plane equation of the triangle.

### BVH Acceleration

To avoid checking every ray against every triangle (which would be O(n*m) complexity), we implemented a Bounding Volume Hierarchy (BVH) acceleration structure. This enables O(log(n)) complexity for each ray, dramatically improving performance for complex meshes.

### Parallelization Strategy

Our implementation parallelizes ray tracing in several key ways:
1. **Ray generation**: All rays are generated in parallel
2. **Ray tracing**: Each ray is traced independently
3. **BVH traversal**: Each ray traverses the BVH hierarchy independently
4. **Post-processing**: Coordinate transformations are done in parallel

## Performance Comparison

This implementation can achieve significantly higher performance compared to the Warp-based implementation, especially for complex scenes and large numbers of rays. Our benchmarks show:

- **Mesh complexity scaling**: Near-linear scaling with mesh complexity due to BVH acceleration
- **Ray count scaling**: Excellent scaling with increased ray counts due to efficient parallelization
- **Memory usage**: Efficient memory usage even for large meshes

## Usage

### Basic Usage

```python
from sensor.taichi_lidar_sensor import TaichiLidarSensor
from sensor.taichi_lidar_example import LidarConfig, create_complex_mesh

# Create sensor configuration
lidar_config = LidarConfig()
lidar_config.vertical_line_num = 64
lidar_config.horizontal_line_num = 512
lidar_config.horizontal_fov_deg_min = -45.0
lidar_config.horizontal_fov_deg_max = 45.0
lidar_config.vertical_fov_deg_min = -15.0
lidar_config.vertical_fov_deg_max = 15.0
lidar_config.max_range = 100.0
lidar_config.pointcloud_in_world_frame = True

# Create environment data
env = {
    'num_envs': batch_size,
    'sensor_pos_tensor': sensor_positions,  # Tensor of shape [batch_size, num_sensors, 3]
    'sensor_quat_tensor': sensor_orientations,  # Tensor of shape [batch_size, num_sensors, 4]
    'mesh_ids': None  # Not used in Taichi implementation
}

# Create LiDAR sensor
lidar_sensor = TaichiLidarSensor(
    env=env,
    env_cfg=env_cfg,
    sensor_config=lidar_config,
    num_sensor=1,
    device='cuda:0'
)

# Set mesh data
vertices, indices = load_mesh_data()  # Your function to load mesh data
lidar_sensor.set_mesh_data(vertices, indices)

# Perform scan
point_cloud, distances = lidar_sensor.update()
```

### Running Benchmarks

You can benchmark the Taichi LiDAR implementation against the Warp implementation:

```bash
python -m sensor.benchmark_lidar --batch-size 1 --num-triangles 10000 --visualize
```

## Implementation Files

1. `taichi_lidar_sensor.py` - Main implementation of the Taichi-based LiDAR sensor
2. `taichi_lidar_example.py` - Example usage and utilities
3. `benchmark_lidar.py` - Benchmark scripts to compare with Warp implementation

## Requirements

- Taichi (1.4.0 or higher)
- PyTorch (1.8.0 or higher)
- NumPy
- Matplotlib (for visualization)
- CUDA-capable GPU (for GPU acceleration)

## Future Improvements

1. **Custom BVH construction**: Implement more advanced BVH construction techniques for even better performance
2. **Memory optimization**: Further reduce memory usage for large scenes
3. **Multi-bounce simulation**: Extend to support multiple bounces for more realistic simulation
4. **Material properties**: Add support for different material properties affecting the sensor readings
5. **Mixed precision**: Experiment with mixed precision for better performance

## References

1. Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection.
2. Karras, T. (2012). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees.
3. Taichi documentation: https://docs.taichi.graphics/
4. NVIDIA CUDA documentation: https://docs.nvidia.com/cuda/ 