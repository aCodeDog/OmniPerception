# LiDAR Sensor Integration for IsaacGym

This directory contains complete LiDAR sensor integration for NVIDIA IsaacGym, featuring GPU-accelerated ray tracing with Warp integration and support for multiple Livox sensor types.

## 📁 Directory Structure

```
isaacgym/
├── unitree_g1.py                    # G1 humanoid robot with LiDAR integration
├── unitree_go2.py                   # Go2 quadruped robot with LiDAR integration  
├── utils/
│   └── terrain/
│       ├── terrain.py               # Terrain generation utilities
│       ├── terrain_cfg.py           # Terrain configuration
│       └── __init__.py
├── images/                          # Example result images
├── videos/                          # Demo videos
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install IsaacGym (requires NVIDIA account)
# Download from: https://developer.nvidia.com/isaac-gym
# Follow IsaacGym installation instructions

# Install required dependencies
pip install torch warp-lang numpy matplotlib
```

### Step 1: Install LidarSensor Package

```bash
# Navigate to LidarSensor root directory
cd /path/to/OmniPerception/LidarSensor

# Install in development mode
pip install -e .
```

### Step 2: Run Example Scripts

```bash
# Navigate to IsaacGym examples
cd LidarSensor/LidarSensor/example/isaacgym

# Run Go2 quadruped with LiDAR
python unitree_go2.py

# Run G1 humanoid with LiDAR
python unitree_g1.py
```

## 🔧 Usage Guide

### Basic LiDAR Integration

```python
import torch
import warp as wp
from isaacgym import gymapi, gymtorch
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

# Initialize IsaacGym
gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Initialize Warp AFTER IsaacGym
wp.init()

# Configure LiDAR sensor
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,
    dt=0.02,  # Must match simulation timestep
    max_range=20.0,
    enable_sensor_noise=False,
    update_frequency=50.0
)

# Create LiDAR sensor instance
lidar_sensor = LidarSensor(
    env=env_data,
    env_cfg={'sensor_noise': False},
    sensor_config=sensor_config,
    num_sensors=1,
    device='cuda:0'
)
```

### Environment Integration

```python
# In your main simulation loop
for step in range(max_steps):
    # Step physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # Update robot states
    gym.refresh_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    
    base_pos = root_states[:, 0:3]
    base_quat = root_states[:, 3:7]  # IsaacGym format (xyzw)
    
    # Calculate sensor pose with offset
    sensor_quat = quat_mul(base_quat, sensor_offset_quat.expand(num_envs, -1))
    sensor_pos = base_pos + quat_apply(base_quat, sensor_translation.expand(num_envs, -1))
    
    # CRITICAL: Use [:] assignment to preserve tensor addresses
    lidar_sensor.lidar_positions_tensor[:] = sensor_pos
    lidar_sensor.lidar_quat_tensor[:] = sensor_quat
    
    # Get LiDAR measurements
    point_cloud, distances = lidar_sensor.update()
    
    if point_cloud is not None:
        print(f"Step {step}: Got {point_cloud.shape[1]} points per environment")
```

## ⚙️ Configuration Options

### Supported LiDAR Sensor Types

| Type | FOV (H×V) | Max Rays | Description |
|------|-----------|----------|-------------|
| `MID360` | 360°×59° | 20,000 | 360° coverage, most popular |
| `AVIA` | 70.4°×77.2° | 24,000 | Wide vertical FOV |
| `HORIZON` | 81.7°×25.1° | 24,000 | Automotive grade |
| `HAP` | 81.7°×25.1° | 45,300 | High density scanning |
| `SIMPLE_GRID` | Configurable | Configurable | Basic grid-based pattern |

### LidarConfig Parameters

```python
@dataclass
class LidarConfig:
    # === Core Settings ===
    sensor_type: LidarType = LidarType.MID360
    dt: float = 0.02                    # CRITICAL: Must match sim dt
    num_sensors: int = 1
    update_frequency: float = 50.0
    
    # === Range Settings ===
    max_range: float = 20.0
    min_range: float = 0.2
    
    # === Output Settings ===
    pointcloud_in_world_frame: bool = False
    
    # === Noise Settings ===
    enable_sensor_noise: bool = False
    random_distance_noise: float = 0.03
    pixel_dropout_prob: float = 0.01
    
    # === Placement ===
    nominal_position: list = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg: list = [0, 0, 0]
    randomize_placement: bool = True
```

## 🏗️ Implementation Details

### Key Features

1. **GPU-Accelerated Ray Tracing**
   - Direct CUDA/Warp integration for maximum performance
   - Vectorized operations for thousands of environments
   - No Python loops in critical paths

2. **Realistic Sensor Models**
   - Authentic Livox scan patterns from real hardware
   - Configurable noise simulation and pixel dropout
   - Multiple sensor mounting options

3. **IsaacGym Integration**
   - Seamless integration with IsaacGym's tensor system
   - Proper quaternion handling (xyzw format)
   - Efficient memory management

4. **Flexible Configuration**
   - Simple dataclass-based configuration
   - Runtime parameter updates
   - Debug visualization support

### Architecture

```
LidarSensor (Warp-based)
├── LidarConfig (configuration)
├── Warp Kernels (GPU ray tracing)
├── Mesh Management (collision detection)
└── Sensor Pattern Generation
    ├── Livox patterns (authentic .npy files)
    └── Grid patterns (configurable)
```

## 📊 Performance Benchmarks

### Expected Performance (NVIDIA RTX 4090)

| Environments | Sensor Type | Rays | FPS Without LiDAR | FPS With LiDAR | Overhead |
|-------------|-------------|------|-------------------|----------------|----------|
| 256 | MID360 | 8K | 1200+ | 800-1000 | ~25% |
| 512 | MID360 | 8K | 800+ | 500-600 | ~35% |
| 1024 | MID360 | 8K | 400+ | 250-300 | ~40% |
| 2048 | MID360 | 4K | 200+ | 120-150 | ~40% |

### Optimization Tips

```python
# For maximum performance
sensor_config = LidarConfig(
    sensor_type=LidarType.SIMPLE_GRID,  # Use grid instead of Livox
    max_range=15.0,                     # Limit range
    enable_sensor_noise=False,          # Disable noise
    update_frequency=25.0,              # Lower update rate
    horizontal_line_num=32,             # Reduce resolution
    vertical_line_num=16
)

# For maximum realism
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,       # Use authentic patterns
    max_range=50.0,                     # Full range
    enable_sensor_noise=True,           # Add noise
    random_distance_noise=0.02,         # 2cm noise
    pixel_dropout_prob=0.01,            # 1% dropout
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Tensor Address Changes (CRITICAL)**
   ```python
   # ❌ WRONG - Breaks Warp references
   lidar_sensor.lidar_positions_tensor = new_positions
   
   # ✅ CORRECT - Preserves memory addresses
   lidar_sensor.lidar_positions_tensor[:] = new_positions
   ```

2. **Warp Initialization**
   ```python
   wp.init() 
   ...
   gym = gymapi.acquire_gym()
   ...
   sim = gym.create_sim(...)
   ```

3. **Quaternion Format Issues**
   - IsaacGym uses `xyzw` format (same as Warp)
   - No conversion needed for IsaacGym quaternions
   - Use `quat_mul` and `quat_apply` from IsaacGym torch_utils

4. **Mesh Data Type Issues**
   ```python
   # Ensure correct data types
   faces_wp_array = wp.from_numpy(faces.flatten(), dtype=wp.int32, device=device)
   mesh_ids = wp.array([wp_mesh.id], dtype=wp.uint64, device=device)
   ```

5. **Performance Issues**
   - Use `SIMPLE_GRID` for training, authentic patterns for validation
   - Reduce `max_range` and ray counts for better performance
   - Monitor GPU memory usage with large environment counts

### Debug Commands

```python
# Enable debug visualization
sensor_config = LidarConfig(
    debug_vis=True,  # Shows rays in viewer
    # ... other config
)

# Check tensor shapes and values
print(f"Sensor positions: {lidar_sensor.lidar_positions_tensor.shape}")
print(f"Sensor quaternions: {lidar_sensor.lidar_quat_tensor.shape}")
print(f"Distance range: {distances.min():.2f} - {distances.max():.2f}")
```

## 📝 File Descriptions

### Example Scripts

- **`unitree_go2.py`**: Complete integration example with Unitree Go2 quadruped robot
  - Terrain generation and navigation
  - LiDAR sensor mounting and configuration
  - Real-time visualization and data processing
  - Performance benchmarking options

- **`unitree_g1.py`**: Complete integration example with Unitree G1 humanoid robot
  - Bipedal locomotion with LiDAR perception
  - Multi-sensor configuration examples
  - Advanced noise simulation and realistic settings
  - Integration with RL training frameworks

### Utility Files

- **`utils/terrain/terrain.py`**: Procedural terrain generation for testing
- **`utils/terrain/terrain_cfg.py`**: Terrain configuration parameters
- **`images/`**: Example output images and visualizations
- **`videos/`**: Demo videos showing LiDAR integration

## 🎯 Next Steps

1. **Run Examples**: Execute the provided scripts to verify installation
2. **Experiment**: Try different sensor configurations and robot platforms
3. **Integrate**: Add LiDAR to your existing IsaacGym environments
4. **Optimize**: Tune performance parameters for your specific use case
5. **Develop**: Create custom applications using the LiDAR integration

## 🔗 Related Links

- [IsaacGym Documentation](https://developer.nvidia.com/isaac-gym)
- [Warp Documentation](https://nvidia.github.io/warp/)
- [Main OmniPerception README](../../README.md)
- [LidarSensor Source Code](../../lidar_sensor.py)

For questions or issues, please refer to the main repository documentation or create a GitHub issue. 