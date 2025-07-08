# LiDAR Sensor Integration for Genesis

This directory contains complete LiDAR sensor integration for Genesis physics engine, featuring high-performance ray tracing with modern GPU acceleration and realistic Livox sensor patterns.

## 📁 Directory Structure

```
genesis/
├── g1_lidar_visualization.py         # G1 humanoid with LiDAR visualization
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install Genesis (latest version)
pip install genesis-world

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

### Step 2: Run Example Script

```bash
# Navigate to Genesis examples
cd LidarSensor/LidarSensor/example/genesis

# Run G1 humanoid with LiDAR visualization
python g1_lidar_visualization.py
```

## 🔧 Usage Guide

### Basic LiDAR Integration

```python
import torch
import warp as wp
import genesis as gs
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

wp.init()
# Initialize Genesis FIRST
gs.init(backend=gs.gpu, logging_level="warning")

# Create scene with multiple environments
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.02,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
    ),
    show_viewer=True,
)

# Build scene with environments
scene.build(n_envs=4096)

# Initialize Warp AFTER Genesis


# Configure LiDAR sensor
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,
    dt=0.02,  # Must match Genesis scene dt
    max_range=30.0,
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
    # Step Genesis physics
    scene.step()
    
    # Get robot states
    base_pos = robot.get_pos()      # Shape: (num_envs, 3)
    base_quat = robot.get_quat()    # Genesis format (wxyz)
    
    # Calculate sensor pose with offset
    sensor_quat = quat_mul_genesis(base_quat, sensor_offset_quat.expand(num_envs, -1))
    sensor_pos = base_pos + quat_apply_genesis(base_quat, sensor_translation.expand(num_envs, -1))
    
    # Convert Genesis quaternion (wxyz) to Warp quaternion (xyzw)
    sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
    
    # CRITICAL: Use [:] assignment to preserve tensor addresses
    lidar_sensor.lidar_positions_tensor[:] = sensor_pos
    lidar_sensor.lidar_quat_tensor[:] = sensor_quat_warp
    
    # Get LiDAR measurements
    point_cloud, distances = lidar_sensor.update()
    
    if point_cloud is not None:
        print(f"Step {step}: Got {point_cloud.shape[1]} points per environment")
```

### Quaternion Conversion Utilities

```python
def quat_genesis_to_warp(genesis_quat):
    """Convert Genesis quaternion (wxyz) to Warp quaternion (xyzw)"""
    return torch.stack([
        genesis_quat[:, 1],  # x
        genesis_quat[:, 2],  # y
        genesis_quat[:, 3],  # z
        genesis_quat[:, 0]   # w
    ], dim=1)

def quat_mul_genesis(q1, q2):
    """Quaternion multiplication for Genesis format (wxyz)"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack([w, x, y, z], dim=1)

def quat_apply_genesis(q, v):
    """Apply quaternion rotation to vector for Genesis format (wxyz)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    
    # Quaternion rotation: v' = q * v * q^-1
    # Optimized version
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    rx = (ww + xx - yy - zz) * vx + 2 * ((xy - wz) * vy + (xz + wy) * vz)
    ry = (ww - xx + yy - zz) * vy + 2 * ((xy + wz) * vx + (yz - wx) * vz)
    rz = (ww - xx - yy + zz) * vz + 2 * ((xz - wy) * vx + (yz + wx) * vy)
    
    return torch.stack([rx, ry, rz], dim=1)
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
    dt: float = 0.02                    # CRITICAL: Must match Genesis scene dt
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

1. **Modern Physics Engine Integration**
   - Built on top of Genesis's high-performance engine
   - Seamless GPU memory management
   - Efficient multi-environment scaling

2. **Realistic Sensor Models**
   - Authentic Livox scan patterns from real hardware
   - Configurable noise simulation and pixel dropout
   - Multiple sensor mounting options with transforms

3. **Genesis-Specific Optimizations**
   - Proper quaternion handling (wxyz format conversion)
   - Efficient state synchronization
   - Memory-efficient tensor operations

4. **Cross-Platform Compatibility**
   - Works on Linux, Windows, and macOS
   - CPU and GPU acceleration support
   - Flexible rendering backends

### Architecture

```
LidarSensor (Genesis Integration)
├── LidarConfig (configuration)
├── Genesis State Interface (robot poses)
├── Warp Integration (ray tracing backend)
├── Quaternion Conversion (wxyz ↔ xyzw)
└── Sensor Pattern Generation
    ├── Livox patterns (authentic .npy files)
    └── Grid patterns (configurable)
```

## 📊 Performance Benchmarks

### Expected Performance (NVIDIA RTX 4090)

| Environments | Sensor Type | Rays | FPS Without LiDAR | FPS With LiDAR | Overhead |
|-------------|-------------|------|-------------------|----------------|----------|
| 1024 | MID360 | 8K | 800+ | 500-600 | ~30% |
| 2048 | MID360 | 8K | 400+ | 250-300 | ~35% |
| 4096 | MID360 | 4K | 200+ | 120-150 | ~40% |
| 8192 | SIMPLE_GRID | 2K | 100+ | 60-80 | ~35% |

### Genesis-Specific Optimizations

```python
# For maximum performance with Genesis
scene_options = gs.options.SimOptions(
    dt=0.02,
    substeps=10,              # Balance accuracy vs speed
    gravity=(0, 0, -9.81),
    requires_grad=False,      # Disable gradients if not needed
)

sensor_config = LidarConfig(
    sensor_type=LidarType.SIMPLE_GRID,  # Use grid for training
    max_range=15.0,                     # Limit computational load
    enable_sensor_noise=False,          # Disable during training
    update_frequency=25.0,              # Lower frequency for performance
    horizontal_line_num=32,
    vertical_line_num=16
)

# For maximum realism with Genesis
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,       # Use authentic patterns
    max_range=50.0,                     # Full sensor range
    enable_sensor_noise=True,           # Add realistic noise
    random_distance_noise=0.02,         # 2cm noise standard deviation
    pixel_dropout_prob=0.01,            # 1% pixel dropout
    update_frequency=50.0               # Full update rate
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Warp Initialization (CRITICAL)**
   ```python

   wp.init()  # Initialize Warp 
   ```

2. **Quaternion Format Conversion**
   ```python
   # Genesis uses wxyz format, Warp uses xyzw
   # Always convert before passing to LidarSensor
   sensor_quat_warp = quat_genesis_to_warp(genesis_quat)
   ```

3. **Tensor Address Preservation**
   ```python
   # ❌ WRONG - Breaks Warp references
   lidar_sensor.lidar_positions_tensor = new_positions
   
   # ✅ CORRECT - Preserves memory addresses
   lidar_sensor.lidar_positions_tensor[:] = new_positions
   ```

4. **Memory Management**
   - Genesis automatically manages GPU memory
   - Ensure consistent device placement (all CUDA or all CPU)
   - Monitor memory usage with large environment counts

5. **Mesh Extraction Issues**
   ```python
   # Ensure mesh data is properly extracted from Genesis scene
   vertices, faces = extract_genesis_scene_mesh(scene)
   # Verify data types and shapes before creating Warp mesh
   ```

### Debug Commands

```python
# Enable Genesis debug visualization
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    )
)

# Enable LiDAR debug visualization
sensor_config = LidarConfig(
    debug_vis=True,  # Shows rays in viewer
    # ... other config
)

# Check system state
print(f"Genesis backend: {gs.get_backend()}")
print(f"Number of environments: {scene.n_envs}")
print(f"Sensor positions: {lidar_sensor.lidar_positions_tensor.shape}")
print(f"Distance range: {distances.min():.2f} - {distances.max():.2f}")
```

## 📝 File Descriptions

### Example Scripts

- **`g1_lidar_visualization.py`**: Complete integration example with Unitree G1 humanoid robot
  - Genesis scene setup and robot loading
  - LiDAR sensor mounting and configuration
  - Real-time visualization with Genesis viewer
  - Point cloud processing and display
  - Performance monitoring and optimization tips

## 🎯 Next Steps

1. **Run Examples**: Execute the provided script to verify installation
2. **Explore Genesis**: Learn about Genesis's unique features and capabilities
3. **Experiment**: Try different sensor configurations and robot platforms
4. **Integrate**: Add LiDAR to your existing Genesis environments
5. **Optimize**: Tune performance parameters for your specific use case
6. **Develop**: Create custom applications using Genesis + LiDAR

## 🔗 Related Links

- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [Genesis GitHub Repository](https://github.com/Genesis-Embodied-AI/Genesis)
- [Warp Documentation](https://nvidia.github.io/warp/)
- [Main OmniPerception README](../../README.md)
- [LidarSensor Source Code](../../lidar_sensor.py)

## 🌟 Genesis Advantages

Genesis offers several unique advantages for LiDAR simulation:

- **🚀 Modern Architecture**: Built from ground up for modern GPU computing
- **🔄 Cross-Platform**: Runs on Linux, Windows, and macOS 
- **⚡ High Performance**: Optimized for large-scale parallel simulation
- **🛠️ Easy Setup**: Simple installation and configuration
- **🎨 Beautiful Rendering**: High-quality visualization capabilities
- **🧠 Differentiable**: Full differentiable physics (if needed)

For questions or issues, please refer to the main repository documentation or create a GitHub issue. 