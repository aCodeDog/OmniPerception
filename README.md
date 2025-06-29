<div align="center">

# 🌟 **OmniPerception**

*High-Performance Multi-Modal Sensor Simulation for Robotics*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

**Accelerated sensor simulation with GPU-optimized lidar sensors for large-scale robotics environments**

[🚀 Quick Start](#-getting-started) • [📚 Documentation](#-integration-guide) • [🎯 Examples](#-quick-usage) • [🐛 Issues](https://github.com/aCodeDog/OmniPerception/issues)

</div>

---

## 📖 **Table of Contents**

- [🌟 Project Overview](#-project-status)
- [⚡ Performance](#-performance-highlights)  
- [🔧 Supported Hardware](#-supported-hardware)
- [🎨 Visualization](#-visualization-examples)
- [🚀 Getting Started](#-getting-started)
- [📚 Integration Guide](#-integration-guide)
- [🤝 Contributing](#-acknowledge)

---

## 📋 **Project Status**

| Component | Status | Description |
|-----------|---------|-------------|
| 🎯 **LidarSensor** | 🚧 **Released[Half]** | High-performance GPU-accelerated lidar simulation |
| 🤖 **Training Code** | 🚧 **In Progress** | Reinforcement learning integration |
| 🚀 **Deploy Code** | 📅 **Planned** | Production deployment utilities |

## ⚡ **Performance Highlights**

- **🔥 Ultra-Fast Rendering**: 250ms per step for 4,096 environments with 20,000 rays each (RTX 4090)
- **🎯 Multi-Sensor Support**: 11+ lidar sensor types including Livox and Velodyne series
- **🌐 Multi-Platform**: Supports IsaacGym, Genesis, Mujoco, and Isaac Sim
- **⚡ GPU Acceleration**: CUDA-optimized ray tracing with Warp backend

## ✨ **Key Features**

<div align="center">

| 🎯 **Precision** | ⚡ **Performance** | 🔧 **Flexibility** | 🌐 **Integration** |
|:---:|:---:|:---:|:---:|
| Realistic sensor modeling | GPU-accelerated computation | 11+ sensor types | multi-simulation platforms |
| Self-occlusion support | 250ms/4K environments | Custom configurations | Easy API integration |
| Noise simulation | CUDA optimization | Pattern-based scanning | Multi-robot support |

</div>

## 🔧 **Supported Hardware**

### **Lidar Sensors**
<table>
<tr>
<td>

**Livox Series**
- Mid-360 🟢
- Avia 🟢  
- Horizon 🟢
- HAP 🟢
- Mid-40/70 🟢
- Tele 🟢

</td>
<td>

**Traditional Spinning**
- Velodyne HDL-64 🟢
- Velodyne VLP-32 🟢
- Ouster OS-128 🟢
- Custom Grid Patterns 🟢

</td>
</tr>
</table>

### **Simulation Platforms**
- **IsaacGym** - NVIDIA's physics simulation platform
- **Genesis** - High-performance physics engine  
- **Mujoco** - Advanced physics simulation
- **Isaac Sim** - Omniverse-based robotics simulation

## 🎨 **Visualization Examples**

### **Real-time Lidar Simulation**

<div align="center">

**Livox Mid-360 on Unitree G1 Robot**
![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/Mid360_g1.gif)

**Environment Scanning with Obstacle Detection**
![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/mid360_no_shelf.gif)

*Features: Self-occlusion modeling, real-time point cloud generation, multi-environment support*

</div>

---

## 💡 **Development Status**

> **📢 Important Note**: This project is under active development. While the LidarSensor module is fully functional and optimized, we're continuously improving documentation and code structure. For any issues or questions, please [open an issue](https://github.com/aCodeDog/OmniPerception/issues).

---

# 🚀 **Getting Started**

## 📦 **Installation**

### **Prerequisites**
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- One of: IsaacGym, Genesis, Mujoco, or Isaac Sim

### **Quick Install**
```bash
# Install core dependencies
pip install warp-lang[extras] taichi

# Install LidarSensor
cd LidarSensor
pip install -e .
```

### **Optional Dependencies**
```bash
# For ROS integration (optional)
source /opt/ros/humble/setup.bash

# For advanced visualization (optional)  
pip install matplotlib open3d
```

## 🎯 **Quick Usage**

### **1. Basic Example - IsaacGym**
```bash
# Generate self-occlusion mesh (first time only)
cd LidarSensor/resources/robots/g1_29/
python process_body_mesh.py

# Run example with Unitree G1
cd LidarSensor/example/isaacgym
python unitree_g1.py
```

### **2. ROS Integration Example**
```bash
# Start ROS visualization
source /opt/ros/humble/setup.bash
/usr/bin/python3 LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/lidar_vis_ros2.py
```

### **3. Custom Configuration**
```python
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

# Create custom sensor configuration
config = LidarConfig(
    sensor_type=LidarType.MID360,
    max_range=30.0,
    enable_sensor_noise=False
)
```


# **Integration Guide**

> **🔧 RECENT FIX:** The `LidarConfig` class has been converted to a proper dataclass. You can now use parameterized instantiation as shown in the examples below.

## **LidarSensor Integration - Unified Steps for All Simulators**

### **Core Integration Steps (Same for Genesis & IsaacGym)**

#### **Step 1: Create Warp Mesh, Get Sensor Position/Quat, and num_envs**

**For Genesis:**
```python
import genesis as gs
import warp as wp
import torch
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

# Initialize Genesis FIRST
gs.init(logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.02, ...))
# Add your robots and environment
scene.build(n_envs=num_envs)

# Initialize Warp AFTER Genesis
wp.init()

# Get simulation variables
num_envs = 4096  # From your Genesis scene
device = 'cuda:0'  # or 'cpu'
sim_dt = 0.02  # From Genesis scene dt

# Extract mesh data and create Warp mesh
vertices, faces = extract_genesis_scene_mesh()  # Your implementation
vertex_tensor = torch.tensor(vertices, device=device, dtype=torch.float32)
vertex_vec3_array = wp.from_torch(vertex_tensor, dtype=wp.vec3)
faces_wp_array = wp.from_numpy(faces.flatten(), dtype=wp.int32, device=device)
wp_mesh = wp.Mesh(points=vertex_vec3_array, indices=faces_wp_array)
mesh_ids = wp.array([wp_mesh.id], dtype=wp.uint64, device=device)

# Get robot states and calculate sensor pose
base_pos = robot.get_pos()  # Shape: (num_envs, 3)
base_quat = robot.get_quat()  # Shape: (num_envs, 4) - Genesis format (wxyz)

# Define sensor offset (relative to robot base)
sensor_translation = torch.tensor([0.1, 0.0, 0.436], device=device)  # x, y, z offset
sensor_offset_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # wxyz format

# Calculate sensor pose with offset
sensor_quat = quat_mul_genesis(base_quat, sensor_offset_quat.expand(num_envs, -1))
sensor_pos = base_pos + quat_apply_genesis(base_quat, sensor_translation.expand(num_envs, -1))

# Convert Genesis quaternion (wxyz) to Warp quaternion (xyzw)
def quat_genesis_to_warp(genesis_quat):
    return torch.stack([genesis_quat[:, 1], genesis_quat[:, 2], 
                       genesis_quat[:, 3], genesis_quat[:, 0]], dim=1)

sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
```

**For IsaacGym:**
```python
from isaacgym import gymapi, gymtorch
import torch
import warp as wp
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

# Initialize IsaacGym
gym = gymapi.acquire_gym()
sim = gym.create_sim(device_id=0, graphics_device_id=0, ...)
# Create environments and robots
envs = [gym.create_env(sim, ...) for _ in range(num_envs)]

# Initialize Warp AFTER IsaacGym
wp.init()

# Get simulation variables
num_envs = len(envs)
device = 'cuda:0'
sim_dt = gym.get_sim_time_step(sim)

# Extract mesh data and create Warp mesh
vertices, faces, mesh_ids = extract_isaacgym_mesh_data(gym, sim, envs)
vertex_tensor = torch.tensor(vertices, device=device, dtype=torch.float32)
vertex_vec3_array = wp.from_torch(vertex_tensor, dtype=wp.vec3)
faces_wp_array = wp.from_numpy(faces.flatten(), dtype=wp.int32, device=device)
wp_mesh = wp.Mesh(points=vertex_vec3_array, indices=faces_wp_array)

# Get robot states
gym.refresh_actor_root_state_tensor(sim)
_root_states = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(_root_states)

base_pos = root_states[:, 0:3]  # Shape: (num_envs, 3)
base_quat = root_states[:, 3:7]  # Shape: (num_envs, 4) - IsaacGym format (xyzw)

# Define sensor offset (relative to robot base)
sensor_translation = torch.tensor([0.1, 0.0, 0.436], device=device)
sensor_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # xyzw format

# Calculate sensor pose with offset (IsaacGym uses xyzw format - same as Warp)
from isaacgym.torch_utils import quat_mul, quat_apply
sensor_quat = quat_mul(base_quat, sensor_offset_quat.expand(num_envs, -1))
sensor_pos = base_pos + quat_apply(base_quat, sensor_translation.expand(num_envs, -1))

sensor_quat_warp = sensor_quat  # Already in correct format
```

#### **Step 2: Create LidarSensor**

> **✅ FIXED:** LidarConfig now supports parameterized instantiation! You can pass arguments to the constructor as shown below.

**Same for Both Simulators:**
```python
# Create LidarSensor configuration - Basic Example
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,  # Choose your sensor type
    dt=sim_dt,  # CRITICAL: Must match simulation dt
    max_range=20.0,
    update_frequency=1.0/sim_dt,  # Update every simulation step
    return_pointcloud=True,
    pointcloud_in_world_frame=False,  # Get local coordinates first
    enable_sensor_noise=False,  # Disable for faster processing
)

# Alternative: Use default values (same as empty constructor)
# sensor_config = LidarConfig()  # Uses all default values

# Advanced Example: Full Configuration
sensor_config_advanced = LidarConfig(
    # Core settings
    sensor_type=LidarType.AVIA,
    dt=0.02,
    update_frequency=50.0,
    
    # Range settings
    max_range=30.0,
    min_range=0.1,
    
    # Grid lidar settings (only for SIMPLE_GRID type)
    horizontal_line_num=64,
    vertical_line_num=32,
    horizontal_fov_deg_min=-180,
    horizontal_fov_deg_max=180,
    vertical_fov_deg_min=-15,
    vertical_fov_deg_max=15,
    
    # Output settings
    pointcloud_in_world_frame=True,  # World coordinates
    
    # Noise settings
    enable_sensor_noise=True,
    random_distance_noise=0.05,
    pixel_dropout_prob=0.02,
    
    # Placement settings
    randomize_placement=False,
    nominal_position=[0.15, 0.0, 0.5],
    
    # Data processing
    normalize_range=True,
)

# Create environment data dictionary
env_data = {
    'num_envs': num_envs,
    'sensor_pos_tensor': sensor_pos,      # Shape: (num_envs, 3)
    'sensor_quat_tensor': sensor_quat_warp,  # Shape: (num_envs, 4) - xyzw format
    'vertices': vertices,
    'faces': faces, 
    'mesh_ids': mesh_ids  # Essential for collision detection
}

# Create LidarSensor instance
lidar_sensor = LidarSensor(
    env=env_data,
    env_cfg={'sensor_noise': False},
    sensor_config=sensor_config,
    num_sensors=1,
    device=device
)

print(f"✓ LidarSensor created: {num_envs} environments, {sensor_config.sensor_type.value} sensor")

# ⚠️ IMPORTANT: For ALL future tensor updates, use slice assignment [:]
# lidar_sensor.lidar_positions_tensor[:] = new_positions  # ✅ Correct
# lidar_sensor.lidar_positions_tensor = new_positions     # ❌ Wrong
```

#### **Step 3: Update in Main Loop**

> **⚠️ CRITICAL: Tensor Address Immutability**
> 
> **NEVER change tensor addresses/references!** Always use slice assignment to update tensor values:
> ```python
> # ✅ CORRECT - Preserves tensor address
> self.lidar_sensor.lidar_positions_tensor[:] = new_sensor_positions
> self.lidar_sensor.lidar_quat_tensor[:] = new_sensor_quaternions
> 
> # ❌ WRONG - Changes tensor address, breaks Warp references
> self.lidar_sensor.lidar_positions_tensor = new_sensor_positions
> self.lidar_sensor.lidar_quat_tensor = new_sensor_quaternions
> ```
> 
> **Why this matters:**
> - Warp maintains internal references to tensor memory addresses
> - Changing the tensor reference breaks these internal connections
> - Will cause crashes, incorrect data, or silent failures
> - **Always use `tensor[:] = new_values`**, never `tensor = new_values`

**For Genesis:**
```python
# Main simulation loop
for step in range(max_steps):
    # Step Genesis physics
    scene.step()
    
    # Get updated robot states
    base_pos = robot.get_pos()
    base_quat = robot.get_quat()  # Genesis format (wxyz)
    
    # Calculate sensor pose with offset
    sensor_quat = quat_mul_genesis(base_quat, sensor_offset_quat.expand(num_envs, -1))
    sensor_pos = base_pos + quat_apply_genesis(base_quat, sensor_translation.expand(num_envs, -1))
    sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
    
    # Update sensor pose - CRITICAL: Use [:] assignment to preserve tensor addresses
    lidar_sensor.lidar_positions_tensor[:] = sensor_pos
    lidar_sensor.lidar_quat_tensor[:] = sensor_quat_warp
    
    # Get lidar measurements
    point_cloud, distances = lidar_sensor.update()
    
    # Process lidar data
    if point_cloud is not None:
        # point_cloud shape: (num_envs, num_points, 3)
        # distances shape: (num_envs, num_points)
        print(f"Step {step}: Got {point_cloud.shape[1]} points per environment")
```

**For IsaacGym:**
```python
# Main simulation loop
for step in range(max_steps):
    # Step IsaacGym physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # Get updated robot states - CRITICAL: Refresh tensors first
    gym.refresh_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    
    base_pos = root_states[:, 0:3]
    base_quat = root_states[:, 3:7]  # IsaacGym format (xyzw)
    
    # Calculate sensor pose with offset
    sensor_quat = quat_mul(base_quat, sensor_offset_quat.expand(num_envs, -1))
    sensor_pos = base_pos + quat_apply(base_quat, sensor_translation.expand(num_envs, -1))
    
    # Update sensor pose - CRITICAL: Use [:] assignment to preserve tensor addresses
    lidar_sensor.lidar_positions_tensor[:] = sensor_pos
    lidar_sensor.lidar_quat_tensor[:] = sensor_quat
    
    # Get lidar measurements
    point_cloud, distances = lidar_sensor.update()
    
    # Process lidar data
    if point_cloud is not None:
        print(f"Step {step}: Got {point_cloud.shape[1]} points per environment")
```

---


### **Supported Sensor Types**

```python
# Available sensor types
LidarType.SIMPLE_GRID    # Basic grid-based scanning
LidarType.MID360         # Livox Mid-360 (most common)
LidarType.AVIA           # Livox Avia (high performance)
LidarType.HORIZON        # Livox Horizon
LidarType.HAP            # Livox HAP
LidarType.HDL64          # Velodyne HDL-64
LidarType.VLP32          # Velodyne VLP-32
LidarType.OS128          # Ouster OS-128
```

### **Configuration Parameters Reference**

#### **Available LidarConfig Parameters**

```python
@dataclass
class LidarConfig:
    # === Core Sensor Settings ===
    sensor_type: LidarType = LidarType.MID360      # Sensor type (see supported types above)
    num_sensors: int = 1                           # Number of sensor instances
    dt: float = 0.02                               # Simulation timestep (MUST match sim dt)
    update_frequency: float = 50.0                 # Sensor update rate in Hz
    
    # === Range Settings ===
    max_range: float = 20.0                        # Maximum detection range (meters)
    min_range: float = 0.2                         # Minimum detection range (meters)
    
    # === Grid Lidar Settings (SIMPLE_GRID only) ===
    horizontal_line_num: int = 80                  # Number of horizontal scan lines
    vertical_line_num: int = 50                    # Number of vertical scan lines
    horizontal_fov_deg_min: float = -180           # Min horizontal FOV (degrees)
    horizontal_fov_deg_max: float = 180            # Max horizontal FOV (degrees)
    vertical_fov_deg_min: float = -2               # Min vertical FOV (degrees)
    vertical_fov_deg_max: float = 57               # Max vertical FOV (degrees)
    
    # === Output Settings ===
    pointcloud_in_world_frame: bool = False        # Point cloud coordinate frame
    
    # === Noise Settings ===
    enable_sensor_noise: bool = False              # Enable sensor noise simulation
    random_distance_noise: float = 0.03           # Distance noise standard deviation
    random_angle_noise: float = 0.15 * π/180      # Angular noise standard deviation
    pixel_dropout_prob: float = 0.01              # Probability of pixel dropout
    pixel_std_dev_multiplier: float = 0.01        # Noise multiplier
    
    # === Transform Settings ===
    euler_frame_rot_deg: list = [0.0, 0.0, 0.0]   # Frame rotation (roll, pitch, yaw)
    
    # === Placement Randomization ===
    randomize_placement: bool = True               # Enable placement randomization
    min_translation: list = [0.07, -0.06, 0.01]   # Min random translation (x,y,z)
    max_translation: list = [0.12, 0.03, 0.04]    # Max random translation (x,y,z)
    min_euler_rotation_deg: list = [-5, -5, -5]   # Min random rotation (degrees)
    max_euler_rotation_deg: list = [5, 5, 5]      # Max random rotation (degrees)
    
    # === Nominal Position (IsaacGym) ===
    nominal_position: list = [0.10, 0.0, 0.03]    # Default sensor position
    nominal_orientation_euler_deg: list = [0, 0, 0] # Default sensor orientation
    
    # === Data Processing ===
    normalize_range: bool = False                  # Normalize range values
    far_out_of_range_value: float = -1.0          # Value for far out-of-range
    near_out_of_range_value: float = -1.0         # Value for near out-of-range
```

### **Configuration Guidelines**

#### **Simulation dt Matching**
```python
# CRITICAL: LidarSensor dt must match simulation dt
scene_dt = 0.02  # Genesis/IsaacGym simulation timestep
sensor_config = LidarConfig(
    dt=scene_dt,  # Must match!
    update_frequency=50.0,  # 1/dt for every step updates
    sensor_type=LidarType.MID360
)
```

#### **Performance Optimization**
```python
# For large-scale simulations
sensor_config = LidarConfig(
    max_range=20.0,          # Limit range to reduce computation
    enable_sensor_noise=False, # Disable for faster processing
    pointcloud_in_world_frame=False,  # Get local coords first
    update_frequency=20.0     # Reduce update rate if needed
)
```

### **Common Issues and Solutions**

1. **⚠️ MOST CRITICAL: Tensor Address Changes (Memory Reference Issues)**
   - **Symptoms**: Crashes, silent failures, incorrect lidar data, Warp errors
   - **Cause**: Using `tensor = new_value` instead of `tensor[:] = new_value`
   - **Solution**: 
     ```python
     # ✅ ALWAYS DO THIS - Preserves memory address
     self.lidar_sensor.lidar_positions_tensor[:] = sensor_pos
     self.lidar_sensor.lidar_quat_tensor[:] = sensor_quat
     
     # ❌ NEVER DO THIS - Changes memory address
     self.lidar_sensor.lidar_positions_tensor = sensor_pos
     self.lidar_sensor.lidar_quat_tensor = sensor_quat
     ```
   - **Rule**: For ANY tensor in LidarSensor, use `[:]` assignment to preserve Warp references

2. **"Could not convert array interface" Error**
   - Solution: Ensure correct data types (int32 for faces, uint64 for mesh_ids)

3. **Quaternion Rotation Issues**
   - Solution: Check quaternion format (Genesis: wxyz, IsaacGym/Warp: xyzw)

4. **Sensor Position Drift**
   - Solution: Use proper offset calculation with quaternion rotation

5. **Performance Issues**
   - Solution: Reduce sensor range, lower update frequency, or limit environments

# **locomotion Policy Training**








### **Acknowledge**