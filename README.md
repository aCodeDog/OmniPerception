# **OmniPerception**

- [Half] **LidarSensor release**.
- [ ] **training code release**.
- [ ] **deploy code release**.

## **NOTE**

**[Trying hard!] This repository is still not fully built. This is due to my terrible coding habits, and many places need to be reorganized ！！！**

According to isaacgym's test on GPU 4090, using **4096 lidar (20000 rays)** environment, the rendering time of each step is about **250 ms**.

The current lidar supports **livox mid360, avia, hap, horizon, mid40, mid70, tele, Velodyne HDL-64, Velodyne VLP-32, Ouster OS-128**, etc. 

Supports simulation env include: **isaacgym, genesis, mujoco, isaac sim**. However, due to time constraints, the current repo instructions for use have not been fully prepared. 

If you have any difficulties in use, you can raise an issue. In the next two months, the documentation and code structure will be gradually completed.

# **LidarSensor**

### **Visualization**

**One Example**
![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/Mid360_g1.gif)

![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/mid360_no_shelf.gif)

**Considering self-occlusion vs. not considering**

### **Install**

Please install **isaacgym**, **warp-lang[extras]** and **LidarSensor**, **taichi**
```
pip install warp-lang[extras],taichi
cd LidarSensor 
pip install -e.
```

### **Usage**

```
# You can see the mid360 lidar sensor visualization
pip install -e.

cd LidarSensor/resources/robots/g1_29/process_body_mesh.py
python process_body_mesh.py  # Consider self-occlusion
cd LidarSensor/example/isaacgym
python unitree_g1.py

# If you have installed mujoco and ros, you can also visualize taichi kernel lidar in the mujoco.

source /opt/ros/humble/setup.bash 
# Please use /usr/bin/python3
/usr/bin/python3 LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/lidar_vis_ros1.py or lidar_vis_ros2.py
```


# **Integration Guide**

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

**Same for Both Simulators:**
```python
# Create LidarSensor configuration
sensor_config = LidarConfig(
    sensor_type=LidarType.MID360,  # Choose your sensor type
    dt=sim_dt,  # CRITICAL: Must match simulation dt
    max_range=20.0,
    update_frequency=1.0/sim_dt,  # Update every simulation step
    return_pointcloud=True,
    pointcloud_in_world_frame=False,  # Get local coordinates first
    enable_sensor_noise=False,  # Disable for faster processing
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