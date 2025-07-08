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
- **IsaacLab** - Next-generation robotics simulation platform
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






## **🚀 Platform-Specific Integration Guides**

Choose your simulation platform for detailed installation and usage instructions:

### **📦 IsaacLab** - *Recommended for large-scale RL training*
> **Native LiDAR integration with 7+ Livox sensor types and optimized performance**

**Key Features**: 
- ✅ Native `LidarSensor` class integration  
- ✅ Realistic Livox patterns (`.npy` files)
- ✅ Optimized for 1000+ environments  
- ✅ Easy dataclass configuration

**📂 [**Complete IsaacLab Guide**](https://github.com/aCodeDog/OmniPerception/blob/main/LidarSensor/LidarSensor/example/isaaclab/)**

---

### **🎮 IsaacGym** - *For NVIDIA GPU-accelerated physics*  
> **Direct GPU ray-casting with Warp integration**

**Key Features**:
- ✅ GPU-accelerated ray tracing
- ✅ Multiple sensor configurations  
- ✅ Real-time visualization
- ✅ Flexible terrain integration

**📂 [**Complete IsaacGym Guide**](https://github.com/aCodeDog/OmniPerception/blob/main/LidarSensor/LidarSensor/example/isaacgym/)**

---

### **🌟 Genesis** - *For high-performance physics simulation*
> **Modern physics engine with optimized LiDAR support**

**Key Features**:
- ✅ High-performance physics
- ✅ Multiple robot platforms
- ✅ Realistic sensor modeling
- ✅ Cross-platform support

**📂 [**Complete Genesis Guide**](https://github.com/aCodeDog/OmniPerception/blob/main/LidarSensor/LidarSensor/example/genesis/)**

---


### **🎯 Quick Start**

Choose your platform above and follow the complete installation guide. Each platform includes:
- ✅ **Step-by-step installation** with automated scripts
- ✅ **Configuration examples** for different use cases  
- ✅ **Performance benchmarks** and optimization tips
- ✅ **Troubleshooting guides** for common issues
- ✅ **Complete working examples** with popular robots

---

## **✨ Key Features**

### **🔬 Realistic Sensor Models**
- **7+ Livox LiDAR Types**: Mid-360, Avia, Horizon, HAP, Mid-40, Mid-70, Tele
- **Authentic Scan Patterns**: Real `.npy` files from actual hardware
- **Velodyne & Ouster Support**: HDL-64, VLP-32, OS-128 patterns
- **Noise Simulation**: Distance noise, pixel dropout, and angle variance

### **⚡ High Performance**
- **GPU-Accelerated**: CUDA/Warp-based ray tracing for massive parallelism
- **Vectorized Operations**: No Python loops, optimized for 1000+ environments
- **Memory Efficient**: Smart tensor management and reuse
- **Configurable LOD**: Adjustable ray density for performance tuning

### **🎨 Easy Integration**
- **One-Line Setup**: Simple sensor configuration with sensible defaults
- **Multiple Output Formats**: Distance arrays, point clouds, hit positions
- **Flexible Mounting**: Dynamic sensor positioning with offset transforms
- **Debug Visualization**: Built-in ray visualization for development

---

## **🤖 Supported Robots & Platforms**

### **Robot Platforms**
- **Quadrupeds**: Unitree Go2, G1, ANYmal, Spot
- **Humanoids**: G1, Atlas, Digit, custom bipeds
- **Mobile Robots**: Custom wheeled/tracked platforms
- **Multi-Robot**: Support for heterogeneous robot swarms

### **Simulation Engines**
- **IsaacLab** ⭐ - Next-gen RL platform with native integration
- **IsaacGym** - NVIDIA's GPU-accelerated physics engine  
- **Genesis** - High-performance modern physics simulation
- **Mujoco** - Advanced contact dynamics (coming soon)
- **Isaac Sim** - Omniverse integration (coming soon)

---

## **🏃‍♂️ Policy Training**

> 📚 **Ready for RL Training?** Check our training examples in each platform directory with complete locomotion policies for quadruped robots.

---

## **🎓 What You Get**

After following your platform-specific guide, you'll have:

✅ **Realistic LiDAR simulation** with authentic sensor patterns  
✅ **High-performance ray-casting** optimized for large-scale training  
✅ **Complete robot examples** with popular quadruped and humanoid robots  
✅ **Flexible configuration** for different use cases and research needs  
✅ **Production-ready code** tested with 1000+ environments  

---

## **🤝 Contributing & Support**

- **🐛 Issues**: Report bugs or request features via GitHub Issues
- **📖 Documentation**: Platform-specific guides in `/example/` directories  
- **🔬 Research**: Cite our work if you use OmniPerception in research
- **💬 Discussions**: Join our community for tips and collaboration

---

### **Acknowledgments**