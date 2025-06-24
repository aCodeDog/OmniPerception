# LidarSensor

### **Install**

Please install **isaacgym**, **warp-lang[extras]** and **LidarSensor**, **taichi** (**Python=3.8**)
```
pip install warp-lang[extras] taichi yourdfpy
cd LidarSensor 
pip install -e.
```

### **Usage**

```
# You can see the mid360 lidar sensor visualization
pip install -e.

cd LidarSensor/resources/robots/g1_29/
python process_body_mesh.py  # Consider self-occlusion
cd LidarSensor/example/isaacgym
python unitree_g1.py

# If you have installed mujoco and ros, you can also visualize taichi kernel lidar in the mujoco.

source /opt/ros/humble/setup.bash 
# Please use /usr/bin/python3
/usr/bin/python3 LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/lidar_vis_ros1.py or lidar_vis_ros2.py
```