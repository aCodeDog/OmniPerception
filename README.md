# OmniPerception

- [x] LidarSensor release.
- [ ] training code release.
- [ ] deploy code release.

## NOTE

According to isaacgym's test on GPU 4090, using 4096 lidar (20000 rays) environment, the rendering time of each step is about 250 ms.

The current lidar supports livox mid360, avia, hap, horizon, mid40, mid70, tele, Velodyne HDL-64, Velodyne VLP-32, Ouster OS-128, etc. 

Supports simulation env include: isaacgym, genesis, mujoco, isaac sim. However, due to time constraints, the current repo combination and instructions for use have not been fully prepared. 

If you have any difficulties in use, you can raise an issue. In the next two months, the documentation and code structure will be gradually improved.



# LidarSensor

### visulization

One Example
![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/Mid360_g1.gif)


![Mid360](https://github.com/aCodeDog/OmniPerception/blob/main/resources/images/mid360_no_shelf.gif)

Considering self-occlusion vs. not considering

### install

please install isaacgym,warp-lang[extras] and LidarSensor
```
cd LidarSensor 
pip install -e.

```

### usage

```
#you can see the mid360 lidar sensor visuliazation
pip install -e.

python LidarSensor/example/isaacgym/unitree_g1.py

#if you have installed mujoco and ros,you can also visulize in the mujoco.

python LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/lidar_vis_ros1.py or lidar_vis_ros2.py

 ```

 ### note

 Due to time limitation,

