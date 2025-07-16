import time
import mujoco
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
import argparse

import rospy
import tf2_ros
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray


from mj_lidar import MjLidarSensor, create_lidar_rays, create_demo_scene
from visual_utils import KeyboardListener, create_marker_from_geom
from genera_lidar_scan_pattern import \
    LivoxGenerator, \
    generate_HDL64, \
    generate_vlp32, \
    generate_os128

def publish_scene(publisher, mj_scene, frame_id="world"):
    """将MuJoCo场景发布为ROS可视化标记数组"""
    marker_array = MarkerArray()
    
    # 记录当前使用的标记ID
    current_id = 0
    
    # 创建每个几何体的标记
    for i in range(mj_scene.ngeom):
        geom = mj_scene.geoms[i]
        # 现在 create_marker_from_geom 返回一个标记列表
        markers = create_marker_from_geom(geom, current_id, frame_id)
        
        # 添加所有返回的标记到标记数组
        for marker in markers:
            marker_array.markers.append(marker)
            current_id += 1
    
    # 发布标记数组
    publisher.publish(marker_array)


def publish_point_cloud(publisher, points, frame_id):
    """将点云数据发布为ROS PointCloud2消息"""
    stamp = rospy.Time.now()
        
    # 定义点云字段
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points
        
        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack([
                points_transposed, 
                np.ones(points_transposed.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([
                points, 
                np.ones(points.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points
        
    # 转换为ROS消息格式的点云
    pc_msg = pc2.create_cloud(
        header=rospy.Header(frame_id=frame_id, stamp=stamp),
        fields=fields,
        points=np.transpose(points_with_intensity)  # 转置回(N, 4)格式
    )
    
    publisher.publish(pc_msg)

def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp=None):
    """广播TF变换"""
    if stamp is None:
        stamp = rospy.Time.now()
        
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]
    
    broadcaster.sendTransform(t)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS集成')
    parser.add_argument('--h-res', type=int, default=720, help='水平分辨率 (默认: 720)')
    parser.add_argument('--v-res', type=int, default=64, help='垂直分辨率 (默认: 64)')
    parser.add_argument('--profiling', action='store_true', help='启用性能分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=10, help='循环频率 (Hz) (默认: 10)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS集成")
    print("=" * 60)
    print(f"配置：")
    print(f"- 水平分辨率: {args.h_res}")
    print(f"- 垂直分辨率: {args.v_res}")
    print(f"- 总射线数量: {args.h_res * args.v_res}")
    print(f"- 循环频率: {args.rate} Hz")
    print(f"- 性能分析: {'启用' if args.profiling else '禁用'}")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")
    print("=" * 60)
    print("控制说明:")
    print("  WASD: 控制水平移动")
    print("  Q/E: 控制高度上升/下降")
    print("  方向键上/下: 控制俯仰角")
    print("  方向键左/右: 控制偏航角")
    print("  ESC: 退出程序")
    print("=" * 60 + "\n")

    # 初始化ROS节点
    rospy.init_node('mujoco_lidar_test', anonymous=True)

    # 创建点云发布者
    pub_taichi = rospy.Publisher('/lidar_points_taichi', PointCloud2, queue_size=1)

    # 创建场景可视化标记发布者
    pub_scene = rospy.Publisher('/mujoco_scene', MarkerArray, queue_size=1)

    # 创建TF广播者
    tf_broadcaster = tf2_ros.TransformBroadcaster()

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

    # livox模式: avia mid40 mid70 mid360 tele
    livox_generator = LivoxGenerator("mid360")
    rays_theta, rays_phi = livox_generator.sample_ray_angles()

    # 创建激光雷达射线 - 使用参数指定的分辨率
    # rays_theta, rays_phi = create_lidar_rays(horizontal_resolution=args.h_res, vertical_resolution=args.v_res)
    # rays_theta, rays_phi = generate_avia_lidar(t=0.) # Livox Avia
    # rays_theta, rays_phi = generate_vlp32() # VLP-32
    # rays_theta, rays_phi = generate_os128()

    print(f"射线数量: {len(rays_phi)}")

    # 设置激光雷达位置
    lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lidar_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 四元数(x,y,z,w)

    # 创建键盘监听器
    kb_listener = KeyboardListener(lidar_position, lidar_orientation)

    # 主循环
    rate = rospy.Rate(args.rate)  # 使用参数指定的循环频率
    last_time = time.time()

    print("在RViz中设置以下显示：")
    print("1. 添加TF显示，用于查看坐标系")
    print("2. 添加PointCloud2显示，话题为/lidar_points_taichi")
    print("3. 添加MarkerArray显示，话题为/mujoco_scene")
    print("4. 设置Fixed Frame为'world'")
       
    try:
        while not rospy.is_shutdown():
            # 计算时间增量
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 更新激光雷达位置和方向
            lidar_position, lidar_orientation = kb_listener.update_lidar_pose(dt)
            
            # 构建激光雷达位姿矩阵
            lidar_pose = np.eye(4, dtype=np.float32)
            
            # 旋转部分（从四元数构建旋转矩阵）
            lidar_pose[:3, :3] = Rotation.from_quat(lidar_orientation).as_matrix()

            # 平移部分
            lidar_pose[:3, 3] = lidar_position
            
            # 更新模拟
            mujoco.mj_step(mj_model, mj_data)
            
            # 更新场景
            mujoco.mjv_updateScene(
                mj_model, mj_data, mujoco.MjvOption(), 
                None, mujoco.MjvCamera(), 
                mujoco.mjtCatBit.mjCAT_ALL.value, scene
            )
            
            # 发布场景可视化标记
            publish_scene(pub_scene, scene)
            
            # 执行光线追踪
            start_time = time.time()

            rays_theta, rays_phi = livox_generator.sample_ray_angles()

            points = lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, scene)
            ti.sync()
            end_time = time.time()
            
            # 打印性能信息和当前位置
            if args.verbose:
                # 格式化欧拉角为度数
                euler_deg = np.degrees(kb_listener.euler_angles)
                print(f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                      f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                      f"耗时: {(end_time - start_time)*1000:.2f} ms")
                
                if args.profiling:
                    print(f"  准备时间: {lidar.prepare_time:.2f}ms, 内核时间: {lidar.kernel_time:.2f}ms")
            
            # 发布点云
            publish_point_cloud(pub_taichi, points, "lidar")
            
            # 广播激光雷达的TF
            broadcast_tf(tf_broadcaster, "world", "lidar", lidar_position, lidar_orientation)
            
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("程序结束")