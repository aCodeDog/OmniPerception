import time
import mujoco
import argparse
import traceback
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation


import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray


from mj_lidar import MjLidarSensor, create_lidar_rays, create_demo_scene
from visual_utils import KeyboardListener, create_marker_from_geom
from genera_lidar_scan_pattern import \
    LivoxGenerator, \
    generate_HDL64, \
    generate_vlp32, \
    generate_os128

class LidarVisualizer(Node):
    def __init__(self, args):
        super().__init__('mujoco_lidar_test')
        
        # 创建点云发布者
        self.pub_taichi = self.create_publisher(PointCloud2, '/lidar_points_taichi', 1)

        # 创建场景可视化标记发布者
        self.pub_scene = self.create_publisher(MarkerArray, '/mujoco_scene', 1)

        # 创建TF广播者
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 创建MuJoCo场景
        self.mj_model, self.mj_data = create_demo_scene()

        # 创建场景渲染对象
        self.scene = mujoco.MjvScene(self.mj_model, maxgeom=100)
        # 更新模拟
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 更新场景
        mujoco.mjv_updateScene(
            self.mj_model, self.mj_data, mujoco.MjvOption(), 
            None, mujoco.MjvCamera(), 
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )

        # 创建激光雷达传感器
        self.lidar = MjLidarSensor(self.scene, enable_profiling=args.profiling, verbose=args.verbose)

        # livox模式: avia mid40 mid70 mid360 tele
        self.livox_generator = LivoxGenerator("mid360")
        rays_theta, rays_phi = self.livox_generator.sample_ray_angles()

        # 创建激光雷达射线 - 使用参数指定的分辨率
        # rays_theta, rays_phi = create_lidar_rays(horizontal_resolution=args.h_res, vertical_resolution=args.v_res)
        # rays_theta, rays_phi = generate_avia_lidar(t=0.) # Livox Avia
        # rays_theta, rays_phi = generate_vlp32() # VLP-32
        # rays_theta, rays_phi = generate_os128()
        self.get_logger().info(f"射线数量: {len(rays_phi)}")

        # 设置激光雷达位置
        self.lidar_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.lidar_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 四元数(x,y,z,w)

        # 创建键盘监听器
        self.kb_listener = KeyboardListener(self.lidar_position, self.lidar_orientation)
        
        # 创建定时器
        self.timer = self.create_timer(1.0/args.rate, self.timer_callback)
        
        # 保存上一次更新的时间
        self.last_time = time.time()
        
        # 保存参数
        self.args = args

    def timer_callback(self):
        # 计算时间增量
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 更新激光雷达位置和方向
        self.lidar_position, self.lidar_orientation = self.kb_listener.update_lidar_pose(dt)
        
        # 构建激光雷达位姿矩阵
        lidar_pose = np.eye(4, dtype=np.float32)
        
        # 旋转部分（从四元数构建旋转矩阵）
        lidar_pose[:3, :3] = Rotation.from_quat(self.lidar_orientation).as_matrix()

        # 平移部分
        lidar_pose[:3, 3] = self.lidar_position
        
        # 更新模拟
        mujoco.mj_step(self.mj_model, self.mj_data)
        
        # 更新场景
        mujoco.mjv_updateScene(
            self.mj_model, self.mj_data, mujoco.MjvOption(), 
            None, mujoco.MjvCamera(), 
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        
        # 发布场景可视化标记
        self.publish_scene(self.pub_scene, self.scene)
        
        # 执行光线追踪
        start_time = time.time()

        rays_theta, rays_phi = self.livox_generator.sample_ray_angles()

        points = self.lidar.ray_cast_taichi(rays_phi, rays_theta, lidar_pose, self.scene)
        ti.sync()
        end_time = time.time()
        
        # 打印性能信息和当前位置
        if self.args.verbose:
            # 格式化欧拉角为度数
            euler_deg = np.degrees(self.kb_listener.euler_angles)
            self.get_logger().info(f"位置: [{self.lidar_position[0]:.2f}, {self.lidar_position[1]:.2f}, {self.lidar_position[2]:.2f}], "
                  f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                  f"耗时: {(end_time - start_time)*1000:.2f} ms")
            
            if self.args.profiling:
                self.get_logger().info(f"  准备时间: {self.lidar.prepare_time:.2f}ms, 内核时间: {self.lidar.kernel_time:.2f}ms")
        
        # 发布点云
        self.publish_point_cloud(self.pub_taichi, points, "lidar")
        
        # 广播激光雷达的TF
        self.broadcast_tf(self.tf_broadcaster, "world", "lidar", self.lidar_position, self.lidar_orientation)

    def publish_scene(self, publisher, mj_scene, frame_id="world"):
        """将MuJoCo场景发布为ROS可视化标记数组"""
        marker_array = MarkerArray()
        
        # 记录当前使用的标记ID
        current_id = 0
        
        # 创建每个几何体的标记
        for i in range(mj_scene.ngeom):
            geom = mj_scene.geoms[i]
            # 创建标记并返回一个标记列表
            markers = create_marker_from_geom(geom, current_id, frame_id)
            
            # 添加所有返回的标记到标记数组
            for marker in markers:
                # 在ROS2中，需要设置stamp为ROS2的时间类型
                marker.header.stamp = self.get_clock().now().to_msg()
                marker_array.markers.append(marker)
                current_id += 1
        
        # 发布标记数组
        publisher.publish(marker_array)

    def publish_point_cloud(self, publisher, points, frame_id):
        """将点云数据发布为ROS PointCloud2消息"""
        stamp = self.get_clock().now().to_msg()
            
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
        
        # 创建ROS2 PointCloud2消息
        pc_msg = PointCloud2()
        pc_msg.header.frame_id = frame_id
        pc_msg.header.stamp = stamp
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16  # 4 个 float32 (x,y,z,intensity)
        pc_msg.row_step = pc_msg.point_step * points_with_intensity.shape[1]
        pc_msg.height = 1
        pc_msg.width = points_with_intensity.shape[1]
        pc_msg.is_dense = True
        
        # 转置回(N, 4)格式并转换为字节数组
        pc_msg.data = np.transpose(points_with_intensity).astype(np.float32).tobytes()
        
        publisher.publish(pc_msg)

    def broadcast_tf(self, broadcaster, parent_frame, child_frame, translation, rotation, stamp=None):
        """广播TF变换"""
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
            
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        
        t.transform.rotation.x = float(rotation[0])
        t.transform.rotation.y = float(rotation[1])
        t.transform.rotation.z = float(rotation[2])
        t.transform.rotation.w = float(rotation[3])
        
        broadcaster.sendTransform(t)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument('--h-res', type=int, default=720, help='水平分辨率 (默认: 720)')
    parser.add_argument('--v-res', type=int, default=64, help='垂直分辨率 (默认: 64)')
    parser.add_argument('--profiling', action='store_true', help='启用性能分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=10, help='循环频率 (Hz) (默认: 10)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS2集成")
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
    print("  选中python终端窗口，按下以下键盘按键控制:")
    print("  WASD: 控制水平移动")
    print("  Q/E: 控制高度上升/下降")
    print("  方向键上/下: 控制俯仰角")
    print("  方向键左/右: 控制偏航角")
    print("  ESC: 退出程序")
    print("=" * 60 + "\n")

    print("在RViz2中设置以下显示：")
    print("1. 设置Fixed Frame为'world'")
    print("2. 添加TF显示，用于查看坐标系")
    print("3. 添加PointCloud2显示，话题为/lidar_points_taichi")
    print("4. 添加MarkerArray显示，话题为/mujoco_scene")
    print("5. 设置PointCloud2，size=0.03,Color Transformer=AxisColor")

    # 初始化ROS2
    rclpy.init()
    
    # 创建节点并运行
    node = LidarVisualizer(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("用户中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()
        print("程序结束")


if __name__ == "__main__":
    main() 