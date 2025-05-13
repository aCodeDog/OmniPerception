
import numpy as np
from scipy.spatial.transform import Rotation

from pynput import keyboard

from visualization_msgs.msg import Marker

class KeyboardListener:
    def __init__(self, lidar_position, lidar_orientation):
        # 存储激光雷达的位置和方向
        self.lidar_position = lidar_position
        self.lidar_orientation = lidar_orientation
        
        # 存储当前的欧拉角（俯仰、偏航）
        self.euler_angles = Rotation.from_quat(lidar_orientation).as_euler('xyz')
        
        # 移动和旋转的速度
        self.move_speed = 1.  # 平移速度 (米/秒)
        self.rotate_speed = 1.  # 旋转速度 (弧度/秒)
        self.height_speed = 0.5  # 高度调整速度 (米/秒)
        
        # 当前按下的键
        self.pressed_keys = set()
        
        # 启动键盘监听器
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        print("键盘控制已启动:")
        print("  WASD: 控制水平移动")
        print("  Q/E: 控制高度上升/下降")
        print("  方向键上/下: 控制俯仰角")
        print("  方向键左/右: 控制偏航角")
        print("  ESC: 退出程序")

    def on_press(self, key):
        """处理按键按下事件"""
        try:
            # 将按键添加到已按下的键集合中
            if hasattr(key, 'char'):
                self.pressed_keys.add(key.char.lower())
            else:
                self.pressed_keys.add(key)
            
            # 如果按下ESC键，则停止监听
            if key == keyboard.Key.esc:
                print('ESC 键按下，退出程序')
                return False
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def on_release(self, key):
        """处理按键释放事件"""
        try:
            # 从已按下的键集合中移除释放的键
            if hasattr(key, 'char'):
                self.pressed_keys.discard(key.char.lower())
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            # 忽略特殊键的AttributeError
            pass

    def update_lidar_pose(self, dt):
        """根据当前按下的键更新激光雷达的位置和姿态"""
        # 获取当前的欧拉角
        pitch = self.euler_angles[1]
        yaw = self.euler_angles[2]
        
        # 计算前进方向（基于当前的yaw角）
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        right_dir = np.array([np.sin(yaw), -np.cos(yaw), 0])
        
        # 处理平移（WASD）
        move_delta = np.zeros(3)
        
        if 'w' in self.pressed_keys:
            move_delta += forward_dir * self.move_speed * dt
        if 's' in self.pressed_keys:
            move_delta -= forward_dir * self.move_speed * dt
        if 'd' in self.pressed_keys:
            move_delta += right_dir * self.move_speed * dt
        if 'a' in self.pressed_keys:
            move_delta -= right_dir * self.move_speed * dt
            
        # 处理高度调整（QE）
        if 'q' in self.pressed_keys:
            move_delta[2] += self.height_speed * dt
        if 'e' in self.pressed_keys:
            move_delta[2] -= self.height_speed * dt
            
        # 应用平移
        self.lidar_position += move_delta
        
        # 确保激光雷达不会低于地面
        self.lidar_position[2] = max(0.1, self.lidar_position[2])
        
        # 处理旋转（方向键）
        if keyboard.Key.up in self.pressed_keys:
            pitch += self.rotate_speed * dt
        if keyboard.Key.down in self.pressed_keys:
            pitch -= self.rotate_speed * dt
        if keyboard.Key.left in self.pressed_keys:
            yaw += self.rotate_speed * dt
        if keyboard.Key.right in self.pressed_keys:
            yaw -= self.rotate_speed * dt
            
        # 限制俯仰角范围
        pitch = np.clip(pitch, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        # 更新欧拉角
        self.euler_angles[1] = pitch
        self.euler_angles[2] = yaw
        
        # 将欧拉角转换为四元数
        self.lidar_orientation = Rotation.from_euler('xyz', self.euler_angles).as_quat()
        
        # 返回更新后的位置和姿态
        return self.lidar_position.copy(), self.lidar_orientation.copy()


def create_marker_from_geom(geom, marker_id, frame_id="world"):
    """从MuJoCo几何体创建ROS可视化标记"""
    # 胶囊体需要特殊处理，返回标记列表
    if geom.type == 3:  # CAPSULE
        return create_capsule_markers(geom, marker_id, frame_id)
    
    # 其他几何体正常处理，返回单个标记
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "mujoco_geoms"
    marker.id = marker_id
    marker.action = Marker.ADD
    
    # 设置位置
    marker.pose.position.x = float(geom.pos[0])
    marker.pose.position.y = float(geom.pos[1])
    marker.pose.position.z = float(geom.pos[2])
    
    # 设置旋转（从四元数）
    quat = Rotation.from_matrix(geom.mat.reshape(3, 3)).as_quat()
    marker.pose.orientation.x = float(quat[0])
    marker.pose.orientation.y = float(quat[1])
    marker.pose.orientation.z = float(quat[2])
    marker.pose.orientation.w = float(quat[3])
    
    # 设置颜色
    marker.color.r = float(geom.rgba[0])
    marker.color.g = float(geom.rgba[1])
    marker.color.b = float(geom.rgba[2])
    marker.color.a = float(geom.rgba[3])
    
    # 根据几何体类型设置标记类型和大小
    if geom.type == 0:  # PLANE
        marker.type = Marker.CUBE
        marker.scale.x = float(geom.size[0]) * 2  # 平面的长度
        marker.scale.y = float(geom.size[1]) * 2  # 平面的宽度
        marker.scale.z = 1e-3  # 平面的高度
    elif geom.type == 2:  # SPHERE
        marker.type = Marker.SPHERE
        marker.scale.x = float(geom.size[0]) * 2  # 直径 = 半径 * 2
        marker.scale.y = float(geom.size[1]) * 2
        marker.scale.z = float(geom.size[2]) * 2
    elif geom.type == 4:  # ELLIPSOID
        marker.type = Marker.SPHERE
        # 椭球体通过缩放球体来显示
        marker.scale.x = float(geom.size[0]) * 2
        marker.scale.y = float(geom.size[1]) * 2
        marker.scale.z = float(geom.size[2]) * 2
    elif geom.type == 5:  # CYLINDER
        marker.type = Marker.CYLINDER
        marker.scale.x = float(geom.size[0]) * 2  # 直径
        marker.scale.y = float(geom.size[1]) * 2
        marker.scale.z = float(geom.size[2]) * 2  # 高度
    elif geom.type == 6:  # BOX
        marker.type = Marker.CUBE
        marker.scale.x = float(geom.size[0]) * 2
        marker.scale.y = float(geom.size[1]) * 2
        marker.scale.z = float(geom.size[2]) * 2
    else:
        return []
        # 不支持的几何体类型
    
    return [marker]  # 返回单个标记的列表，保持接口一致

def create_capsule_markers(geom, marker_id, frame_id="world"):
    """创建胶囊体的可视化标记（一个圆柱体和两个半球）"""
    markers = []
    radius = float(geom.size[0])         # 半径
    half_height = float(geom.size[2])    # 圆柱部分的半高
    
    # 获取旋转矩阵和四元数
    rot_matrix = geom.mat.reshape(3, 3)
    quat = Rotation.from_matrix(rot_matrix).as_quat()
    
    # 计算胶囊体中心点位置
    center = np.array([geom.pos[0], geom.pos[1], geom.pos[2]])
    
    # 计算圆柱体的高度（直接使用2*半高）
    cylinder_height = 2 * half_height
    
    # 计算胶囊体方向的单位向量 (假设沿z轴)
    z_dir = np.array([0, 0, 1])
    # 应用旋转矩阵得到实际方向
    capsule_dir = rot_matrix.dot(z_dir)
    capsule_dir = capsule_dir / np.linalg.norm(capsule_dir)  # 确保是单位向量
    
    # 1. 创建中间的圆柱体
    cylinder = Marker()
    cylinder.header.frame_id = frame_id
    cylinder.ns = "mujoco_geoms"
    cylinder.id = marker_id * 3     # 使用基础ID的3倍，确保唯一性
    cylinder.type = Marker.CYLINDER
    cylinder.action = Marker.ADD
    
    cylinder.pose.position.x = float(center[0])
    cylinder.pose.position.y = float(center[1])
    cylinder.pose.position.z = float(center[2])
    
    cylinder.pose.orientation.x = float(quat[0])
    cylinder.pose.orientation.y = float(quat[1])
    cylinder.pose.orientation.z = float(quat[2])
    cylinder.pose.orientation.w = float(quat[3])
    
    cylinder.scale.x = radius * 2  # 直径
    cylinder.scale.y = radius * 2
    cylinder.scale.z = cylinder_height  # 高度
    
    cylinder.color.r = float(geom.rgba[0])
    cylinder.color.g = float(geom.rgba[1])
    cylinder.color.b = float(geom.rgba[2])
    cylinder.color.a = float(geom.rgba[3])
    
    markers.append(cylinder)
    
    # 计算两个半球的中心位置
    # 半球中心位于距离主中心点半高距离的位置
    sphere1_center = center + capsule_dir * half_height
    sphere2_center = center - capsule_dir * half_height
    
    # 2. 创建上半球
    top_sphere = Marker()
    top_sphere.header.frame_id = frame_id
    top_sphere.ns = "mujoco_geoms"
    top_sphere.id = marker_id * 3 + 1  # 使用基础ID的3倍+1
    top_sphere.type = Marker.SPHERE
    top_sphere.action = Marker.ADD
    
    top_sphere.pose.position.x = float(sphere1_center[0])
    top_sphere.pose.position.y = float(sphere1_center[1])
    top_sphere.pose.position.z = float(sphere1_center[2])
    
    top_sphere.pose.orientation.x = float(quat[0])
    top_sphere.pose.orientation.y = float(quat[1])
    top_sphere.pose.orientation.z = float(quat[2])
    top_sphere.pose.orientation.w = float(quat[3])
    
    top_sphere.scale.x = radius * 2
    top_sphere.scale.y = radius * 2
    top_sphere.scale.z = radius * 2
    
    top_sphere.color.r = float(geom.rgba[0])
    top_sphere.color.g = float(geom.rgba[1])
    top_sphere.color.b = float(geom.rgba[2])
    top_sphere.color.a = float(geom.rgba[3])
    
    markers.append(top_sphere)
    
    # 3. 创建下半球
    bottom_sphere = Marker()
    bottom_sphere.header.frame_id = frame_id
    bottom_sphere.ns = "mujoco_geoms"
    bottom_sphere.id = marker_id * 3 + 2  # 使用基础ID的3倍+2
    bottom_sphere.type = Marker.SPHERE
    bottom_sphere.action = Marker.ADD
    
    bottom_sphere.pose.position.x = float(sphere2_center[0])
    bottom_sphere.pose.position.y = float(sphere2_center[1])
    bottom_sphere.pose.position.z = float(sphere2_center[2])
    
    bottom_sphere.pose.orientation.x = float(quat[0])
    bottom_sphere.pose.orientation.y = float(quat[1])
    bottom_sphere.pose.orientation.z = float(quat[2])
    bottom_sphere.pose.orientation.w = float(quat[3])
    
    bottom_sphere.scale.x = radius * 2
    bottom_sphere.scale.y = radius * 2
    bottom_sphere.scale.z = radius * 2
    
    bottom_sphere.color.r = float(geom.rgba[0])
    bottom_sphere.color.g = float(geom.rgba[1])
    bottom_sphere.color.b = float(geom.rgba[2])
    bottom_sphere.color.a = float(geom.rgba[3])
    
    markers.append(bottom_sphere)
    
    return markers
