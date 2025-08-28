import numpy as np
from functools import lru_cache

def _generate_avia_pattern(s, a, b, c):
    """Avia雷达模式的核心计算函数，使用改进的算法实现更充分的空间覆盖"""
    # 预计算重复使用的值，避免重复计算
    sin_a_s = np.sin(a * s)
    cos_b_s = np.cos(b * s)
    sin_b_s = np.sin(b * s)
    
    # 预计算指数部分
    exp_term1 = 1 - np.exp(-c * s / (12 * np.pi))
    exp_term2 = 1 - np.exp(-c * s / (10 * np.pi))
    exp_term3 = 1 - np.exp(-c * s / (8 * np.pi))
    
    # 相位偏移值
    sin_a_s_p3 = np.sin(a * s + 0.3)
    cos_b_s_p7 = np.cos(b * s + 0.7)
    sin_a_s_p7 = np.sin(a * s + 0.7)
    cos_b_s_p3 = np.cos(b * s + 0.3)
    
    # X坐标的三个组件
    t1_x = sin_a_s * cos_b_s * exp_term1
    t2_x = sin_a_s_p3 * cos_b_s_p7 * exp_term2
    t3_x = sin_a_s_p7 * cos_b_s_p3 * exp_term3
    
    # 使用向量化操作合并
    x = (t1_x + 0.4*t2_x + 0.2*t3_x) / 1.6
    
    # 预计算Y坐标的相位偏移值
    sin_a_s_p5 = np.sin(a * s + 0.5)
    sin_b_s_p5 = np.sin(b * s + 0.5)
    sin_a_s_p9 = np.sin(a * s + 0.9)
    sin_b_s_p1 = np.sin(b * s + 0.1)
    
    # Y坐标的三个组件
    t1_y = sin_a_s * sin_b_s * exp_term1
    t2_y = sin_a_s_p5 * sin_b_s_p5 * exp_term2
    t3_y = sin_a_s_p9 * sin_b_s_p1 * exp_term3
    
    # 使用向量化操作合并
    y = (t1_y + 0.4*t2_y + 0.2*t3_y) / 1.6
    
    return x, y

def generate_avia_lidar(yaw_fov=70.4, vertical_fov=77.2, point_num=20000, t=0.0):
    """
    生成Livox Avia激光雷达的扫描模式，优化版本，提高空间覆盖
    
    参数:
        yaw_fov (float): 水平视场角，默认70.4度
        vertical_fov (float): 垂直视场角，默认77.2度
        point_num (int): 需要生成的点数
        t (float): 时刻参数，用于模拟不同时刻的微小变化
        
    返回:
        ray_theta (numpy.array): 水平角度数组，单位为弧度
        ray_phi (numpy.array): 垂直角度数组，单位为弧度
    """
    # 设置随机种子，但不在每次调用都设置，减少开销
    seed = int((t * 1000) % 10000)
    np.random.seed(seed)
    
    # 预计算常量，转换为弧度
    phi_range = np.radians(vertical_fov / 2.0)
    theta_range = np.radians(yaw_fov / 2.0)
    
    # 生成更长的参数范围，增加覆盖密度
    s = np.linspace(0, 18 * np.pi, point_num)
    
    # 时间相关参数，增加变化幅度
    sin_t = np.sin(t * 0.5)
    cos_t = np.cos(t * 0.5)
    a = 1.5 + 0.2 * sin_t
    b = 2.5 + 0.3 * cos_t
    c = 4.0 - 0.2 * sin_t
    
    # 使用加速的核心函数
    x, y = _generate_avia_pattern(s, a, b, c)
    
    # 归一化，更高效的实现
    max_val = max(np.max(np.abs(x)), np.max(np.abs(y)))
    if max_val > 0:
        x /= max_val
        y /= max_val
    
    # 直接计算角度，避免中间数组
    ray_theta = x * theta_range
    ray_phi = y * phi_range
    
    # 多扫描模式组合，形成更均匀的覆盖 - 使用向量化运算
    line_count = 8
    line_idx = np.random.randint(0, line_count, size=point_num)
    ray_phi += np.sin(line_idx * np.pi / line_count) * np.radians(3.0)
    
    # 确保在扫描范围内
    ray_phi = np.clip(ray_phi, -phi_range, phi_range)
    ray_theta = np.clip(ray_theta, -theta_range, theta_range)
    
    return ray_theta, ray_phi

def _generate_minicf_pattern(s, a, b, delta):
    """Minicf雷达模式的核心计算函数，改进算法实现更充分的空间覆盖"""
    # 预计算以节省计算
    a_s_delta = a * s + delta
    ap1_5_s_delta = (a+1.5) * s + delta * 1.2
    am0_5_s_delta = (a-0.5) * s + delta * 0.8
    
    # X坐标的三个组件
    x1 = np.sin(a_s_delta)
    x2 = np.sin(ap1_5_s_delta)
    x3 = np.sin(am0_5_s_delta) * 0.5
    
    # 计算总和并归一化
    x = (x1 + 0.6*x2 + 0.3*x3) / 1.9
    
    # 预计算Y坐标的参数
    b_s = b * s
    bp1_2_s = (b+1.2) * s + 0.4
    bm0_7_s = (b-0.7) * s + 0.8
    
    # Y坐标的三个组件
    y1 = np.sin(b_s)
    y2 = np.sin(bp1_2_s)
    y3 = np.sin(bm0_7_s) * 0.6
    
    # 计算总和并归一化
    y = (y1 + 0.5*y2 + 0.4*y3) / 1.9
    
    return x, y

def generate_minicf_lidar(vertical_fov=90.0, is_360lidar=True, yaw_fov=360.0, point_num=24000, t=0.0):
    """
    生成Livox Mini或Mid-40激光雷达的扫描模式，优化版本，提高空间覆盖
    
    参数:
        vertical_fov (float): 垂直视场角，默认90.0度
        is_360lidar (bool): 是否配置为360度扫描模式，默认True
        yaw_fov (float): 水平视场角，is_360lidar为True时为360度，否则设置实际值
        point_num (int): 需要生成的点数
        t (float): 时刻参数，用于模拟不同时刻的微小变化
        
    返回:
        ray_theta (numpy.array): 水平角度数组，单位为弧度
        ray_phi (numpy.array): 垂直角度数组，单位为弧度
    """
    # 设置随机种子
    np.random.seed(int((t * 1000) % 10000))
    
    # 预计算时间相关项
    sin_t = np.sin(t * 2.5)
    cos_t = np.cos(t * 2.5)
    
    # 增加参数变化，生成更丰富的扫描模式
    a = 6.5 + 0.3 * sin_t
    b = 5.3 + 0.4 * cos_t
    delta = t * 0.3
    
    # 生成参数
    s = np.linspace(0, 3 * np.pi, point_num)
    
    # 使用加速的核心函数
    x, y = _generate_minicf_pattern(s, a, b, delta)
    
    # 归一化 - 使用并行计算
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    max_x = np.max(abs_x)
    max_y = np.max(abs_y)
    max_val = max(max_x, max_y)
    
    if max_val > 0:
        x /= max_val
        y /= max_val
    
    # 根据模式计算角度
    if is_360lidar:
        # 使用改进的极坐标映射 - 避免平方计算
        r = np.sqrt(x*x + y*y)  # 更高效的距离计算
        theta = np.arctan2(y, x)
        
        # 使用非线性映射增加覆盖均匀性 - 快速幂运算
        r = np.power(r, 0.8)  # 非线性变换使边缘点更多
        r = np.clip(r, 0, 1)
        
        # 直接计算角度
        ray_theta = theta
        ray_phi = (r * 2 - 1) * np.radians(vertical_fov / 2)
    else:
        # 使用改进的直接映射
        ray_theta = x * np.radians(yaw_fov / 2)
        ray_phi = y * np.radians(vertical_fov / 2)
    
    # 预计算常用值
    phi_rad_factor = np.radians(1.0)
    s_multi = np.linspace(0, 8*np.pi, point_num)
    t_phase1 = s_multi + t
    t_phase2 = 2*s_multi + t*1.5
    t_phase3 = 3*s_multi + t*0.7
    
    # 添加多层特有模式增加覆盖 - 向量化计算
    phi_pattern1 = np.sin(t_phase1) * (3.0 * phi_rad_factor)
    phi_pattern2 = np.sin(t_phase2) * (2.5 * phi_rad_factor)
    phi_pattern3 = np.sin(t_phase3) * (1.8 * phi_rad_factor)
    ray_phi += (phi_pattern1 + 0.7*phi_pattern2 + 0.5*phi_pattern3) / 2.2
    
    # 确保在允许范围内
    ray_phi = np.clip(ray_phi, -np.radians(vertical_fov/2), np.radians(vertical_fov/2))
    
    return ray_theta, ray_phi

if __name__ == "__main__":
    # 测试代码
    import time
    import matplotlib.pyplot as plt
    
    # 测试不同雷达的扫描模式并可视化
    def visualize_pattern(ray_theta, ray_phi, title):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 将球面坐标转换为笛卡尔坐标以便可视化
        # 假设所有点距离相同
        r = 1.0
        
        # 使用向量化操作计算坐标 - 减少重复计算
        cos_phi = np.cos(ray_phi)
        sin_phi = np.sin(ray_phi)
        cos_theta = np.cos(ray_theta)
        sin_theta = np.sin(ray_theta)
        
        x = r * cos_theta * cos_phi
        y = r * sin_theta * cos_phi
        z = r * sin_phi
        
        # 使用小样本加快可视化
        sample_size = min(5000, len(x))
        sample_idx = np.random.choice(len(x), sample_size, replace=False)
        
        ax.scatter(x[sample_idx], y[sample_idx], z[sample_idx], s=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.tight_layout()
    
    # 预热函数
    print("预热函数...")
    _ = generate_avia_lidar(t=0.1, point_num=1000)
    _ = generate_minicf_lidar(t=0.1, point_num=1000)
    print("预热完成，开始性能测试...")
    
    # 生成并可视化各种激光雷达模式
    t = 0.5  # 时刻参数
    test_times = 10
    
    # Livox Avia
    st = time.time()
    for _ in range(test_times):
        ray_theta, ray_phi = generate_avia_lidar(t=t)
    et = time.time()
    print(f"Livox Avia Pattern , Point Num: {len(ray_theta)}, Time: {1e3*(et - st)/test_times:.3f} ms")
    visualize_pattern(ray_theta, ray_phi, "Livox Avia Pattern")
    
    # Livox Mini/Mid-40
    st = time.time()
    for _ in range(test_times):
        ray_theta, ray_phi = generate_minicf_lidar(t=t, yaw_fov=40.)
    et = time.time()
    print(f"Livox Mini/Mid-40 Pattern , Point Num: {len(ray_theta)}, Time: {1e3*(et - st)/test_times:.3f} ms")
    visualize_pattern(ray_theta, ray_phi, "Livox Mini/Mid-40 Pattern")

    # Livox Mini/Mid-360
    st = time.time()
    for _ in range(test_times):
        ray_theta, ray_phi = generate_minicf_lidar(t=t, is_360lidar=True,point_num=24000)
    et = time.time()
    print(f"Livox Mini/Mid-360 Pattern , Point Num: {len(ray_theta)}, Time: {1e3*(et - st)/test_times:.3f} ms")
    visualize_pattern(ray_theta, ray_phi, "Livox Mini/Mid-360 Pattern")
   
    plt.show()