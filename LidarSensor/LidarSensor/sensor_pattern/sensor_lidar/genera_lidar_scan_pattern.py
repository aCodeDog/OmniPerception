import os
import numpy as np
from functools import lru_cache
from typing import Tuple, Union


class SpinningLidarGenerator:
    """Generator for traditional spinning lidar patterns"""
    
    @staticmethod
    def generate_HDL64(f_rot=10.0, sample_rate=2.2e6, n_channels=64, phi_fov=(-24.9, 2.0)):
        """Generate Velodyne HDL-64 scan pattern"""
        phi_min, phi_max = np.deg2rad(phi_fov)
        
        # Time sequence
        t = np.arange(0, 1./f_rot, n_channels/sample_rate)[:, None]
        
        # Horizontal angle
        theta = (2 * np.pi * f_rot * t) % (2 * np.pi)
        
        # Vertical angle
        phi = np.linspace(phi_min, phi_max, n_channels)
        
        # Generate grids
        theta_grid = theta + np.zeros((1, n_channels))
        phi_grid = np.zeros_like(theta) + phi
        
        return theta_grid.flatten(), phi_grid.flatten()
    
    @staticmethod
    def generate_VLP32(f_rot=10.0, sample_rate=1.2e6):
        """Generate Velodyne VLP-32 scan pattern"""
        vlp32_angles = np.array([
            -25.0, -22.5, -20.0, -15.0, -13.0, -10.0, -5.0, -3.0, 
            -2.333, -1.0, -0.667, -0.333, 0.0, 0.0, 0.333, 0.667, 
            1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333, 
            3.667, 4.0, 5.0, 7.0, 10.0, 15.0, 17.0, 20.0
        ])
        phi = np.deg2rad(vlp32_angles)
        
        # Time sequence
        t = np.arange(0, 1/f_rot, 32/sample_rate)[:, None]
        
        # Horizontal angle
        theta = (2 * np.pi * f_rot * t) % (2 * np.pi)
        
        # Generate grids
        theta_grid = theta + np.zeros_like(phi)
        phi_grid = np.zeros_like(theta) + phi
        
        return theta_grid.flatten(), phi_grid.flatten()
    
    @staticmethod
    def generate_OS128(f_rot=20.0, sample_rate=5.2e6):
        """Generate Ouster OS-128 scan pattern"""
        n_channels = 128
        phi = np.deg2rad(np.linspace(-22.5, 22.5, n_channels))
        
        # Time sequence
        t = np.arange(0, 1/f_rot, n_channels/sample_rate)[:, None]
        
        # Horizontal angle
        theta = (2 * np.pi * f_rot * t) % (2 * np.pi)
        
        # Generate grids
        theta_grid = theta + np.zeros_like(phi)
        phi_grid = np.zeros_like(theta) + phi
        
        return theta_grid.flatten(), phi_grid.flatten()


class LidarRayGeneratorFactory:
    """Factory class to create ray generators for different lidar types"""
    
    @staticmethod
    def create_generator(sensor_type: str):
        """Create appropriate ray generator based on sensor type"""
        if sensor_type == "simple_grid":
            return None  # Handled directly in LidarSensor
        elif sensor_type in ["avia", "horizon", "HAP", "mid360", "mid40", "mid70", "tele"]:
            return LivoxGenerator(sensor_type)
        elif sensor_type == "hdl64":
            return SpinningLidarGenerator()
        elif sensor_type == "vlp32":
            return SpinningLidarGenerator()
        elif sensor_type == "os128":
            return SpinningLidarGenerator()
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
    
    @staticmethod
    def generate_ray_angles(sensor_type: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ray angles for any supported sensor type"""
        if sensor_type == "hdl64":
            return SpinningLidarGenerator.generate_HDL64(**kwargs)
        elif sensor_type == "vlp32":
            return SpinningLidarGenerator.generate_VLP32(**kwargs)
        elif sensor_type == "os128":
            return SpinningLidarGenerator.generate_OS128(**kwargs)
        elif sensor_type in ["avia", "horizon", "HAP", "mid360", "mid40", "mid70", "tele"]:
            generator = LivoxGenerator(sensor_type)
            return generator.sample_ray_angles()
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")


class LivoxGenerator:
    """
    生成 Livox 激光雷达的扫描模式
    """
    livox_lidar_params = {
        "avia": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 77.2,
            "samples": 24000
        },
        "HAP": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 45300,
            "downsample": 1
        },
        "horizon" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid40" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid70" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 70.4,
            "samples": 10000,
        },
        "mid360" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "samples": 20000,
        },
        "tele" : {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 14.5,
            "vertical_fov": 16.1,
            "samples": 24000,
        }
    }
    def __init__(self, name):
        if name in self.livox_lidar_params:
            self.laser_min_range = self.livox_lidar_params[name]["laser_min_range"]
            self.laser_max_range = self.livox_lidar_params[name]["laser_max_range"]
            self.samples = self.livox_lidar_params[name]["samples"]
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                scan_mode_file = os.path.join(script_dir, "scan_mode", f"{name}.npy")
                self.ray_angles = np.load(scan_mode_file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Scan mode file not found!")
            self.n_rays = len(self.ray_angles)
        else:
            raise ValueError(f"Invalid LiDAR name: {name}")
        self.currStartIndex = 0

    def sample_ray_angles(self, downsample=1):
        if self.currStartIndex + self.samples > self.n_rays:
            self.ray_part1 = self.ray_angles[self.currStartIndex:]
            self.ray_part2 = self.ray_angles[:self.samples - len(self.ray_part1)]
            self.currStartIndex = self.samples - len(self.ray_part1)
            self.ray_out = np.concatenate([self.ray_part1, self.ray_part2], axis=0)
        else:
            self.ray_part1 = self.ray_angles[self.currStartIndex:self.currStartIndex + self.samples]
            self.currStartIndex += self.samples
            self.ray_out = self.ray_part1
        if downsample > 1:
            self.ray_out = self.ray_out[::downsample]
        return self.ray_out[:, 0], self.ray_out[:, 1]

# =======================================================================
# Legacy functions for backward compatibility
# =======================================================================

# Deprecated: Use SpinningLidarGenerator.generate_HDL64 instead
def generate_HDL64(f_rot=10.0, sample_rate=2.2e6, n_channels=64, phi_fov=(-24.9, 2.)):
    """DEPRECATED: Use SpinningLidarGenerator.generate_HDL64 instead"""
    return SpinningLidarGenerator.generate_HDL64(f_rot, sample_rate, n_channels, phi_fov)

# Deprecated: Use SpinningLidarGenerator.generate_VLP32 instead  
def generate_vlp32(f_rot=10.0, sample_rate=1.2e6):
    """DEPRECATED: Use SpinningLidarGenerator.generate_VLP32 instead"""
    return SpinningLidarGenerator.generate_VLP32(f_rot, sample_rate)

# Deprecated: Use SpinningLidarGenerator.generate_OS128 instead
def generate_os128(f_rot=20.0, sample_rate=5.2e6):
    """DEPRECATED: Use SpinningLidarGenerator.generate_OS128 instead"""
    return SpinningLidarGenerator.generate_OS128(f_rot, sample_rate)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    def visulize_lidar_scan_pattern(theta, phi, title, gap=50):

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        dir_local = np.stack([cos_theta * cos_phi, sin_theta * cos_phi, sin_phi], axis=1)[::gap, :]
        points = dir_local / np.linalg.norm(dir_local, axis=1, keepdims=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Draw coordinate axes
        axis_length = 1.0  # Length of the axis arrows
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r')  # X-axis (red)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g')  # Y-axis (green)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b')  # Z-axis (blue)

        # Add labels for axes
        ax.text(axis_length*1.1, 0, 0, "X", color='r')
        ax.text(0, axis_length*1.1, 0, "Y", color='g')
        ax.text(0, 0, axis_length*1.1, "Z", color='b')

        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.set_zlim(-1., 1.)
        # Add a transparent unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, linewidth=0)

        # Try to set box aspect if available (newer matplotlib versions)
        try:
            ax.set_box_aspect([1, 1, 1])
        except:
            pass
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)






    # 测试时间
    test_times = 10

    livox_generator_mid360 = LivoxGenerator("mid360")        
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = livox_generator_mid360.sample_ray_angles()
    end_time = time.time()
    print(f"生成Mid360模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    rays_theta = livox_generator_mid360.ray_angles[:,0]
    rays_phi = livox_generator_mid360.ray_angles[:,1]
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox Mid360 Pattern")

    livox_generator_mid40 = LivoxGenerator("mid40")        
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = livox_generator_mid40.sample_ray_angles()
    end_time = time.time()
    print(f"生成Mid40模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    rays_theta = livox_generator_mid40.ray_angles[:,0]
    rays_phi = livox_generator_mid40.ray_angles[:,1]
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox Mid40 Pattern")

    livox_generator_mid70 = LivoxGenerator("mid70")
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = livox_generator_mid70.sample_ray_angles()
    end_time = time.time()
    print(f"生成Mid70模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    rays_theta = livox_generator_mid70.ray_angles[:,0]
    rays_phi = livox_generator_mid70.ray_angles[:,1]
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox Mid70 Pattern")

    livox_generator_avia = LivoxGenerator("avia")
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = livox_generator_avia.sample_ray_angles()
    end_time = time.time()
    print(f"生成Avia模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    rays_theta = livox_generator_avia.ray_angles[:,0]
    rays_phi = livox_generator_avia.ray_angles[:,1]
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox Avia Pattern")

    livox_generator_tele = LivoxGenerator("tele")
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = livox_generator_tele.sample_ray_angles()
    end_time = time.time()
    print(f"生成Tele模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    rays_theta = livox_generator_tele.ray_angles[:,0]
    rays_phi = livox_generator_tele.ray_angles[:,1]
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox Tele Pattern")

    # 测试HDL64模式
    rays_theta, rays_phi = generate_HDL64()
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = generate_HDL64()
    end_time = time.time()
    print(f"生成HDL64模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox HDL64 Pattern", 50)

    # 测试VLP-32模式
    rays_theta, rays_phi = generate_vlp32()
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = generate_vlp32()
    end_time = time.time()
    print(f"生成VLP-32模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox VLP-32 Pattern", 10)

    # 测试OS-128模式
    rays_theta, rays_phi = generate_os128()
    start_time = time.time()
    for i in range(test_times):
        rays_theta, rays_phi = generate_os128()
    end_time = time.time()
    print(f"生成OS-128模式 点数: {len(rays_theta)} 平均时间: {1e3*(end_time-start_time)/test_times:.4f} ms")
    visulize_lidar_scan_pattern(rays_theta, rays_phi, "Livox OS-128 Pattern", 50)

    plt.show()