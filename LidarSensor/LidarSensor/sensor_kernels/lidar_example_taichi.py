import numpy as np
import taichi as ti
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from lidar_kernels_taichi import LidarTaichiKernels, create_lidar_taichi_kernels
    from LidarSensor.sensor_kernels.lidar_kernels_taichi_bvh import create_optimized_lidar_taichi_kernels
except ImportError:
    # Fallback for different import scenarios
    try:
        from .lidar_kernels_taichi import LidarTaichiKernels, create_lidar_taichi_kernels
    except ImportError as e:
        print(f"Failed to import LidarTaichiKernels: {e}")
        raise

# Note: Taichi is initialized by Genesis when importing lidar_kernels_taichi


def create_example_mesh():
    """Create a simple example mesh (a cube)."""
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
    ], dtype=np.float32)
    
    # Cube triangles (2 triangles per face, 6 faces)
    triangles = np.array([
        # Bottom face
        [0, 1, 2], [0, 2, 3],
        # Top face
        [4, 6, 5], [4, 7, 6],
        # Front face
        [0, 4, 5], [0, 5, 1],
        # Back face
        [2, 6, 7], [2, 7, 3],
        # Left face
        [0, 3, 7], [0, 7, 4],
        # Right face
        [1, 5, 6], [1, 6, 2]
    ], dtype=np.int32)
    
    return vertices, triangles

def create_lidar_rays(n_scan_lines=32, n_points_per_line=64, fov_v=30.0, fov_h=60.0):
    """Create ray vectors for a typical LiDAR sensor."""
    # Convert FOV to radians
    fov_v_rad = np.radians(fov_v)
    fov_h_rad = np.radians(fov_h)
    
    # Create angular grids
    vertical_angles = np.linspace(-fov_v_rad/2, fov_v_rad/2, n_scan_lines)
    horizontal_angles = np.linspace(-fov_h_rad/2, fov_h_rad/2, n_points_per_line)
    
    # Generate ray vectors in spherical coordinates
    ray_vectors = np.zeros((n_scan_lines, n_points_per_line, 3), dtype=np.float32)
    
    for i, v_angle in enumerate(vertical_angles):
        for j, h_angle in enumerate(horizontal_angles):
            # Convert spherical to cartesian (x=forward, y=left, z=up)
            ray_vectors[i, j, 0] = np.cos(v_angle) * np.cos(h_angle)  # x (forward)
            ray_vectors[i, j, 1] = np.cos(v_angle) * np.sin(h_angle)  # y (left)
            ray_vectors[i, j, 2] = np.sin(v_angle)                    # z (up)
    
    return ray_vectors

class LidarWrapper:
    """
    Unified wrapper for both Warp and Taichi LiDAR implementations.
    Allows easy switching between backends.
    """
    
    def __init__(self, backend='taichi'):
        """
        Initialize LiDAR wrapper.
        
        Args:
            backend: 'taichi' or 'warp'
        """
        self.backend = backend
        
        if backend == 'taichi':
            self.lidar_kernels = create_optimized_lidar_taichi_kernels()
        elif backend == 'warp':
            # Import Warp implementation (placeholder)
            # from lidar_kernels_warp import LidarWarpKernels
            # self.lidar_kernels = LidarWarpKernels()
            raise NotImplementedError("Warp backend not implemented in this example")
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def register_mesh(self, mesh_id: int, vertices: np.ndarray, triangles: np.ndarray):
        """Register a mesh for ray casting."""
        if self.backend == 'taichi':
            self.lidar_kernels.register_mesh(mesh_id, vertices, triangles)
        elif self.backend == 'warp':
            # Warp mesh registration would go here
            pass
    
    def cast_rays(self, lidar_positions: np.ndarray, lidar_quaternions: np.ndarray, 
                  ray_vectors: np.ndarray, far_plane: float = 100.0, 
                  pointcloud_in_world_frame: bool = True):
        """
        Cast rays and return hit points.
        
        Args:
            lidar_positions: [n_env, n_cam, 3] array of LiDAR positions
            lidar_quaternions: [n_env, n_cam, 4] array of LiDAR orientations (x,y,z,w)
            ray_vectors: [n_scan_lines, n_points, 3] array of ray directions
            far_plane: Maximum ray distance
            pointcloud_in_world_frame: If True, return points in world frame
            
        Returns:
            pixels: Hit points [n_env, n_cam, n_scan_lines, n_points, 3]
            distances: Hit distances [n_env, n_cam, n_scan_lines, n_points]
        """
        n_env, n_cam = lidar_positions.shape[:2]
        n_scan_lines, n_points = ray_vectors.shape[:2]
        
        # Prepare output arrays
        pixels = np.zeros((n_env, n_cam, n_scan_lines, n_points, 3), dtype=np.float32)
        distances = np.zeros((n_env, n_cam, n_scan_lines, n_points), dtype=np.float32)
        
        if self.backend == 'taichi':
            # Use the optimized wrapper's cast_rays method
            pixels, distances = self.lidar_kernels.cast_rays(
                lidar_positions,
                lidar_quaternions, 
                ray_vectors,
                far_plane,
                pointcloud_in_world_frame
            )
        elif self.backend == 'warp':
            # Warp kernel call would go here
            pass
            
        return pixels, distances

def example_usage():
    """Example demonstrating how to use the LiDAR wrapper."""
    print("Creating Taichi LiDAR example...")
    
    # Create example mesh
    vertices, triangles = create_example_mesh()
    print(f"Created cube mesh with {len(vertices)} vertices and {len(triangles)} triangles")
    
    # Create LiDAR rays
    ray_vectors = create_lidar_rays(n_scan_lines=16, n_points_per_line=32)
    print(f"Created ray pattern: {ray_vectors.shape}")
    
    # Initialize LiDAR with Taichi backend
    lidar = LidarWrapper(backend='taichi')
    
    # Register the mesh
    lidar.register_mesh(mesh_id=0, vertices=vertices, triangles=triangles)
    print("Registered mesh with LiDAR system")
    
    # Set up LiDAR pose (single environment, single camera)
    lidar_positions = np.array([[[0.0, 0.0, 3.0]]], dtype=np.float32)  # 3 units above cube
    lidar_quaternions = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)  # No rotation
    
    print("Casting rays...")
    
    # Cast rays
    hit_points, hit_distances = lidar.cast_rays(
        lidar_positions=lidar_positions,
        lidar_quaternions=lidar_quaternions,
        ray_vectors=ray_vectors,
        far_plane=50.0,
        pointcloud_in_world_frame=True
    )
    
    print(f"Ray casting complete!")
    print(f"Hit points shape: {hit_points.shape}")
    print(f"Hit distances shape: {hit_distances.shape}")
    
    # Analyze results
    valid_hits = hit_distances[0, 0] < 50.0  # Points that hit something
    n_hits = np.sum(valid_hits)
    total_rays = ray_vectors.shape[0] * ray_vectors.shape[1]
    
    print(f"Hit rate: {n_hits}/{total_rays} ({100.0 * n_hits / total_rays:.1f}%)")
    
    if n_hits > 0:
        min_dist = np.min(hit_distances[0, 0][valid_hits])
        max_dist = np.max(hit_distances[0, 0][valid_hits])
        mean_dist = np.mean(hit_distances[0, 0][valid_hits])
        print(f"Distance range: {min_dist:.3f} - {max_dist:.3f} (mean: {mean_dist:.3f})")
        
        # Show some example hit points
        print("First 5 hit points:")
        hit_points_flat = hit_points[0, 0][valid_hits][:5]
        for i, point in enumerate(hit_points_flat):
            print(f"  Hit {i+1}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

def benchmark_comparison():
    """Compare Taichi vs reference implementation performance."""
    print("\n=== Performance Benchmark ===")
    
    # Create larger test case
    vertices, triangles = create_example_mesh()
    ray_vectors = create_lidar_rays(n_scan_lines=64, n_points_per_line=128)
    
    lidar_positions = np.array([[[0.0, 0.0, 3.0]]], dtype=np.float32)
    lidar_quaternions = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
    
    # Test Taichi implementation
    import time
    
    lidar_taichi = LidarWrapper(backend='taichi')
    lidar_taichi.register_mesh(0, vertices, triangles)
    
    # Warm up
    lidar_taichi.cast_rays(lidar_positions, lidar_quaternions, ray_vectors)
    
    # Benchmark
    n_trials = 10
    start_time = time.time()
    
    for _ in range(n_trials):
        hit_points, hit_distances = lidar_taichi.cast_rays(
            lidar_positions, lidar_quaternions, ray_vectors
        )
    
    end_time = time.time()
    taichi_time = (end_time - start_time) / n_trials
    
    total_rays = ray_vectors.shape[0] * ray_vectors.shape[1]
    rays_per_second = total_rays / taichi_time
    
    print(f"Taichi implementation:")
    print(f"  Average time per frame: {taichi_time*1000:.2f} ms")
    print(f"  Rays per second: {rays_per_second:.0f}")
    print(f"  Total rays per test: {total_rays}")

if __name__ == "__main__":
    example_usage()
    benchmark_comparison() 