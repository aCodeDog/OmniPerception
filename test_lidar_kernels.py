#!/usr/bin/env python3

"""
Test script to verify the LiDAR Taichi kernel setup.
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Test import with correct path
    sys.path.insert(0, '/home/zifanw/rl_robot/OmniPerception/LidarSensor/LidarSensor/sensor_kernels')
    from lidar_kernels_taichi_bvh import create_optimized_lidar_taichi_kernels
    print("✓ Successfully imported create_optimized_lidar_taichi_kernels")
    
    # Test creation
    wrapper = create_optimized_lidar_taichi_kernels()
    print("✓ Successfully created OptimizedLidarWrapper")
    
    # Check if methods exist
    if hasattr(wrapper, 'register_mesh'):
        print("✓ wrapper.register_mesh exists")
    else:
        print("✗ wrapper.register_mesh missing")
        
    if hasattr(wrapper, 'cast_rays'):
        print("✓ wrapper.cast_rays exists")
    else:
        print("✗ wrapper.cast_rays missing")
        
    # Check internal kernels
    if hasattr(wrapper, 'lidar_kernels'):
        print("✓ wrapper.lidar_kernels exists")
        kernels = wrapper.lidar_kernels
        
        if hasattr(kernels, 'draw_optimized_kernel_pointcloud'):
            print("✓ kernels.draw_optimized_kernel_pointcloud exists")
        else:
            print("✗ kernels.draw_optimized_kernel_pointcloud missing")
            print(f"Available methods: {[attr for attr in dir(kernels) if not attr.startswith('_')]}")
    else:
        print("✗ wrapper.lidar_kernels missing")
        
    # Test basic functionality with a more complex mesh
    vertices = np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],  # Bottom square
        [0, 0, 1]  # Top vertex (pyramid)
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face (2 triangles)
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]  # Side faces
    ], dtype=np.int32)
    
    print(f"Testing with {len(vertices)} vertices and {len(triangles)} triangles")
    
    try:
        print(f"Testing with {len(vertices)} vertices and {len(triangles)} triangles")
        print(f"Vertices shape: {vertices.shape}")
        print(f"Triangles shape: {triangles.shape}")
        wrapper.register_mesh(0, vertices, triangles)
        print("✓ Successfully registered mesh")
    except Exception as e:
        print(f"✗ Failed to register mesh: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTest completed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the correct directory and Genesis is installed")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
