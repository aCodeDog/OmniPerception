#!/usr/bin/env python3
"""
Comprehensive Performance Test for Taichi LiDAR System

This script performs detailed benchmarking of the Taichi-based LiDAR system
with different configurations to analyze performance characteristics.

Test parameters:
- Number of environments: 1, 100, 500, 1000
- Ray counts: 256, 1024, 4096, 10000
- Multiple test runs for statistical accuracy
"""

import torch
import numpy as np
import time
import sys
import os
import csv
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Matplotlib/seaborn not available. Plotting will be disabled.")
    HAS_PLOTTING = False

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../../sensor_kernels'))
sys.path.insert(0, os.path.join(current_dir, '../../'))

try:
    from LidarSensor.sensor_kernels.lidar_example_taichi import LidarWrapper, create_example_mesh, create_lidar_rays
except ImportError:
    # Fallback import paths
    try:
        sys.path.insert(0, os.path.join(current_dir, '../..'))
        from sensor_kernels.lidar_example_taichi import LidarWrapper, create_example_mesh, create_lidar_rays
    except ImportError:
        from lidar_example_taichi import LidarWrapper, create_example_mesh, create_lidar_rays


@dataclass
class TestConfig:
    """Test configuration parameters"""
    num_envs: int
    ray_count: int
    n_scan_lines: int
    n_points_per_line: int
    n_trials: int = 10
    warmup_trials: int = 3
    
    @property
    def total_rays(self) -> int:
        return self.n_scan_lines * self.n_points_per_line


@dataclass
class TestResult:
    """Test result data"""
    config: TestConfig
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    rays_per_second: float
    memory_usage: float
    hit_rate: float
    success: bool
    error_msg: str = ""


class LidarPerformanceTester:
    """Comprehensive LiDAR performance testing suite"""
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.results: List[TestResult] = []
        self.lidar_wrapper = None
        
        print(f"Initializing LiDAR Performance Tester on {device}")
        
        # Create test mesh once
        self.vertices, self.triangles = create_example_mesh()
        print(f"Test mesh: {len(self.vertices)} vertices, {len(self.triangles)} triangles")
        
        # Setup output directory
        self.output_dir = f"lidar_benchmark_results_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
    
    def generate_test_configurations(self) -> List[TestConfig]:
        """Generate all test configurations"""
        
        # Base configurations
        num_envs_list = [1, 100, 500, 1000]
        ray_counts = [256, 1024, 4096, 10000]
        
        configs = []
        
        for num_envs in num_envs_list:
            for ray_count in ray_counts:
                # Calculate scan lines and points per line
                # Aim for roughly square patterns
                n_scan_lines = int(np.sqrt(ray_count))
                n_points_per_line = ray_count // n_scan_lines
                actual_rays = n_scan_lines * n_points_per_line
                
                # Skip if too far from target
                if abs(actual_rays - ray_count) > ray_count * 0.1:
                    continue
                
                # Adjust trials based on complexity
                if num_envs <= 100 and ray_count <= 1024:
                    n_trials = 20
                elif num_envs <= 500 and ray_count <= 4096:
                    n_trials = 10
                else:
                    n_trials = 5
                
                config = TestConfig(
                    num_envs=num_envs,
                    ray_count=actual_rays,
                    n_scan_lines=n_scan_lines,
                    n_points_per_line=n_points_per_line,
                    n_trials=n_trials
                )
                configs.append(config)
        
        return configs
    
    def setup_lidar_for_test(self, config: TestConfig) -> bool:
        """Setup LiDAR system for specific test configuration"""
        try:
            # Clean up previous instance
            if self.lidar_wrapper is not None:
                del self.lidar_wrapper
            
            # Force garbage collection and clear GPU cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create new wrapper
            self.lidar_wrapper = LidarWrapper(backend='taichi')
            
            # Register mesh
            self.lidar_wrapper.register_mesh(
                mesh_id=0,
                vertices=self.vertices,
                triangles=self.triangles
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to setup LiDAR for config {config}: {e}")
            return False
    
    def run_single_test(self, config: TestConfig) -> TestResult:
        """Run a single performance test"""
        print(f"\nTesting: {config.num_envs} envs, {config.ray_count} rays ({config.n_scan_lines}x{config.n_points_per_line})")
        
        # Setup LiDAR system
        if not self.setup_lidar_for_test(config):
            return TestResult(
                config=config,
                mean_time=0, std_time=0, min_time=0, max_time=0,
                rays_per_second=0, memory_usage=0, hit_rate=0,
                success=False,
                error_msg="Failed to setup LiDAR system"
            )
        
        try:
            # Create ray pattern
            ray_vectors = create_lidar_rays(
                n_scan_lines=config.n_scan_lines,
                n_points_per_line=config.n_points_per_line,
                fov_v=30.0,
                fov_h=120.0
            )
            
            # Setup test positions and orientations
            lidar_positions = np.zeros((config.num_envs, 1, 3), dtype=np.float32)
            lidar_quaternions = np.zeros((config.num_envs, 1, 4), dtype=np.float32)
            
            # Distribute environments in a grid
            if config.num_envs > 1:
                grid_size = int(np.ceil(np.sqrt(config.num_envs)))
                spacing = 5.0
                
                for i in range(config.num_envs):
                    row = i // grid_size
                    col = i % grid_size
                    lidar_positions[i, 0, 0] = (col - grid_size/2) * spacing
                    lidar_positions[i, 0, 1] = (row - grid_size/2) * spacing
                    lidar_positions[i, 0, 2] = 2.0
            else:
                lidar_positions[0, 0] = [0.0, 0.0, 2.0]
            
            # Set orientations (no rotation)
            lidar_quaternions[:, 0, 3] = 1.0  # w = 1
            
            # Warmup runs
            print(f"  Warming up ({config.warmup_trials} runs)...")
            for _ in range(config.warmup_trials):
                self.lidar_wrapper.cast_rays(
                    lidar_positions=lidar_positions,
                    lidar_quaternions=lidar_quaternions,
                    ray_vectors=ray_vectors,
                    far_plane=20.0,
                    pointcloud_in_world_frame=True
                )
            
            # Benchmark runs
            print(f"  Benchmarking ({config.n_trials} runs)...")
            times = []
            hit_rates = []
            
            # Memory usage before test
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            else:
                memory_before = 0
            
            for trial in range(config.n_trials):
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                # Actual ray casting
                hit_points, hit_distances = self.lidar_wrapper.cast_rays(
                    lidar_positions=lidar_positions,
                    lidar_quaternions=lidar_quaternions,
                    ray_vectors=ray_vectors,
                    far_plane=20.0,
                    pointcloud_in_world_frame=True
                )
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                
                # Calculate hit rate (only for first trial to save time)
                if trial == 0 and hit_distances is not None:
                    valid_hits = np.sum(hit_distances < 20.0)
                    total_rays = config.num_envs * config.ray_count
                    hit_rate = valid_hits / total_rays
                    hit_rates.append(hit_rate)
                
                # Progress indicator
                if (trial + 1) % max(1, config.n_trials // 5) == 0:
                    print(f"    Trial {trial + 1}/{config.n_trials}: {elapsed*1000:.2f}ms")
            
            # Memory usage after test
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            # Calculate statistics
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            total_rays_per_frame = config.num_envs * config.ray_count
            rays_per_second = total_rays_per_frame / mean_time
            
            avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
            
            result = TestResult(
                config=config,
                mean_time=mean_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                rays_per_second=rays_per_second,
                memory_usage=memory_usage,
                hit_rate=avg_hit_rate,
                success=True
            )
            
            print(f"  ✓ Results: {mean_time*1000:.2f}±{std_time*1000:.2f}ms, "
                  f"{rays_per_second:.0f} rays/s, hit rate: {avg_hit_rate:.1%}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TestResult(
                config=config,
                mean_time=0, std_time=0, min_time=0, max_time=0,
                rays_per_second=0, memory_usage=0, hit_rate=0,
                success=False,
                error_msg=str(e)
            )
    
    def run_full_benchmark(self) -> List[TestResult]:
        """Run the complete benchmark suite"""
        print("="*80)
        print("TAICHI LIDAR PERFORMANCE BENCHMARK")
        print("="*80)
        
        configs = self.generate_test_configurations()
        print(f"Generated {len(configs)} test configurations")
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running test configuration...")
            
            result = self.run_single_test(config)
            results.append(result)
            
            # Save intermediate results
            self.save_results_csv([result], append=True)
            
            # Memory cleanup between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results = results
        return results
    
    def save_results_csv(self, results: List[TestResult], append: bool = False):
        """Save results to CSV file"""
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        mode = 'a' if append and os.path.exists(csv_path) else 'w'
        
        with open(csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            
            if mode == 'w':
                # Write header
                writer.writerow([
                    'num_envs', 'ray_count', 'n_scan_lines', 'n_points_per_line',
                    'mean_time_ms', 'std_time_ms', 'min_time_ms', 'max_time_ms',
                    'rays_per_second', 'memory_usage_gb', 'hit_rate', 
                    'success', 'error_msg'
                ])
            
            for result in results:
                writer.writerow([
                    result.config.num_envs,
                    result.config.ray_count,
                    result.config.n_scan_lines,
                    result.config.n_points_per_line,
                    result.mean_time * 1000,
                    result.std_time * 1000,
                    result.min_time * 1000,
                    result.max_time * 1000,
                    result.rays_per_second,
                    result.memory_usage,
                    result.hit_rate,
                    result.success,
                    result.error_msg
                ])
    
    def save_results_json(self, results: List[TestResult]):
        """Save detailed results to JSON"""
        json_data = []
        
        for result in results:
            json_data.append({
                'config': {
                    'num_envs': result.config.num_envs,
                    'ray_count': result.config.ray_count,
                    'n_scan_lines': result.config.n_scan_lines,
                    'n_points_per_line': result.config.n_points_per_line,
                    'n_trials': result.config.n_trials
                },
                'performance': {
                    'mean_time_ms': result.mean_time * 1000,
                    'std_time_ms': result.std_time * 1000,
                    'min_time_ms': result.min_time * 1000,
                    'max_time_ms': result.max_time * 1000,
                    'rays_per_second': result.rays_per_second,
                },
                'resources': {
                    'memory_usage_gb': result.memory_usage,
                },
                'quality': {
                    'hit_rate': result.hit_rate,
                },
                'success': result.success,
                'error_msg': result.error_msg
            })
        
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def generate_analysis_plots(self, results: List[TestResult]):
        """Generate analysis plots"""
        if not HAS_PLOTTING:
            print("Plotting libraries not available. Skipping plot generation.")
            return
            
        print("\nGenerating analysis plots...")
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Prepare data
        num_envs = [r.config.num_envs for r in successful_results]
        ray_counts = [r.config.ray_count for r in successful_results]
        mean_times = [r.mean_time * 1000 for r in successful_results]
        rays_per_sec = [r.rays_per_second for r in successful_results]
        memory_usage = [r.memory_usage for r in successful_results]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Taichi LiDAR Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Execution time vs number of environments
        ax1 = axes[0, 0]
        for ray_count in sorted(set(ray_counts)):
            subset = [(e, t) for e, r, t in zip(num_envs, ray_counts, mean_times) if r == ray_count]
            if subset:
                envs, times = zip(*subset)
                ax1.plot(envs, times, 'o-', label=f'{ray_count} rays', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Environments')
        ax1.set_ylabel('Mean Execution Time (ms)')
        ax1.set_title('Execution Time vs Environment Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. Execution time vs ray count
        ax2 = axes[0, 1]
        for env_count in sorted(set(num_envs)):
            subset = [(r, t) for e, r, t in zip(num_envs, ray_counts, mean_times) if e == env_count]
            if subset:
                rays, times = zip(*subset)
                ax2.plot(rays, times, 's-', label=f'{env_count} envs', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Rays')
        ax2.set_ylabel('Mean Execution Time (ms)')
        ax2.set_title('Execution Time vs Ray Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 3. Throughput (rays per second)
        ax3 = axes[0, 2]
        total_rays = [e * r for e, r in zip(num_envs, ray_counts)]
        scatter = ax3.scatter(total_rays, rays_per_sec, c=mean_times, s=60, alpha=0.7, cmap='viridis')
        ax3.set_xlabel('Total Rays per Frame')
        ax3.set_ylabel('Rays per Second')
        ax3.set_title('Throughput vs Total Ray Count')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Execution Time (ms)')
        
        # 4. Memory usage
        ax4 = axes[1, 0]
        if any(m > 0 for m in memory_usage):
            for ray_count in sorted(set(ray_counts)):
                subset = [(e, m) for e, r, m in zip(num_envs, ray_counts, memory_usage) if r == ray_count and m > 0]
                if subset:
                    envs, mem = zip(*subset)
                    ax4.plot(envs, mem, '^-', label=f'{ray_count} rays', linewidth=2, markersize=6)
            ax4.set_xlabel('Number of Environments')
            ax4.set_ylabel('Memory Usage (GB)')
            ax4.set_title('Memory Usage vs Environment Count')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Memory data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Memory Usage (No Data)')
        
        # 5. Scaling efficiency
        ax5 = axes[1, 1]
        baseline_times = {}
        for ray_count in sorted(set(ray_counts)):
            subset = [(e, t) for e, r, t in zip(num_envs, ray_counts, mean_times) if r == ray_count]
            if subset:
                envs, times = zip(*subset)
                if 1 in envs:
                    baseline_idx = envs.index(1)
                    baseline_time = times[baseline_idx]
                    baseline_times[ray_count] = baseline_time
                    
                    # Calculate efficiency (ideal vs actual scaling)
                    efficiency = [(baseline_time * e) / t for e, t in zip(envs, times)]
                    ax5.plot(envs, efficiency, 'o-', label=f'{ray_count} rays', linewidth=2, markersize=6)
        
        ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect scaling')
        ax5.set_xlabel('Number of Environments')
        ax5.set_ylabel('Scaling Efficiency')
        ax5.set_title('Parallel Scaling Efficiency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        
        # 6. Performance summary heatmap
        ax6 = axes[1, 2]
        
        # Create performance matrix
        unique_envs = sorted(set(num_envs))
        unique_rays = sorted(set(ray_counts))
        
        perf_matrix = np.full((len(unique_rays), len(unique_envs)), np.nan)
        
        for i, ray_count in enumerate(unique_rays):
            for j, env_count in enumerate(unique_envs):
                matching = [r for r in successful_results 
                           if r.config.ray_count == ray_count and r.config.num_envs == env_count]
                if matching:
                    perf_matrix[i, j] = matching[0].rays_per_second
        
        # Create heatmap
        im = ax6.imshow(perf_matrix, cmap='viridis', aspect='auto')
        ax6.set_xticks(range(len(unique_envs)))
        ax6.set_xticklabels(unique_envs)
        ax6.set_yticks(range(len(unique_rays)))
        ax6.set_yticklabels(unique_rays)
        ax6.set_xlabel('Number of Environments')
        ax6.set_ylabel('Number of Rays')
        ax6.set_title('Performance Heatmap (Rays/sec)')
        plt.colorbar(im, ax=ax6, label='Rays per Second')
        
        # Add text annotations
        for i in range(len(unique_rays)):
            for j in range(len(unique_envs)):
                if not np.isnan(perf_matrix[i, j]):
                    text = ax6.text(j, i, f'{perf_matrix[i, j]:.0f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "performance_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to: {plot_path}")
    
    def generate_summary_report(self, results: List[TestResult]):
        """Generate summary report"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("TAICHI LIDAR PERFORMANCE BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total tests: {len(results)}\n")
            f.write(f"Successful: {len(successful_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n\n")
            
            if successful_results:
                # Best performance
                best_throughput = max(successful_results, key=lambda r: r.rays_per_second)
                fastest_time = min(successful_results, key=lambda r: r.mean_time)
                
                f.write("BEST PERFORMANCE:\n")
                f.write(f"Highest throughput: {best_throughput.rays_per_second:.0f} rays/sec ")
                f.write(f"({best_throughput.config.num_envs} envs, {best_throughput.config.ray_count} rays)\n")
                f.write(f"Fastest execution: {fastest_time.mean_time*1000:.2f}ms ")
                f.write(f"({fastest_time.config.num_envs} envs, {fastest_time.config.ray_count} rays)\n\n")
                
                # Performance by configuration
                f.write("PERFORMANCE BY CONFIGURATION:\n")
                f.write("-" * 30 + "\n")
                
                for result in sorted(successful_results, key=lambda r: (r.config.num_envs, r.config.ray_count)):
                    f.write(f"Envs: {result.config.num_envs:4d}, Rays: {result.config.ray_count:5d} | ")
                    f.write(f"Time: {result.mean_time*1000:6.2f}ms | ")
                    f.write(f"Throughput: {result.rays_per_second:8.0f} rays/s | ")
                    f.write(f"Hit rate: {result.hit_rate:.1%}\n")
            
            if failed_results:
                f.write("\nFAILED TESTS:\n")
                f.write("-" * 15 + "\n")
                for result in failed_results:
                    f.write(f"Envs: {result.config.num_envs}, Rays: {result.config.ray_count} - {result.error_msg}\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    """Main benchmarking function"""
    print("Taichi LiDAR Performance Benchmark")
    print("-" * 40)
    
    # Check device availability
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU. Performance will be significantly slower.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create tester
    tester = LidarPerformanceTester(device=device)
    
    try:
        # Run benchmark
        results = tester.run_full_benchmark()
        
        # Save results
        print("\nSaving results...")
        tester.save_results_csv(results)
        tester.save_results_json(results)
        
        # Generate analysis
        print("Generating analysis...")
        tester.generate_analysis_plots(results)
        tester.generate_summary_report(results)
        
        print(f"\nBenchmark completed! Results saved to: {tester.output_dir}")
        
        # Print quick summary
        successful = [r for r in results if r.success]
        if successful:
            best = max(successful, key=lambda r: r.rays_per_second)
            print(f"\nBest performance: {best.rays_per_second:.0f} rays/sec ")
            print(f"Configuration: {best.config.num_envs} envs, {best.config.ray_count} rays")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if tester.lidar_wrapper is not None:
            del tester.lidar_wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
