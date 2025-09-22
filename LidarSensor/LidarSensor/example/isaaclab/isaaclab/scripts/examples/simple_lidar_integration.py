#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment for a quadruped robot with LiDAR sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The LiDAR sensor is used to detect the environment.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/examples/simple_lidar_integration.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment with LiDAR.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--enable_lidar", type=bool, default=True ,help="Enable LiDAR sensor for benchmarking.")
parser.add_argument("--benchmark_steps", type=int, default=1000, help="Number of steps to run for benchmarking.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import LidarSensor, LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import RayCasterCfg, patterns
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0), rot=(1, 0, 0., 0.)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Conditionally add LiDAR sensor based on args
    if args_cli.enable_lidar:
        # LiDAR sensor
        lidar_sensor = LidarSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=LidarSensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 1, 0., 0.)),
            attach_yaw_only=False,
            ray_alignment = "world",
            pattern_cfg=LivoxPatternCfg(
                sensor_type="mid360",
                samples=24000,  # Reduced for better performance with 1024 envs
            ),
            mesh_prim_paths=["/World/ground","/World/static"], #this is for global dynamic and static mesh
            # You can also specify specific prim paths for dynamic objects if needed
            
                
            #this is for local dynamic mesh,other env sensor will not consider this
            dynamic_env_mesh_prim_paths=["{ENV_REGEX_NS}/Cube/geometry/mesh","{ENV_REGEX_NS}/Sphere/geometry/mesh",
                                         "{ENV_REGEX_NS}/Robot/LF_HIP/visuals/mesh_0",
                                         "{ENV_REGEX_NS}/Robot/LF_SHANK/visuals/mesh_0",
                                         "{ENV_REGEX_NS}/Robot/LF_THIGH/visuals/mesh_0",
                                         "{ENV_REGEX_NS}/Robot/LF_FOOT/visuals/mesh_0",
                                         "{ENV_REGEX_NS}/Robot/RF_HIP/visuals/mesh_0",
                                         "{ENV_REGEX_NS}/Robot/LF_HIP/visuals/mesh_1",
                                         ],
            max_distance=20.0,
            min_range=0.2,
            return_pointcloud=False,  # Disable pointcloud for performance
            pointcloud_in_world_frame=False,
            enable_sensor_noise=False,  # Disable noise for pure performance test
            random_distance_noise=0.0,
            update_frequency=25.0,  # 25 Hz for better performance
            debug_vis=True,  # Disable visualization for performance
        )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    Cube = AssetBaseCfg(
        prim_path="/World/static/Cubes_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 1.0)),
    )
    Cube_2 = AssetBaseCfg(
        prim_path="/World/static/Cubes_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -1.0, 1.0)),
    )
    Sphere = AssetBaseCfg(
        prim_path="/World/static/Spheres_1",
        spawn=sim_utils.SphereCfg(
            radius=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 1.0)),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # Using LiDAR distance data instead of height scan
        # lidar_distances = ObsTerm(
        #     func=mdp.height_scan,  # We can reuse height_scan function as it works with any ray caster
        #     params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, 'lidar_sensor') and self.scene.lidar_sensor is not None:
            self.scene.lidar_sensor.update_period = self.decimation * self.sim.dt  # 50 Hz
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz

def main():
    """Main function."""
    print(f"[INFO]: Running benchmark with {args_cli.num_envs} environments")
    print(f"[INFO]: LiDAR sensor {'ENABLED' if args_cli.enable_lidar else 'DISABLED'}")
    print(f"[INFO]: Will run {args_cli.benchmark_steps} steps for timing")
    
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # load level policy
    policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    file_bytes = read_file(policy_path)
    # jit load the policy
    policy = torch.jit.load(file_bytes).to(env.device).eval()

    # Reset environment
    print("[INFO]: Resetting environment...")
    reset_start = time.time()
    obs, _ = env.reset()
    reset_time = time.time() - reset_start
    print(f"[INFO]: Environment reset took {reset_time:.3f} seconds")
    
    # Print sensor information
    if args_cli.enable_lidar and hasattr(env.scene, "lidar_sensor"):
        lidar_sensor = env.scene["lidar_sensor"]
        print(f"[INFO]: LiDAR sensor with {lidar_sensor.num_rays} rays per environment")
        print(f"[INFO]: Total rays across all environments: {lidar_sensor.num_rays * args_cli.num_envs}")
    
    # Warmup
    print("[INFO]: Running warmup...")
    for _ in range(10):
        with torch.inference_mode():
            action = policy(obs["policy"])
            obs, _ = env.step(action)
    
    # Benchmark
    print(f"[INFO]: Starting benchmark for {args_cli.benchmark_steps} steps...")
    step_times = []
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    benchmark_start = time.time()
    
    for step in range(args_cli.benchmark_steps):
        step_start = time.time()
        
        with torch.inference_mode():
            # infer action
            action = policy(obs["policy"])
            # step env
            obs, _ = env.step(action)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_end = time.time()
        step_times.append(step_end - step_start)
        
        if (step + 1) % 100 == 0:
            avg_time = sum(step_times[-100:]) / 100
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"[INFO]: Step {step + 1}/{args_cli.benchmark_steps}, "
                  f"Avg time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
    
    benchmark_end = time.time()
    total_time = benchmark_end - benchmark_start
    
    # Results
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"LiDAR sensor: {'ENABLED' if args_cli.enable_lidar else 'DISABLED'}")
    if args_cli.enable_lidar and hasattr(env.scene, "lidar_sensor"):
        lidar_sensor = env.scene["lidar_sensor"]
        total_rays = lidar_sensor.num_rays * args_cli.num_envs
        print(f"Total rays: {total_rays:,}")
    print(f"Total steps: {args_cli.benchmark_steps}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average step time: {avg_step_time*1000:.2f} ms")
    print(f"Min step time: {min_step_time*1000:.2f} ms")
    print(f"Max step time: {max_step_time*1000:.2f} ms")
    print(f"Average FPS: {fps:.1f}")
    print(f"Environment throughput: {fps * args_cli.num_envs:.1f} env-steps/sec")
    print("="*80)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
