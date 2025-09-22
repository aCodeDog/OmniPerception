# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
import trimesh
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # the warp meshes used for raycasting.
        self.meshes: dict[str, wp.Mesh] = {}
        # Dynamic mesh support - additional variables for efficient updates
        self.combined_mesh: wp.Mesh | None = None
        self.all_mesh_view: XFormPrim | None = None
        self.all_base_points: torch.Tensor | None = None
        self.vertex_counts_per_instance: torch.Tensor | None = None
        self.mesh_instance_indices: torch.Tensor | None = None

        # Environment dynamic meshes (per-environment prims like Cube/Sphere)
        self.all_env_dynamic_mesh_view: XFormPrim | None = None
        self.env_dynamic_mesh: wp.Mesh | None = None
        self.env_base_points: torch.Tensor | None = None
        self.env_vertex_counts_per_instance: torch.Tensor | None = None
        self.env_mesh_instance_indices: torch.Tensor | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)
        # resample the drift
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the height drift
        r = torch.empty(num_envs_ids, device=self.device)
        self.ray_cast_drift[env_ids, 0] = r.uniform_(*self.cfg.ray_cast_drift_range["x"])
        self.ray_cast_drift[env_ids, 1] = r.uniform_(*self.cfg.ray_cast_drift_range["y"])
        self.ray_cast_drift[env_ids, 2] = r.uniform_(*self.cfg.ray_cast_drift_range["z"])

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_enhanced_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()
        # initialize dynamic environment meshes if configured
        self._initialize_env_dynamic_meshes()

    def _initialize_warp_meshes(self):
        # Enhanced to support multiple geometry types by combining them into one
        # Support both original single mesh path and automatic discovery of all geometries in /World/*
        
        combined_points = []
        combined_indices = []
        vertex_offset = 0
        total_meshes_found = 0
        
        # Define supported geometry types
        supported_geometry_types = ["Mesh", "Plane", "Sphere", "Cube", "Cylinder", "Capsule", "Cone"]
        
        # Check if we should discover meshes automatically or use provided paths
            # Explicit paths mode: process each provided mesh path and find all instances
        omni.log.info(f"Processing {len(self.cfg.mesh_prim_paths)} explicit mesh paths for ray casting...")
        
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            omni.log.info(f"Processing mesh path: {mesh_prim_path}")
            
            # Check if the prim path exists before processing
            if not sim_utils.find_first_matching_prim(mesh_prim_path):
                omni.log.warn(f"Mesh prim path does not exist: {mesh_prim_path} - skipping.")
                continue
            
            # Find all supported geometry prims under this path
            all_geometry_prims = []
            for geom_type in supported_geometry_types:
                prims = sim_utils.get_all_matching_child_prims(
                    mesh_prim_path, 
                    lambda prim, gt=geom_type: prim.GetTypeName() == gt
                )
                all_geometry_prims.extend(prims)

            # If no geometry prims found directly, try to find exact match
            if len(all_geometry_prims) == 0:
                # Try to get exact prim
                exact_prim = sim_utils.find_first_matching_prim(mesh_prim_path)
                if exact_prim and exact_prim.IsValid() and exact_prim.GetTypeName() in supported_geometry_types:
                    all_geometry_prims = [exact_prim]
            
            # Process all found geometry prims using unified approach
            meshes_for_this_path = 0
            for geom_prim in all_geometry_prims:
                mesh_data = self._extract_mesh_data_from_prim(geom_prim)
                if mesh_data is not None:
                    points, indices = mesh_data
                    
                    # Add vertex offset to indices for combining meshes
                    offset_indices = indices + vertex_offset
                    
                    # Add to combined arrays
                    combined_points.append(points)
                    combined_indices.append(offset_indices)
                    vertex_offset += len(points)
                    total_meshes_found += 1
                    meshes_for_this_path += 1
                    
                    prim_path_str = geom_prim.GetPath().pathString
                    omni.log.info(f"Added {geom_prim.GetTypeName()}: {prim_path_str} with {len(points)} vertices, {len(indices)} faces, transform applied.")
            
            omni.log.info(f"Found {meshes_for_this_path} geometries for path: {mesh_prim_path}")
        
        # Create combined mesh if we found any meshes
        if total_meshes_found > 0:
            # Combine all points and indices
            final_points = np.vstack(combined_points)
            final_indices = np.concatenate(combined_indices)
                        
            # Create single warp mesh from combined data
            wp_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
            
            # Store combined mesh with a standard key
            combined_mesh_key = self.cfg.mesh_prim_paths[0]
            self.meshes[combined_mesh_key] = wp_mesh
            
            omni.log.info(f"Successfully combined {total_meshes_found} meshes into single mesh for ray casting.")
            omni.log.info(f"Combined mesh has {len(final_points)} vertices and {len(final_indices)} faces.")
        else:
            # Fallback: create a default ground plane if no meshes found
            omni.log.warn("No meshes found for ray-casting! Creating default ground plane.")
            plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(plane_mesh.vertices, plane_mesh.faces, device=self.device)
            self.meshes["default_ground"] = wp_mesh

    def _initialize_enhanced_warp_meshes(self):
        """Enhanced mesh initialization that supports dynamic meshes.
        
        This method first calls the original mesh initialization, then sets up
        the dynamic mesh tracking system if dynamic mesh support is needed.
        """
        # First call the original method to maintain all existing functionality
        self._initialize_warp_meshes()
        
        # Now add dynamic mesh support if meshes were found
        if len(self.meshes) > 0:
            self._setup_dynamic_mesh_system()

    def _resolve_env_regex_ns(self) -> str:
        """Resolve environment regex namespace from sensor prim path.

        Returns the substring ending with '/env_.*' if present, otherwise '/World/envs/env_.*'.
        """
        try:
            result = re.search(r"(.*/envs/env_\\\.\*)", self.cfg.prim_path)
            if result:
                return result.group(1)
        except Exception:
            pass
        return "/World/envs/env_.*"

    def _initialize_env_dynamic_meshes(self):
        """Initialize the env dynamic mesh system using parent XFormPrims for transforms.

        New logic (requested):
        1. Use the configured `dynamic_env_mesh_prim_paths` (each points to a mesh prim) and resolve them for `env_0` only.
        2. For each such mesh prim (env_0 instance), obtain its parent XFormPrim path. The parent will be used for world pose updates.
        3. Replace the `env_0` portion with the regex namespace (e.g. `env_.*`) to build pattern parent paths and create a single XFormPrim view.
        4. Extract geometry ONCE per pattern from the env_0 mesh prim, transform vertices into the parent frame (apply relative transform mesh->parent),
           then replicate that geometry across all environments while building a combined warp mesh.
        5. Maintain view ordering assumption: pattern0 [env0..envN], pattern1 [env0..envN], ... so the existing efficient update code works unchanged.
        """
        try:
            # Quick exits
            if not hasattr(self.cfg, "dynamic_env_mesh_prim_paths"):
                return
            if not self.cfg.dynamic_env_mesh_prim_paths:
                return

            from pxr import UsdGeom

            env_regex_ns = self._resolve_env_regex_ns()  # e.g. /World/envs/env_.*
            # Derive a concrete env_0 namespace to sample representative meshes
            # Handle both 'env_.*' and escaped variants 'env_\.*'
            env0_ns = re.sub(r"env_\\\.\*", "env_0", env_regex_ns)
            env0_ns = env0_ns.replace("env_.*", "env_0")

            parent_pattern_paths: list[str] = []  # parent paths with regex namespace
            pattern_points: list[np.ndarray] = []  # geometry (points) in parent frame for each pattern
            pattern_indices: list[np.ndarray] = []  # flattened triangle indices for each pattern

            # Helper: convert mesh prim geometry into parent frame
            def _mesh_vertices_in_parent_frame(mesh_prim) -> tuple[np.ndarray, np.ndarray] | None:
                mesh_data = self._extract_mesh_data_from_prim_dynamic(mesh_prim)
                if mesh_data is None:
                    return None
                points, indices = mesh_data  # points in mesh local frame
                try:
                    mesh_xf = UsdGeom.Xformable(mesh_prim)
                    parent_prim = mesh_prim.GetParent()
                    parent_xf = UsdGeom.Xformable(parent_prim)
                    mesh_world = np.array(mesh_xf.ComputeLocalToWorldTransform(0.0))  # 4x4
                    parent_world = np.array(parent_xf.ComputeLocalToWorldTransform(0.0))
                    parent_world_inv = np.linalg.inv(parent_world)
                    rel = parent_world_inv @ mesh_world  # parent->mesh
                    # Apply relative transform to mesh-local points to express them in parent frame
                    rot = rel[:3, :3]
                    trans = rel[:3, 3]
                    points_parent = points @ rot.T + trans
                    return points_parent.astype(np.float32), indices.astype(np.int32)
                except Exception as e:
                    omni.log.warn(f"Failed relative transform for mesh {mesh_prim.GetPath()}: {e}")
                    return points.astype(np.float32), indices.astype(np.int32)

            # 1 & 2. Resolve each dynamic path for env_0, fetch mesh prim + parent
            # Access stage directly for deterministic prim retrieval
            stage = omni.usd.get_context().get_stage()
            supported_geom_types = {"Mesh", "Sphere", "Cube", "Cylinder", "Capsule", "Cone", "Plane"}

            for raw_path in self.cfg.dynamic_env_mesh_prim_paths:
                # Build env_0 concrete path from pattern
                try:
                    mesh_path_env0 = raw_path.format(ENV_REGEX_NS=env0_ns)
                except Exception:
                    mesh_path_env0 = raw_path.replace(env_regex_ns, env0_ns)

                mesh_prim = stage.GetPrimAtPath(mesh_path_env0)
                if not mesh_prim or not mesh_prim.IsValid():
                    omni.log.warn(f"Env0 mesh prim path not found: {mesh_path_env0} (skipping pattern)")
                    continue

                geom_type = mesh_prim.GetTypeName()
                if geom_type not in supported_geom_types:
                    omni.log.warn(
                        f"Env0 mesh prim at {mesh_path_env0} has unsupported type '{geom_type}' (skipping pattern)"
                    )
                    continue

                parent_prim = mesh_prim.GetParent()
                if parent_prim is None or not parent_prim.IsValid():
                    omni.log.warn(f"Parent prim invalid for mesh {mesh_path_env0} (skipping pattern)")
                    continue

                parent_env0_path = parent_prim.GetPath().pathString
                parent_pattern_path = parent_env0_path.replace(env0_ns, env_regex_ns)
                parent_pattern_paths.append(parent_pattern_path)

                mesh_parent_frame = _mesh_vertices_in_parent_frame(mesh_prim)
                if mesh_parent_frame is None:
                    omni.log.warn(f"Failed to extract geometry (parent frame) for {mesh_path_env0}")
                    parent_pattern_paths.pop()
                    continue
                pts_parent, idx_parent = mesh_parent_frame
                pattern_points.append(pts_parent)
                pattern_indices.append(idx_parent)

            if not parent_pattern_paths:
                omni.log.warn("No valid dynamic env mesh parent patterns resolved.")
                return

            # 5. Create XFormPrim view over parent patterns (expands env instances)
            self.all_env_dynamic_mesh_view = XFormPrim(parent_pattern_paths, reset_xform_properties=False)
            prim_paths_expanded = list(self.all_env_dynamic_mesh_view.prim_paths)
            if len(prim_paths_expanded) == 0:
                omni.log.warn("Parent pattern XFormPrim expansion yielded no prim paths.")
                self.all_env_dynamic_mesh_view = None
                return

            num_patterns = len(parent_pattern_paths)
            if len(prim_paths_expanded) % num_patterns != 0:
                omni.log.warn(
                    "Expanded parent prim count not divisible by number of patterns. Aborting env dynamic mesh setup."
                )
                self.all_env_dynamic_mesh_view = None
                return
            num_envs = len(prim_paths_expanded) // num_patterns

            # Replicate each pattern's geometry across environments; maintain ordering assumption
            combined_points: list[np.ndarray] = []
            combined_indices: list[np.ndarray] = []
            vertex_counts: list[int] = []
            vertex_offset = 0
            total_instances = 0

            for p in range(num_patterns):
                base_points = pattern_points[p]
                base_indices = pattern_indices[p]
                for _env in range(num_envs):
                    offset_indices = base_indices + vertex_offset
                    combined_points.append(base_points)
                    combined_indices.append(offset_indices)
                    vertex_counts.append(len(base_points))
                    vertex_offset += len(base_points)
                    total_instances += 1

            if total_instances == 0:
                omni.log.warn("After replication no env dynamic mesh instances were assembled.")
                self.all_env_dynamic_mesh_view = None
                return

            final_points = np.vstack(combined_points)
            final_indices = np.concatenate(combined_indices)

            # Build warp mesh and cache tensors for fast updates
            self.env_dynamic_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
            self.env_base_points = torch.tensor(final_points, device=self.device, dtype=torch.float32)
            self.env_vertex_counts_per_instance = torch.tensor(vertex_counts, device=self.device, dtype=torch.int32)
            self.env_mesh_instance_indices = torch.arange(total_instances, device=self.device, dtype=torch.int32)

            omni.log.info(
                "Initialized env dynamic mesh (parent-based): patterns=%d, envs=%d, instances=%d, vertices=%d, faces=%d" % (
                    num_patterns,
                    num_envs,
                    total_instances,
                    len(final_points),
                    len(final_indices) // 3,
                )
            )
        except Exception as e:
            omni.log.error(f"Failed to initialize env dynamic meshes (parent-based): {e}")
            self.all_env_dynamic_mesh_view = None
            self.env_dynamic_mesh = None
            self.env_base_points = None
            self.env_vertex_counts_per_instance = None
            self.env_mesh_instance_indices = None
    
    def _setup_dynamic_mesh_system(self):
        """Set up the dynamic mesh tracking system for efficient updates."""
        # Data structures for efficient mesh combination
        all_mesh_prim_paths = []
        combined_points = []
        combined_indices = []
        vertex_counts = []
        vertex_offset = 0
        total_meshes_found = 0
        
        # Define supported geometry types
        supported_geometry_types = ["Mesh", "Plane", "Sphere", "Cube", "Cylinder", "Capsule", "Cone"]
        
        omni.log.info("Setting up dynamic mesh system for ray caster...")
        
        # Process each mesh path to extract geometry data for dynamic tracking
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # Check if the prim path exists before processing
            if not sim_utils.find_first_matching_prim(mesh_prim_path):
                continue
            
            # Find all supported geometry prims under this path
            all_geometry_prims = []
            for geom_type in supported_geometry_types:
                prims = sim_utils.get_all_matching_child_prims(
                    mesh_prim_path, 
                    lambda prim, gt=geom_type: prim.GetTypeName() == gt
                )
                all_geometry_prims.extend(prims)

            # If no geometry prims found directly, try to find exact match
            if len(all_geometry_prims) == 0:
                exact_prim = sim_utils.find_first_matching_prim(mesh_prim_path)
                if exact_prim and exact_prim.IsValid() and exact_prim.GetTypeName() in supported_geometry_types:
                    all_geometry_prims = [exact_prim]
            
            # Process all found geometry prims
            for geom_prim in all_geometry_prims:
                mesh_data = self._extract_mesh_data_from_prim_dynamic(geom_prim)
                if mesh_data is not None:
                    points, indices = mesh_data
                    
                    # Add vertex offset to indices for combining meshes
                    offset_indices = indices + vertex_offset
                    
                    # Add to combined arrays
                    combined_points.append(points)
                    combined_indices.append(offset_indices)
                    vertex_counts.append(len(points))
                    vertex_offset += len(points)
                    total_meshes_found += 1
                    
                    # Store prim path for XFormPrim view creation
                    all_mesh_prim_paths.append(geom_prim.GetPath().pathString)
        
        # Create the dynamic mesh system if we found meshes
        if total_meshes_found > 0:
            try:
                # Combine all points and indices
                final_points = np.vstack(combined_points)
                final_indices = np.concatenate(combined_indices)
                
                # Create single warp mesh from combined data
                self.combined_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
                
                # Create single XFormPrim view for all meshes
                self.all_mesh_view = XFormPrim(all_mesh_prim_paths, reset_xform_properties=False)
                
                # Setup efficient vectorized update system
                self.all_base_points = torch.tensor(final_points, device=self.device, dtype=torch.float32)
                self.vertex_counts_per_instance = torch.tensor(vertex_counts, device=self.device, dtype=torch.int32)
                
                # Create mesh instance indices for mapping
                self.mesh_instance_indices = torch.arange(total_meshes_found, device=self.device, dtype=torch.int32)
                
                omni.log.info(f"Successfully set up dynamic mesh system:")
                omni.log.info(f"  - Tracking {total_meshes_found} mesh instances")
                omni.log.info(f"  - Total vertices: {len(final_points)}")
                omni.log.info(f"  - Total faces: {len(final_indices) // 3}")
                
            except Exception as e:
                omni.log.warn(f"Failed to setup dynamic mesh system: {str(e)}")
                # Reset dynamic mesh variables on failure
                self.combined_mesh = None
                self.all_mesh_view = None
                self.all_base_points = None
                self.vertex_counts_per_instance = None
                self.mesh_instance_indices = None

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift to ray starting position in world frame
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # Update dynamic meshes if available
        if self.combined_mesh is not None:
            self._update_combined_mesh_efficiently()
        if self.env_dynamic_mesh is not None:
            self._update_env_dynamic_mesh_efficiently()

        # ray cast based on the sensor poses
        if self.cfg.ray_alignment == "world":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            # no rotation is considered and directions are not rotated
            ray_starts_w = self.ray_starts[env_ids]
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw" or self.cfg.attach_yaw_only:
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                omni.log.warn(
                    "The `attach_yaw_only` property will be deprecated in a future release. Please use"
                    " `ray_alignment='yaw'` instead."
                )

            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            # apply horizontal drift to ray starting position in ray caster frame
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ray cast and store the hits (dual raycast with distance merge)
        # Use combined mesh if available for dynamic support, otherwise use original mesh
        if self.combined_mesh is not None:
            mesh_to_use = self.combined_mesh
        else:
            mesh_to_use = self.meshes[self.cfg.mesh_prim_paths[0]]

        _, dist1, _, _ = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=mesh_to_use,
            return_distance=True,
        )

        if self.env_dynamic_mesh is not None:
            _, dist2, _, _ = raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=self.env_dynamic_mesh,
                return_distance=True,
            )
            final_dist = torch.minimum(dist1, dist2)
        else:
            final_dist = dist1

        # Compute final hit points. For missed rays, final_dist stays inf and the result becomes inf as well.
        final_hits = ray_starts_w + final_dist.unsqueeze(-1) * ray_directions_w
        self._data.ray_hits_w[env_ids] = final_hits

        # apply vertical drift to ray starting position in ray caster frame
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # remove possible inf values
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        # show ray hit positions
        self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
        # Invalidate dynamic mesh view as well
        self.all_mesh_view = None
        # Invalidate environment dynamic mesh view as well
        self.all_env_dynamic_mesh_view = None

    def _create_trimesh_from_usd_primitive(self, geom_prim, geom_type):
        """Create a trimesh object from USD primitive parameters.
        
        Args:
            geom_prim: USD geometry primitive 
            geom_type: Type of the primitive (Sphere, Cube, Cylinder, Capsule, Cone)
            
        Returns:
            trimesh.Trimesh object or None if creation failed
        """
        try:
            import trimesh
            from pxr import UsdGeom
            
            if geom_type == "Sphere":
                # Get sphere parameters
                sphere_geom = UsdGeom.Sphere(geom_prim)
                radius_attr = sphere_geom.GetRadiusAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                
                # Create trimesh sphere
                return trimesh.creation.uv_sphere(radius=radius)
                
            elif geom_type == "Cube":
                # Get cube parameters (size attribute)
                cube_geom = UsdGeom.Cube(geom_prim)
                size_attr = cube_geom.GetSizeAttr()
                size = size_attr.Get() if size_attr else 2.0  # USD Cube default size is 2.0
                
                # Create trimesh box
                return trimesh.creation.box(extents=[size, size, size])
                
            elif geom_type == "Cylinder":
                # Get cylinder parameters
                cylinder_geom = UsdGeom.Cylinder(geom_prim)
                radius_attr = cylinder_geom.GetRadiusAttr()
                height_attr = cylinder_geom.GetHeightAttr()
                axis_attr = cylinder_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh cylinder
                return trimesh.creation.cylinder(radius=radius, height=height, transform=transform)
                
            elif geom_type == "Capsule":
                # Get capsule parameters
                capsule_geom = UsdGeom.Capsule(geom_prim)
                radius_attr = capsule_geom.GetRadiusAttr()
                height_attr = capsule_geom.GetHeightAttr()
                axis_attr = capsule_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh capsule
                return trimesh.creation.capsule(radius=radius, height=height, transform=transform)
                
            elif geom_type == "Cone":
                # Get cone parameters
                cone_geom = UsdGeom.Cone(geom_prim)
                radius_attr = cone_geom.GetRadiusAttr()
                height_attr = cone_geom.GetHeightAttr()
                axis_attr = cone_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh cone
                return trimesh.creation.cone(radius=radius, height=height, transform=transform)
                
            else:
                omni.log.warn(f"Unsupported primitive type for trimesh creation: {geom_type}")
                return None
                
        except Exception as e:
            omni.log.warn(f"Failed to create trimesh for {geom_type}: {str(e)}")
            return None

    def _extract_mesh_data_from_prim(self, geom_prim):
        """Extract mesh data from any supported USD geometry primitive.
        
        Args:
            geom_prim: USD geometry primitive (Mesh, Plane, Sphere, Cube, Cylinder, Capsule, Cone)
            
        Returns:
            tuple: (points, indices) as numpy arrays, or None if extraction failed
        """
        from pxr import UsdGeom
        try:
            geom_type = geom_prim.GetTypeName()

            if geom_type == "Plane":
                # Handle Plane using make_plane utility (keeps existing logic)
                plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                
                # Apply world transformation to plane
                try:
                    from pxr import UsdGeom
                    xform = UsdGeom.Xformable(geom_prim)
                    if xform:
                        transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                    else:
                        transform_matrix = np.eye(4)
                except:
                    transform_matrix = np.eye(4)
                    
                points = np.matmul(plane_mesh.vertices, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = plane_mesh.faces.flatten()
                
                return points, indices
                
            else:
                # Handle all other geometry types (Mesh, Sphere, Cube, Cylinder, Capsule, Cone)
                from pxr import UsdGeom

                if geom_type == "Mesh":
                    # Direct mesh access
                    mesh_geom = UsdGeom.Mesh(geom_prim)
                    points_attr = mesh_geom.GetPointsAttr()
                    face_indices_attr = mesh_geom.GetFaceVertexIndicesAttr()
                    face_counts_attr = mesh_geom.GetFaceVertexCountsAttr()
                    
                    if not (points_attr and face_indices_attr and face_counts_attr):
                        omni.log.warn(f"Could not find mesh attributes for {geom_type}: {geom_prim.GetPath()}")
                        return None
                        
                    # Get the actual data
                    points_data = points_attr.Get()
                    faces_data = face_indices_attr.Get()
                    face_counts_data = face_counts_attr.Get()
                    
                    if points_data is None or faces_data is None or face_counts_data is None:
                        omni.log.warn(f"Mesh attribute data is None for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    points = list(points_data)
                    points = [np.ravel(x) for x in points]
                    points = np.array(points)
                    
                    if len(points) == 0:
                        omni.log.warn(f"Empty points array for {geom_type}: {geom_prim.GetPath()}")
                        return None
                        
                    faces = list(faces_data)
                    face_counts = list(face_counts_data)
                    
                    if len(faces) == 0 or len(face_counts) == 0:
                        omni.log.warn(f"Empty faces/face_counts for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    # Check if triangulation is needed
                    if not all(count == 3 for count in face_counts):
                        omni.log.info(f"Triangulating {geom_type} {geom_prim.GetPath()} - found faces with {set(face_counts)} vertices")
                        faces = self._triangulate_faces_from_list(faces, face_counts)
                    
                    # Convert to proper triangle format
                    triangulated_indices = np.array(faces)
                    
                    # Apply world transformation
                    try:
                        xform = UsdGeom.Xformable(geom_prim)
                        if xform:
                            transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                        else:
                            transform_matrix = np.eye(4)
                    except:
                        transform_matrix = np.eye(4)
                        
                    points = np.matmul(points, transform_matrix[:3, :3].T)
                    points += transform_matrix[:3, 3]
                    
                    return points, triangulated_indices
                    
                else:
                    # Handle primitive shapes (Sphere, Cube, Cylinder, Capsule, Cone) using trimesh
                    trimesh_mesh = self._create_trimesh_from_usd_primitive(geom_prim, geom_type)
                    
                    if trimesh_mesh is None:
                        omni.log.warn(f"Failed to create trimesh for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    # Get scale from USD prim attribute
                    try:
                        path = geom_prim.GetPath().pathString
                        prim = prim_utils.get_prim_at_path(path)
                        scale_attr = prim.GetAttribute("xformOp:scale")
                        if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                            scale = tuple(scale_attr.Get())
                            # Apply scale to trimesh object if scale is not uniform [1,1,1]
                            if not all(abs(s - 1.0) < 1e-6 for s in scale):
                                trimesh_mesh.apply_scale(scale)
                    except Exception as e:
                        omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                    
                    # Apply world transformation (rotation and translation)
                    try:
                        xform = UsdGeom.Xformable(geom_prim)
                        if xform:
                            transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                        else:
                            transform_matrix = np.eye(4)
                    except:
                        transform_matrix = np.eye(4)
                        
                    # Transform mesh vertices to world coordinates
                    points = np.matmul(trimesh_mesh.vertices, transform_matrix[:3, :3].T)
                    points += transform_matrix[:3, 3]
                    indices = trimesh_mesh.faces.flatten()
                    
                    return points, indices
                
        except Exception as e:
            omni.log.warn(f"Failed to extract mesh data from {geom_prim.GetTypeName()} {geom_prim.GetPath()}: {str(e)}")
            return None

    def _triangulate_faces_from_list(self, faces: list, face_counts: list) -> list:
        """Convert polygonal faces to triangles using list format.
        
        Args:
            faces: Flattened list of face vertex indices
            face_counts: List containing number of vertices per face
            
        Returns:
            Triangulated face indices as flat list
        """
        triangulated_faces = []
        face_idx = 0
        
        for count in face_counts:
            if count == 3:
                # Already a triangle
                triangulated_faces.extend(faces[face_idx:face_idx + 3])
            elif count == 4:
                # Quad to two triangles
                v0, v1, v2, v3 = faces[face_idx:face_idx + 4]
                triangulated_faces.extend([v0, v1, v2])  # First triangle
                triangulated_faces.extend([v0, v2, v3])  # Second triangle
            else:
                # General polygon triangulation (fan triangulation)
                v0 = faces[face_idx]
                for i in range(1, count - 1):
                    v1 = faces[face_idx + i]
                    v2 = faces[face_idx + i + 1]
                    triangulated_faces.extend([v0, v1, v2])
            
            face_idx += count
        
        return triangulated_faces

    def _extract_mesh_data_from_prim_dynamic(self, geom_prim):
        """Extract mesh data from USD geometry primitive for dynamic tracking.
        
        Args:
            geom_prim: USD geometry primitive
            
        Returns:
            tuple: (points, indices) as numpy arrays in local coordinates, or None if extraction failed
        """
        from pxr import UsdGeom
        try:
            geom_type = geom_prim.GetTypeName()

            if geom_type == "Plane":
                # Create a simple plane in local coordinates
                plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                
                # Apply scale if present in the USD prim attribute
                try:
                    path = geom_prim.GetPath().pathString
                    prim = prim_utils.get_prim_at_path(path)
                    scale_attr = prim.GetAttribute("xformOp:scale")
                    if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                        scale = tuple(scale_attr.Get())
                        # Apply scale to plane vertices if scale is not uniform [1,1,1]
                        if not all(abs(s - 1.0) < 1e-6 for s in scale):
                            vertices_scaled = plane_mesh.vertices * np.array(scale)
                            return vertices_scaled, plane_mesh.faces.flatten()
                except Exception as e:
                    omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                
                return plane_mesh.vertices, plane_mesh.faces.flatten()
                
            elif geom_type == "Mesh":
                # Direct mesh access
                mesh_geom = UsdGeom.Mesh(geom_prim)
                points_attr = mesh_geom.GetPointsAttr()
                face_indices_attr = mesh_geom.GetFaceVertexIndicesAttr()
                face_counts_attr = mesh_geom.GetFaceVertexCountsAttr()
                
                if not (points_attr and face_indices_attr and face_counts_attr):
                    return None
                    
                points_data = points_attr.Get()
                faces_data = face_indices_attr.Get()
                face_counts_data = face_counts_attr.Get()
                
                if points_data is None or faces_data is None or face_counts_data is None:
                    return None
                
                points = np.array([[x[0], x[1], x[2]] for x in points_data])
                faces = list(faces_data)
                face_counts = list(face_counts_data)
                
                if len(points) == 0 or len(faces) == 0:
                    return None
                
                # Triangulate if needed
                if not all(count == 3 for count in face_counts):
                    faces = self._triangulate_faces_from_list_dynamic(faces, face_counts)
                
                # Apply scale if present in the USD prim attribute
                try:
                    path = geom_prim.GetPath().pathString
                    prim = prim_utils.get_prim_at_path(path)
                    scale_attr = prim.GetAttribute("xformOp:scale")
                    if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                        scale = tuple(scale_attr.Get())
                        # Apply scale to mesh vertices if scale is not uniform [1,1,1]
                        if not all(abs(s - 1.0) < 1e-6 for s in scale):
                            points_scaled = points * np.array(scale)
                            return points_scaled, np.array(faces)
                except Exception as e:
                    omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                
                return points, np.array(faces)
                
            else:
                # Handle primitive shapes using trimesh in local coordinates with scale applied
                trimesh_mesh = self._create_trimesh_from_usd_primitive_dynamic(geom_prim, geom_type)
                if trimesh_mesh is not None:
                    # Get scale from USD prim attribute
                    try:
                        path = geom_prim.GetPath().pathString
                        prim = prim_utils.get_prim_at_path(path)
                        scale_attr = prim.GetAttribute("xformOp:scale")
                        if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                            scale = tuple(scale_attr.Get())
                            # Apply scale to trimesh object if scale is not uniform [1,1,1]
                            if not all(abs(s - 1.0) < 1e-6 for s in scale):
                                trimesh_mesh.apply_scale(scale)
                    except Exception as e:
                        omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                    
                    return trimesh_mesh.vertices, trimesh_mesh.faces.flatten()
                    
        except Exception as e:
            omni.log.warn(f"Failed to extract dynamic mesh data from {geom_prim.GetTypeName()} {geom_prim.GetPath()}: {str(e)}")
            return None

    def _create_trimesh_from_usd_primitive_dynamic(self, geom_prim, geom_type):
        """Create a trimesh object from USD primitive parameters in local coordinates.
        
        Args:
            geom_prim: USD geometry primitive 
            geom_type: Type of the primitive
            
        Returns:
            trimesh.Trimesh object or None if creation failed
        """
        try:
            from pxr import UsdGeom
            
            if geom_type == "Sphere":
                sphere_geom = UsdGeom.Sphere(geom_prim)
                radius_attr = sphere_geom.GetRadiusAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                return trimesh.creation.uv_sphere(radius=radius)
                
            elif geom_type == "Cube":
                cube_geom = UsdGeom.Cube(geom_prim)
                size_attr = cube_geom.GetSizeAttr()
                size = size_attr.Get() if size_attr else 2.0
                return trimesh.creation.box(extents=[size, size, size])
                
            elif geom_type == "Cylinder":
                cylinder_geom = UsdGeom.Cylinder(geom_prim)
                radius_attr = cylinder_geom.GetRadiusAttr()
                height_attr = cylinder_geom.GetHeightAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                return trimesh.creation.cylinder(radius=radius, height=height)
                
            else:
                return None
                
        except Exception as e:
            omni.log.warn(f"Failed to create dynamic trimesh for {geom_type}: {str(e)}")
            return None

    def _triangulate_faces_from_list_dynamic(self, faces: list, face_counts: list) -> list:
        """Convert polygonal faces to triangles for dynamic meshes."""
        triangulated_faces = []
        face_idx = 0
        
        for count in face_counts:
            if count == 3:
                triangulated_faces.extend(faces[face_idx:face_idx + 3])
            elif count == 4:
                v0, v1, v2, v3 = faces[face_idx:face_idx + 4]
                triangulated_faces.extend([v0, v1, v2, v0, v2, v3])
            else:
                # Fan triangulation for polygons
                for i in range(1, count - 1):
                    triangulated_faces.extend([faces[face_idx], faces[face_idx + i], faces[face_idx + i + 1]])
            face_idx += count
        
        return triangulated_faces

    def _update_combined_mesh_efficiently(self):
        """Efficiently update the combined mesh using vectorized operations."""
        if (self.all_mesh_view is None or self.combined_mesh is None or 
            self.mesh_instance_indices is None or self.vertex_counts_per_instance is None or
            self.all_base_points is None):
            return
            
        try:
            # Get current world poses for all mesh instances
            num_mesh_instances = len(self.mesh_instance_indices)
            current_poses, current_quats = self.all_mesh_view.get_world_poses(
                torch.arange(num_mesh_instances, device=self.device)
            )
            
            # Convert to torch tensors if needed
            if isinstance(current_poses, np.ndarray):
                current_poses = torch.from_numpy(current_poses).to(device=self.device)
            if isinstance(current_quats, np.ndarray):
                current_quats = torch.from_numpy(current_quats).to(device=self.device)
            
            # Expand current poses and quats to vertex level
            expanded_positions = torch.repeat_interleave(
                current_poses, 
                self.vertex_counts_per_instance.long(), 
                dim=0
            )
            
            expanded_quats = torch.repeat_interleave(
                current_quats,
                self.vertex_counts_per_instance.long(), 
                dim=0
            )
            
            # Apply world transform: quat_apply(mesh_quat, base_points) + mesh_pos
            transformed_points = quat_apply(expanded_quats, self.all_base_points) + expanded_positions
            
            # Update the warp mesh with the new transformed points
            updated_points_wp = wp.from_torch(transformed_points, dtype=wp.vec3)
            self.combined_mesh.points = updated_points_wp
            self.combined_mesh.refit()
            
        except Exception as e:
            omni.log.warn(f"Failed to update combined mesh efficiently: {str(e)}")

    def _update_env_dynamic_mesh_efficiently(self):
        """Efficiently update the env dynamic mesh using vectorized operations."""
        if (
            self.all_env_dynamic_mesh_view is None
            or self.env_dynamic_mesh is None
            or self.env_mesh_instance_indices is None
            or self.env_vertex_counts_per_instance is None
            or self.env_base_points is None
        ):
            return

        try:
            num_instances = len(self.env_mesh_instance_indices)
            current_poses, current_quats = self.all_env_dynamic_mesh_view.get_world_poses(
                torch.arange(num_instances, device=self.device)
            )

            if isinstance(current_poses, np.ndarray):
                current_poses = torch.from_numpy(current_poses).to(device=self.device)
            if isinstance(current_quats, np.ndarray):
                current_quats = torch.from_numpy(current_quats).to(device=self.device)

            expanded_positions = torch.repeat_interleave(
                current_poses, self.env_vertex_counts_per_instance.long(), dim=0
            )
            expanded_quats = torch.repeat_interleave(
                current_quats, self.env_vertex_counts_per_instance.long(), dim=0
            )

            transformed_points = quat_apply(expanded_quats, self.env_base_points) + expanded_positions

            updated_points_wp = wp.from_torch(transformed_points, dtype=wp.vec3)
            self.env_dynamic_mesh.points = updated_points_wp
            self.env_dynamic_mesh.refit()

        except Exception as e:
            omni.log.warn(f"Failed to update env dynamic mesh efficiently: {str(e)}")
            
            
            
#stage = omni.usd.get_context().get_stage()

#mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/LF_HIP/visuals/mesh_0")
#mesh_prim.GetTypeName() == "Mesh"? 
#Mesh = UsdGeom.Mesh.Get(stage,mesh_prim.GetPath().pathString)
#mesh_geom = UsdGeom.Mesh(mesh_prim)
#mesh_prim.GetChildren()
#mesh_prim.GetParent()