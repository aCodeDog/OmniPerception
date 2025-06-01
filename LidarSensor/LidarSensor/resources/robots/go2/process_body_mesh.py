import yourdfpy
import trimesh
import numpy as np
import os

def combine_urdf_meshes_to_stl(urdf_path, output_stl_path, package_dir_map=None):
    """
    Loads a URDF, extracts all visual meshes, transforms them to their
    global pose in the zero configuration, combines them, and exports to STL.

    Args:
        urdf_path (str): Path to the URDF file.
        output_stl_path (str): Path where the combined STL file will be saved.
        package_dir_map (dict, optional): A dictionary mapping ROS package names
                                         (e.g., 'my_robot_description') to their
                                         local directory paths. Needed if your URDF
                                         uses 'package://' paths.
                                         Example: {'my_robot_description': '/path/to/my_robot_ws/src/my_robot_description'}
                                         Defaults to None.
    """
    print(f"Loading URDF: {urdf_path}")

    # Create the package directory map if provided, otherwise use an empty dict
    pkg_dirs = package_dir_map if package_dir_map else {}

    try:
        # Load the URDF. yourdfpy automatically calculates FK for the zero configuration.
        # It needs package_dir_map to resolve 'package://' paths.
        robot = yourdfpy.URDF.load(urdf_path, build_scene_graph=True, load_meshes=True)
        print("URDF loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error loading URDF or its resources: {e}")
        print("Please ensure the URDF path is correct and provide 'package_dir_map' if using 'package://' paths.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during URDF loading: {e}")
        return

    all_meshes = []
    processed_mesh_paths = set() # Keep track of processed mesh files to avoid duplicates if shared

    print("Processing visual meshes for each link...")
    # Iterate through the scene graph nodes (links) managed by yourdfpy
    print("Processing visual meshes for each link...")
        # Iterate through all links defined in the URDF using link_map
    for link_name in robot.link_map:  # Iterate through link names
        link = robot.link_map[link_name] # Get the Link object using its name

        if link.visuals: # Check if the link has visual elements
            print(f"  Processing link: {link_name}")
            try:
                # Get the global transform for this link in the zero configuration
                # Assumes the base link is the root frame if frame_b is omitted
                link_global_transform = robot.get_transform(link_name) # Get T_world_link

                if link_global_transform is None:
                    print(f"    Warning: Could not get transform for link '{link_name}'. Skipping.")
                    continue

                # Iterate through the visual elements of this link
# Iterate through the visual elements of this link
                for i, visual in enumerate(link.visuals):
                    # CORRECTED CHECK: Check if the 'mesh' attribute of the Geometry object exists (is not None)
                    if visual.geometry.mesh is not None:
                        # Now access the actual Mesh object
                        mesh_info = visual.geometry.mesh

                        # --- Mesh processing code starts here ---
                        mesh_path = mesh_info.filename
                        mesh_scale = mesh_info.scale if mesh_info.scale is not None else [1.0, 1.0, 1.0]
                        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
                        
                        # Resolve the mesh path relative to URDF directory
                        if mesh_path.startswith('package://'):
                            # Handle package:// URLs if needed
                            mesh_path = mesh_path.replace('package://', '')
                            # You might need additional logic here for package resolution
                        
                        # Create absolute path
                        full_mesh_path = os.path.join(urdf_dir, mesh_path)
                        full_mesh_path = os.path.abspath(full_mesh_path)
                        
                        print(f"      Mesh path resolved to: {full_mesh_path}")
                        #full_mesh_path = yourdfpy.utils.resolve_path(mesh_path, pkg_dirs)

                        if not os.path.exists(full_mesh_path):
                            print(f"    Warning: Mesh file not found: {full_mesh_path} (original: {mesh_path}). Skipping visual {i} for link '{link_name}'.")
                            continue

                        print(f"      Processing visual {i} (Mesh): {os.path.basename(full_mesh_path)}")

                        try:
                            # Load the mesh using trimesh
                            mesh = trimesh.load(full_mesh_path, force='mesh', use_embree=False)

                            if isinstance(mesh, trimesh.Scene):
                                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                                print(f"        (Loaded as Scene, concatenated geometry)")

                            if not isinstance(mesh, trimesh.Trimesh):
                                print(f"      Warning: Loaded object for {full_mesh_path} is not a Trimesh ({type(mesh)}). Skipping visual {i}.")
                                continue

                            # Apply scaling if specified
                            if not np.allclose(mesh_scale, [1.0, 1.0, 1.0]):
                                scale_matrix = np.diag(list(mesh_scale) + [1])
                                mesh.apply_transform(scale_matrix)
                                print(f"        Applied scale: {mesh_scale}")

                            # Get the transform from the link origin to the visual mesh origin (T_link_visual)
                            visual_origin_transform = visual.origin if visual.origin is not None else np.eye(4)

                            # Calculate the global transform for the visual mesh
                            # T_world_visual = T_world_link * T_link_visual
                            mesh_global_transform = link_global_transform @ visual_origin_transform

                            # Apply the global transform to the mesh
                            mesh.apply_transform(mesh_global_transform)

                            all_meshes.append(mesh)

                        except ValueError as e:
                            print(f"      Warning: Failed to load or process mesh {full_mesh_path}: {e}. Skipping visual {i}.")
                        except Exception as e:
                             print(f"      Warning: An unexpected error occurred processing mesh {full_mesh_path}: {e}. Skipping visual {i}.")
                        # --- Mesh processing code ends here ---
                    else:
                        # Optional: Log that we are skipping non-mesh geometry like boxes, spheres, cylinders
                        # print(f"      Skipping visual {i} for link '{link_name}' (Not a mesh geometry).")
                        pass # Simply skip visual elements that are not meshes

            except Exception as e:
                print(f"    Error getting transform or processing visuals for link {link_name}: {e}")
                continue # Skip to the next link if there's an error with this one


    if not all_meshes:
        print("No valid meshes found or processed. Cannot create combined STL.")
        return

    print(f"\nCombining {len(all_meshes)} meshes...")
    try:
        # Combine all transformed meshes into one
        combined_mesh = trimesh.util.concatenate(all_meshes)

        # Ensure the mesh is watertight if possible/needed, or handle potential issues
        if not combined_mesh.is_empty:
             # Optional: Fill holes if needed, might be slow for complex models
             # combined_mesh.fill_holes()
             # Optional: Fix winding and normals
             # combined_mesh.fix_normals(multibody=True)
             print(f"Combined mesh has {combined_mesh.faces.shape[0]} faces and {combined_mesh.vertices.shape[0]} vertices.")
        else:
            print("Warning: The combined mesh is empty.")
            return

    except Exception as e:
        print(f"Error combining meshes: {e}")
        return

    print(f"Exporting combined mesh to: {output_stl_path}")
    try:
        # Export the combined mesh to STL
        combined_mesh.export(output_stl_path, file_type='stl_ascii') # or 'stl' for binary
        print("Export complete.")
    except Exception as e:
        print(f"Error exporting STL file: {e}")

# --- How to use ---
if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your URDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_urdf_path = "urdf/go2.urdf"
    urdf_file = os.path.join(script_dir, relative_urdf_path)
    print(f"Using URDF file: {urdf_file}")
    # IMPORTANT: Replace with the desired output STL file path
    relative_output_path = "robot_combined.stl"
    output_file = os.path.join(script_dir, relative_output_path)
    print(f"Output STL file will be saved to: {output_file}")


    package_directories = None

    # Check if the URDF file exists before running
    if not os.path.exists(urdf_file):
        print(f"Error: URDF file not found at {urdf_file}")
    else:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        combine_urdf_meshes_to_stl(urdf_file, output_file)