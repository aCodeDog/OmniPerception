import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)





class Terrain_cfg(BaseConfig):
    mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    hf2mesh_method = "grid"  # grid or fast
    max_error = 0.1 # for fast
    max_error_camera = 2

    y_range = [-0.4, 0.4]
    
    edge_width_thresh = 0.05
    horizontal_scale = 0.05 # [m] influence computation time by a lot
    horizontal_scale_camera = 0.1
    vertical_scale = 0.005 # [m]
    border_size = 5 # [m]
    height = [0.02, 0.06]
    simplify_grid = False
    gap_size = [0.02, 0.1]
    stepping_stone_distance = [0.02, 0.08]
    downsampled_scale = 0.075
    curriculum = False

    all_vertical = False
    no_flat = True
    
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    measure_heights = True
    measured_points_x = [-1.2, -1.05,-0.9,-0.75,-0.6,-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.4,2.55,2.7] # 1mx1.6m rectangle (without center line)
    measured_points_y = [-1.2,-1.05,-0.9,-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75,0.9,1.05,1.2]
    measure_horizontal_noise = 0.0

    selected = False # select a unique terrain type and pass all arguments
    terrain_kwargs = None # Dict of arguments for selected terrain
    max_init_terrain_level = 5 # starting curriculum state
    max_difficulty = True
    terrain_length = 18.
    terrain_width = 18
    num_rows= 2 # number of terrain rows (levels)  # spreaded is benifitiall !
    num_cols = 2# number of terrain cols (types)
    
    terrain_dict = {"smooth slope": 0., 
                    "rough slope up": 0.0,
                    "rough slope down": 0.0,
                    "rough stairs up": 0.0, 
                    "rough stairs down": 0.0, 
                    "discrete": 0., 
                    "stepping stones": 0.05,
                    "gaps": 0.05, 
                    "smooth flat": 0,
                    "pit": 0.,
                    "wall": 0.,
                    "platform": 0.,
                    "large stairs up": 0.,
                    "large stairs down": 0.,
                    "parkour": 0.2,
                    "parkour_hurdle": 0.2,
                    "parkour_flat": 0.0,
                    "parkour_step": 0.2,
                    "parkour_gap": 0.15,
                    "demo": 0.15,}
    terrain_proportions = list(terrain_dict.values())
    flat_wall = False # if True, wall is flat
    # trimesh only:
    slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
    origin_zero_z = True
    num_sub_terrains = num_rows * num_cols
