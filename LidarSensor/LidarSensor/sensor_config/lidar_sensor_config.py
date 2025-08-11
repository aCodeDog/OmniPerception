from .base_sensor_config import BaseSensorConfig
import numpy as np
from enum import Enum
from dataclasses import dataclass, field


class LidarType(Enum):
    """Standardized lidar sensor types"""
    # Simple grid-based lidar
    SIMPLE_GRID = "simple_grid"
    
    # Height scanner
    HEIGHT_SCANNER = "height_scanner"
    
    # Livox sensors
    AVIA = "avia"
    HORIZON = "horizon" 
    HAP = "HAP"
    MID360 = "mid360"
    MID40 = "mid40"
    MID70 = "mid70"
    TELE = "tele"
    
    # Traditional spinning lidars (to be implemented)
    HDL64 = "hdl64"
    VLP32 = "vlp32"
    OS128 = "os128"


@dataclass
class LidarConfig(BaseSensorConfig):
    """Optimized LidarSensor configuration"""
    
    # Core sensor settings
    sensor_type: LidarType = LidarType.MID360
    num_sensors: int = 1
    dt: float = 0.02  # simulation time step
    update_frequency: float = 50.0  # sensor update rate in Hz
    
    # Range settings
    max_range: float = 50.0
    min_range: float = 0.2
    
    # Grid-based lidar settings (only used when sensor_type is SIMPLE_GRID)
    horizontal_line_num: int = 80
    vertical_line_num: int = 50
    horizontal_fov_deg_min: float = -180
    horizontal_fov_deg_max: float = 180
    vertical_fov_deg_min: float = -2
    vertical_fov_deg_max: float = 57
    
    # Height scanner settings (only used when sensor_type is HEIGHT_SCANNER)
    height_scanner_size: list = field(default_factory=lambda: [2.0, 2.0])  # [length, width] in meters
    height_scanner_resolution: float = 0.1  # spacing between rays in meters
    height_scanner_direction: list = field(default_factory=lambda: [0.0, 0.0, -1.0])  # ray direction (downward)
    height_scanner_ordering: str = "xy"  # grid ordering: "xy" or "yx"
    height_scanner_offset: list = field(default_factory=lambda: [0.0, 0.0])  # [x, y] offset in meters for ray start positions
    height_scanner_height_above_ground: float = 10.0  # height above ground level where rays start (in meters)
    
    # Output settings
    return_pointcloud: bool = True
    pointcloud_in_world_frame: bool = False
    segmentation_camera: bool = False
    
    # Noise settings
    enable_sensor_noise: bool = False
    random_distance_noise: float = 0.03
    random_angle_noise: float = 0.15 / 180 * np.pi
    pixel_dropout_prob: float = 0.01
    pixel_std_dev_multiplier: float = 0.01
    
    # Transform settings
    euler_frame_rot_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Placement randomization
    randomize_placement: bool = True
    min_translation: list = field(default_factory=lambda: [0.07, -0.06, 0.01])
    max_translation: list = field(default_factory=lambda: [0.12, 0.03, 0.04])
    min_euler_rotation_deg: list = field(default_factory=lambda: [-5.0, -5.0, -5.0])
    max_euler_rotation_deg: list = field(default_factory=lambda: [5.0, 5.0, 5.0])
    
    # Nominal position (for Isaac Gym sensors)
    nominal_position: list = field(default_factory=lambda: [0.10, 0.0, 0.03])
    nominal_orientation_euler_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Data normalization
    normalize_range: bool = False
    far_out_of_range_value: float = -1.0
    near_out_of_range_value: float = -1.0
    
    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        # Convert string sensor_type to enum if needed
        if isinstance(self.sensor_type, str):
            self.sensor_type = LidarType(self.sensor_type)
        
        # Auto-adjust normalization settings
        if self.return_pointcloud and self.pointcloud_in_world_frame:
            self.normalize_range = False
        
        # Set out-of-range values based on normalization
        if self.normalize_range:
            self.far_out_of_range_value = self.max_range
            self.near_out_of_range_value = -self.max_range
        else:
            self.far_out_of_range_value = -1.0
            self.near_out_of_range_value = -1.0
    
    @property
    def is_simple_grid(self) -> bool:
        """Check if this is a simple grid-based lidar"""
        return self.sensor_type == LidarType.SIMPLE_GRID
    
    @property
    def is_height_scanner(self) -> bool:
        """Check if this is a height scanner"""
        return self.sensor_type == LidarType.HEIGHT_SCANNER
    
    @property
    def is_livox_sensor(self) -> bool:
        """Check if this is a Livox-type sensor"""
        return self.sensor_type in [
            LidarType.AVIA, LidarType.HORIZON, LidarType.HAP,
            LidarType.MID360, LidarType.MID40, LidarType.MID70, LidarType.TELE
        ]
    
    @property
    def is_spinning_lidar(self) -> bool:
        """Check if this is a traditional spinning lidar"""
        return self.sensor_type in [LidarType.HDL64, LidarType.VLP32, LidarType.OS128]
