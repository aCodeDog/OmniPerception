from .base_sensor_config import BaseSensorConfig
import numpy as np


class LidarConfig(BaseSensorConfig):
    num_sensors = 1  # number of sensors of this type

    sensor_type = "avia"  # mid360,horizon,HAP,mid70,mid40,tele
    
    dt = 0.02  # time step for the sim env
    
    update_frequency = 50.0  # update rate of the sensor in Hz

    # If you use more than one sensors above, there is a need to specify the sensor placement for each sensor
    # this can be added here, but the user can implement this if needed.
    
    # standard OS0-128 configuration
    horizontal_line_num = 80
    vertical_line_num =  50
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = -2
    vertical_fov_deg_max = +57
    
    random_distance_noise = 0.03
    random_angle_noise = 0.15/180*np.pi

    # min and max range do not match with the real sensor, but here we get to limit it for our convenience
    max_range = 20.0
    min_range = 0.2

    # Type of lidar (range, pointcloud, segmentation)
    # You can combine: (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    return_pointcloud = (
        True  # Return a pointcloud instead of an image. Range image will be returned by default
    )
    pointcloud_in_world_frame = False
    segmentation_camera = False  # Setting to true will return a segmentation image along with the range image or pointcloud

    # transform from sensor element coordinate frame to sensor_base_link frame
    euler_frame_rot_deg = [0.0, 0.0, 0.0]

    # Type of data to be returned from the sensor
    normalize_range = True  # will be set to false when pointcloud is in world frame

    # do not change this.
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # divide by max_range. Ignored when pointcloud is in world frame

    # what to do with out of range values
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0

    # randomize placement of the sensor
    randomize_placement = True
    min_translation = [0.07, -0.06, 0.01]
    max_translation = [0.12, 0.03, 0.04]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]

    # nominal position and orientation (only for Isaac Gym Camera Sensors)
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
