import os

PKG_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


SENSOR_ROOT_DIR = os.path.join(PKG_ROOT_DIR, 'LidarSensor')

RESOURCES_DIR = os.path.join(SENSOR_ROOT_DIR, 'resources')