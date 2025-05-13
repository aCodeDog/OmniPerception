from abc import ABC, abstractmethod


class BaseSensor(ABC):
    def __init__(self, env,env_cfg,sensor_config, num_sensor, device):
        self.env =env
        self.env_cfg =env_cfg
        self.sensor_cfg = sensor_config
        self.device = device
        self.num_sensor = num_sensor
        self.robot_position = None
        self.robot_orientation = None
        self.robot_linvel = None
        self.robot_angvel = None

    @abstractmethod
    def init_tensors(self):
        raise NotImplementedError("update func not implemented")

    @abstractmethod
    def update(self):
        raise NotImplementedError("update func not implemented")