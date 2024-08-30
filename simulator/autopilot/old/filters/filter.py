"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from simulator.autopilot.autopilot_messages import DataHandlerHistory
from simulator.common.constants import DEFAULT_HOME_COORDS
from simulator.sensors.accelerometer import Accelerometer
from simulator.sensors.airspeed import Airspeed
from simulator.sensors.barometer import Barometer
from simulator.sensors.gps import GPS
from simulator.sensors.gyroscope import Gyroscope
from simulator.sensors.magnetometer import Magnetometer
from simulator.sensors.sensors_manager import SensorsManager
from simulator.utils.isa import isa_density, isa_pressure


@dataclass
class FilterConfig:
    pass


@dataclass
class FilterEstimations:
    ### MAIN STATE VARIABLES
    position_ned: np.ndarray
    attitude: np.ndarray
    angular_rate: np.ndarray
    groundspeed: float
    course: float
    ### OTHER ESTIMATED VARS
    velocity_ned: np.ndarray = None
    accel_ned: np.ndarray = None
    airspeed: float = None
    angle_of_attack: float = None
    side_slip_angle: float = None
    wind: np.ndarray = None
    
    @property
    def roll(self):
        return self.attitude[0]
    
    @property
    def pitch(self):
        return self.attitude[1]
    
    @property
    def yaw(self):
        return self.attitude[2]
    
    @property
    def p(self):
        return self.angular_rate[0]
    
    @property
    def q(self):
        return self.angular_rate[1]
    
    @property
    def r(self):
        return self.angular_rate[2]
    
    @property
    def altitude(self):
        return -self.position_ned[2]


class FilterEstimationsHistory(DataHandlerHistory):
    handler_type: type = FilterEstimations

class Filter(ABC):
    def __init__(self, config: FilterConfig, sensors_manager: SensorsManager, home_coords=DEFAULT_HOME_COORDS) -> None:
        super().__init__()

        ##### SENSORS #####
        self.sensors_list = sensors_manager.get_sensors_by_group("navigation")
        self.sensors_dict = {sensor.name: sensor for sensor in self.sensors_list}

        self.gyroscope: Gyroscope = self.sensors_dict["main-gyroscope"]
        self.accelerometer: Accelerometer = self.sensors_dict["main-accelerometer"]
        self.magnetometer: Magnetometer = self.sensors_dict["main-magnetometer"]
        self.gps: GPS = self.sensors_dict["main-gps"]
        self.airspeed: Airspeed = self.sensors_dict["main-airspeed"]
        self.barometer: Barometer = self.sensors_dict["main-barometer"]

        ##### HOME PARAMETERS #####
        self.home_coords = home_coords
        self.home_altitude = home_coords[2]
        self.home_pressure = isa_pressure(self.home_altitude)
        self.home_density = isa_density(self.home_altitude)

        self.set_baro_reference(self.home_pressure)

        ##### HISTORY #####
        self.estimations_history = FilterEstimationsHistory()

    @abstractmethod
    def set_baro_reference(self, home_pressure: float) -> None:
        self.home_pressure = home_pressure

    @abstractmethod
    def set_home_coords(self, home_coords: tuple) -> None:
        self.home_coords = home_coords

    @abstractmethod
    def estimate(self, t: float, dt: float) -> FilterEstimations:
        pass
