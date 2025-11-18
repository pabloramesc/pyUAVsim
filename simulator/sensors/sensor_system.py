from ..aircraft.aircraft_state import AircraftState
from .accel import Accel, AccelParams
from .gyro import Gyro, GyroParams
from .baro import Baro, BaroParams
from .airspeed import Airspeed, AirspeedParams
from .compass import Compass, CompassParams
from .gps import GPS, GPSParams
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SensorReadings:
    accel: np.ndarray
    gyro: np.ndarray
    baro: float
    airspeed: float
    compass: float
    gps: np.ndarray
    
    def as_array(self) -> np.ndarray:
        return np.hstack((
            self.accel,
            self.gyro,
            np.array([self.baro]),
            np.array([self.airspeed]),
            np.array([self.compass]),
            self.gps
        ))


class SensorSystem:
    def __init__(
        self,
        state: AircraftState,
        acc_params: Optional[AccelParams] = None,
        gyr_params: Optional[GyroParams] = None,
        baro_params: Optional[BaroParams] = None,
        airspeed_params: Optional[AirspeedParams] = None,
        compass_params: Optional[CompassParams] = None,
        gps_params: Optional[GPSParams] = None,
    ) -> None:
        self.state = state

        acc_params = acc_params or AccelParams()
        self.acc = Accel(acc_params, state)

        gyr_params = gyr_params or GyroParams()
        self.gyr = Gyro(gyr_params, state)

        baro_params = baro_params or BaroParams()
        self.baro = Baro(baro_params, state)

        airspeed_params = airspeed_params or AirspeedParams()
        self.airspeed = Airspeed(airspeed_params, state)

        compass_params = compass_params or CompassParams()
        self.compass = Compass(compass_params, state)

        gps_params = gps_params or GPSParams()
        self.gps = GPS(gps_params, state)

        self.sensors = [
            self.acc,
            self.gyr,
            self.baro,
            self.airspeed,
            self.compass,
            self.gps,
        ]

    def initialize(self, t: float) -> None:
        for sensor in self.sensors:
            sensor.initialize(t)

    def update(self, t: float) -> None:
        for sensor in self.sensors:
            sensor.update(t)

    def read(self, t: float) -> SensorReadings:
        return SensorReadings(
            accel=self.acc.read(t),
            gyro=self.gyr.read(t),
            baro=self.baro.read(t).item(),
            airspeed=self.airspeed.read(t).item(),
            compass=self.compass.read(t).item(),
            gps=self.gps.read(t),
        )
