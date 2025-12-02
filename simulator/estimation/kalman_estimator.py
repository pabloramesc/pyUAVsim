import numpy as np

from .attitude_ekf import AttitudeEKF
from .estimation_filter import EstimationFilter, EstimatedState
from ..sensors.sensor_system import SensorReadings


class KalmanEstimator(EstimationFilter):
    def __init__(self, dt: float) -> None:
        super().__init__(dt)
        self.attitde_ekf = AttitudeEKF(dt)
        
    def update(self, readings: SensorReadings) -> EstimatedState:

        return None