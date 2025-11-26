import numpy as np

from .attitude_ekf import AttitudeEKF
from .estimation_filter import EstimationFilter


class KalmanEstimator(EstimationFilter):
    def __init__(self, dt):
        super().__init__(dt)
        self.attitde_ekf = AttitudeEKF(dt)
