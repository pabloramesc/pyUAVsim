"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.aircraft.airframe_parameters import AirframeParameters


class Aircraft:
    def __init__(
        self,
        params: AirframeParameters,
        state0: np.ndarray = np.zeros(12),
        wind: np.ndarray = np.zeros(3),
    ) -> None:
        self.state = state0  # 12 vars: pn pe pd u v w roll pitch yaw p q r
        self.wind = wind
        
    @property
    def ned_position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def body_velocity(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def attitude_angles(self) -> np.ndarray:
        return self.state[6:9]

    @property
    def angular_rates(self) -> np.ndarray:
        return self.state[9:12]
