"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from dataclasses import dataclass, field

from simulator.aircraft.aircraft_state import AircraftState
from simulator.sensors.sensor import Sensor, SensorParams
from simulator.sensors.noise_models import get_white_noise
from simulator.sensors.signal_simulation import saturate, digitalize


@dataclass
class AccelParams(SensorParams):
    """
    Tri-axial accelerometer configuration handler, children of `SensorParams`

    Attributes
    ----------
    full_scale : float, optional
        The accelerometer measurement range as (-full_scale, full_scale) in g, by default 16.0

    resolution : int, optional
        The resolution of the accelerometer ADC in bits, by default 16

    noise_density : float, optional
        The accelerometer white noise parameter in g/sqrt(Hz), by default 0.0

    offset : numpy.ndarray, optional
        3 size array with accelerometer's constant bias for each axis in g, by default np.zeros(3)
    """

    sample_rate: float = 100.0  # Hz
    reading_delay: float = 0.0  # seconds
    full_scale: float = 16.0
    resolution: int = 16
    noise_density: float = 250.0e-6  # 250 ug/sqrt(Hz)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Accel(Sensor):

    def __init__(self, params: AccelParams, state: AircraftState, name: str = "acc") -> None:
        super().__init__(params, state, name=name)

        self.params = params

        self.ideal_value = np.zeros(3)
        self.noisy_value = np.zeros(3)
        self.prev_noisy_value = np.zeros(3)

    def get_ideal_value(self, t: float) -> np.ndarray:
        R_vb = self.state.R_vb
        g = np.array([0.0, 0.0, 9.81])  # Gravity vector in inertial frame (m/s^2)
        v_dot = self.state.body_acceleration
        omega = self.state.angular_rates
        v = self.state.body_velocity
        acc = v_dot + np.cross(omega, v) - R_vb.T @ g        
        return acc

    def get_noisy_value(self, t: float) -> np.ndarray:
        if self.ideal_value is None:
            raise ValueError("Ideal value has not been computed yet.")
        reading = (
            self.ideal_value
            + get_white_noise(Nd=self.params.noise_density, fs=self.sample_rate, nlen=3)
            + self.params.offset
        )
        reading = saturate(reading, -self.params.full_scale, +self.params.full_scale)
        reading = digitalize(reading, self.params.full_scale, self.params.resolution)
        return reading
