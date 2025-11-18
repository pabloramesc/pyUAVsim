
import numpy as np

from dataclasses import dataclass, field

from simulator.aircraft.aircraft_state import AircraftState
from simulator.sensors.sensor import Sensor, SensorParams
from simulator.sensors.noise_models import get_white_noise
from simulator.sensors.signal_simulation import saturate, digitalize


@dataclass
class GyroParams(SensorParams):
    """
    Tri-axial gyroscope configuration handler, children of `SensorParams`

    Attributes
    ----------
    full_scale : float, optional
        The gyroscope measurement range as (-full_scale, full_scale) in deg/s, by default 16.0

    resolution : int, optional
        The resolution of the gyroscope ADC in bits, by default 16

    noise_density : float, optional
        The gyroscope white noise parameter in deg/s/sqrt(Hz), by default 0.0

    offset : numpy.ndarray, optional
        3 size array with gyroscope's constant bias for each axis in deg/s, by default np.zeros(3)
    """
    
    sample_rate: float = 80.0  # Hz
    reading_delay: float = 0.0  # seconds
    full_scale: float = 350.0
    resolution: int = 16
    noise_density: float = 0.015
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Gyro(Sensor):

    def __init__(self, params: GyroParams, state: AircraftState, name: str = "gyr") -> None:
        super().__init__(params, state, name=name)

        self.params = params

        self.ideal_value = np.zeros(3)
        self.noisy_value = np.zeros(3)
        self.prev_noisy_value = np.zeros(3)

    def get_ideal_value(self, t: float) -> np.ndarray:
        gyr = self.state.angular_rates
        return np.rad2deg(gyr)

    def get_noisy_value(self, t: float) -> np.ndarray:
        if self.ideal_value is None:
            raise ValueError("Ideal value has not been computed yet.")
        reading = self.ideal_value + get_white_noise(
            Nd=self.params.noise_density, fs=self.sample_rate, nlen=3
        ) + self.params.offset
        reading = saturate(reading, -self.params.full_scale, +self.params.full_scale)
        reading = digitalize(reading, self.params.full_scale, self.params.resolution)
        return reading