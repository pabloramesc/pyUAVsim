from dataclasses import dataclass

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.math.angles import wrap_angle_2pi
from simulator.sensors.sensor import Sensor, SensorParams


@dataclass
class CompassParams(SensorParams):
    """
    Digital compass configuration handler, children of `SensorParams`

    Attributes
    ----------
    bias : float, optional
        Constant bias added to the compass reading (in degrees). Default is 0.5.
    accuracy : float, optional
        Standard deviation of the Gaussian noise added to the compass reading (in degrees). Default is 3.0.
    """

    sample_rate: float = 8.0  # Hz
    reading_delay: float = 0.0  # seconds
    bias: float = 0.5
    accuracy: float = 3.0  # degrees


class Compass(Sensor):

    def __init__(
        self, params: CompassParams, state: AircraftState, name: str = "hdg"
    ) -> None:
        super().__init__(
            params, state, name=name, reading_names=["heading"], reading_units=["deg"]
        )

        self.params = params

        self.ideal_value = np.zeros(1)
        self.noisy_value = np.zeros(1)
        self.prev_noisy_value = np.zeros(1)

    def get_ideal_value(self, t: float) -> np.ndarray:
        heading = wrap_angle_2pi(self.state.yaw)  # in radians
        return np.rad2deg([heading])  # convert to degrees

    def get_noisy_value(self, t: float) -> np.ndarray:
        if self.ideal_value is None:
            raise ValueError("Ideal value has not been computed yet.")
        reading = self.ideal_value + np.random.normal(
            self.params.bias, self.params.accuracy
        )
        return reading % 360.0  # wrap to [0, 360) degrees
