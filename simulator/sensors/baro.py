from dataclasses import dataclass

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.environment.isa import isa_pressure
from simulator.sensors.sensor import Sensor, SensorParams
from simulator.sensors.signal_simulation import saturate


@dataclass
class BaroParams(SensorParams):
    """
    Absolute pressure sensor (barometer) configuration handler, children of `SensorParams`

    Attributes
    ----------
    min_value : float, optional
        The minimum measurable absolute pressure in kPa, by default 15.0
    max_value : float, optional
        The maximum measurable absolute pressure in kPa, by default 115.0
    bias : float, optional
        The constant bias in kPa, by default 0.125
    deviation : float, optional
        The standard deviation in kPa, by default 0.01
    """

    sample_rate: float = 10.0  # Hz
    reading_delay: float = 0.0  # seconds
    min_value: float = 15.0
    max_value: float = 115.0
    bias: float = 0.125
    deviation: float = 0.01


class Baro(Sensor):

    def __init__(self, params: BaroParams, state: AircraftState, name: str = "baro") -> None:
        super().__init__(params, state, name=name)

        self.params = params

        self.ideal_value = np.zeros(1)
        self.noisy_value = np.zeros(1)
        self.prev_noisy_value = np.zeros(1)

    def get_ideal_value(self, t: float) -> np.ndarray:
        h = self.state.altitude
        p = isa_pressure(h)
        return np.array([p]) * 1e-3  # Convert to kPa

    def get_noisy_value(self, t: float) -> np.ndarray:
        if self.ideal_value is None:
            raise ValueError("Ideal value has not been computed yet.")
        reading = self.ideal_value + np.random.normal(
            loc=self.params.bias, scale=self.params.deviation, size=1
        )
        reading = saturate(
            reading, min_val=self.params.min_value, max_val=self.params.max_value
        )
        return reading
