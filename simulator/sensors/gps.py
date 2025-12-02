from dataclasses import dataclass, field

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.sensors.sensor import Sensor, SensorParams
from simulator.math.angles import wrap_angle_2pi


@dataclass
class GPSParams(SensorParams):
    """
    GPS configuration handler, children of `SensorParams`

    Attributes
    ----------
    sample_rate : float
        Sampling rate of the GPS sensor in Hz.
    reading_delay : float
        Delay in seconds before a GPS reading is available.
    position_std : np.ndarray
        Standard deviation of position noise in meters (3-element array for
        north, east, down).
    velocity_std : np.ndarray
        Standard deviation of velocity noise in m/s (3-element array for
        north, east, down).
    time_constant : float
        Time constant for the first-order Gauss-Markov process modeling
        GPS position error, in seconds.
    """

    sample_rate: float = 1.0  # Hz
    reading_delay: float = 0.0  # seconds
    position_std: np.ndarray = field(
        default_factory=lambda: np.array([0.21, 0.21, 0.40])
    )  # meters
    velocity_std: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.05])
    )  # m/s
    time_constant: float = 1100.0  # seconds


class GPS(Sensor):

    def __init__(
        self, params: GPSParams, state: AircraftState, name: str = "gps"
    ) -> None:
        super().__init__(
            params,
            state,
            name=name,
            reading_names=["pn", "pe", "pd", "Vg", "chi"],
            reading_units=["m", "m", "m", "m/s", "deg"],
        )

        self.params = params

        self.ideal_value = np.zeros(5)  # pn, pe, pd, Vg, chi
        self.noisy_value = np.zeros(5)
        self.prev_noisy_value = np.zeros(5)

        self.position_error = np.zeros(3)
        self.position_ideal = np.zeros(3)
        self.velocity_ideal = np.zeros(3)

    def get_ideal_value(self, t: float) -> np.ndarray:
        self.position_ideal = self.state.ned_position
        self.velocity_ideal = self.state.ned_velocity
        pn, pe, pd = self.position_ideal
        vn, ve, vd = self.velocity_ideal
        Vg = np.sqrt(vn**2 + ve**2)
        heading = wrap_angle_2pi(np.arctan2(ve, vn))
        return np.array([pn, pe, -pd, Vg, np.rad2deg(heading)])

    def get_noisy_value(self, t: float) -> np.ndarray:
        # Update position error using first-order Gauss-Markov process
        self.position_error = np.exp(
            -self.params.sample_rate / self.params.time_constant
        ) * self.position_error + np.random.normal(
            0.0, self.params.position_std, size=3
        )
        position_noisy = self.position_ideal + self.position_error

        # Calculate velocity uncertainty
        vn, ve, vd = self.velocity_ideal
        Vg = np.sqrt(vn**2 + ve**2)
        heading = np.arctan2(ve, vn)

        sigma_vn, sigma_ve, sigma_vd = self.params.velocity_std
        # sigma_Vg = np.sqrt(
        #     (vn**2 * sigma_vn**2 + ve**2 * sigma_ve**2) / (vn**2 + ve**2)
        # )
        # sigma_chi = np.sqrt(
        #     (ve**2 * sigma_vn**2 + vn**2 * sigma_ve**2) / (vn**2 + ve**2)**2
        # )

        # Simplified expressions for sigma_vn = sigma_ve
        sigma_Vg = sigma_vn
        sigma_chi = sigma_vn / Vg

        Vg_noisy = Vg + np.random.normal(0.0, sigma_Vg)
        heading_noisy = heading + np.random.normal(0.0, sigma_chi)

        return np.array(
            [
                position_noisy[0],
                position_noisy[1],
                -position_noisy[2],
                Vg_noisy,
                np.rad2deg(wrap_angle_2pi(heading_noisy)),
            ]
        )
