"""
Copyright (c) 2022 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.aircraft import AircraftState


@dataclass
class SensorParams(ABC):
    """
    General sensor configuration handler

    Attributes
    ----------
    sample_rate : float
        Sensor sampling rate in Hz

    reading_delay : float
        Sensor processing delay between sampling and reading available
    """

    sample_rate: float
    reading_delay: float


class Sensor(ABC):

    _count = 0

    def __init__(self, params: SensorParams, state: AircraftState, name: str | None = None) -> None:
        """General sensor class initialization.

        Parameters
        ----------
        params : SensorParams
            Parameters of the sensor
        state : AircraftState
            Aircraft's state class instance
        """
        self.sample_rate = params.sample_rate
        self.sample_period = 1.0 / params.sample_rate
        self.reading_delay = params.reading_delay

        self.state = state

        Sensor._count += 1
        self.id = Sensor._count
        
        self.name = name if name is not None else f"{type(self).__name__}_{self.id}"

        self.last_update_time = 0.0
        self.ideal_value: np.ndarray | None = None  # ideal measurement
        self.noisy_value: np.ndarray | None = None  # real measurement (ideal + noise)
        self.prev_noisy_value: np.ndarray | None = (
            None  # previous reading to simulate reading delay
        )

    def initialize(self, t: float) -> None:
        """
        Initilizate the sensor by setting `last_update_time` with `t` and forcing update for `last_reading`
        and `prev_reading`.

        Parameters
        ----------
        t : float
            Current simulation time in seconds
        """
        self.last_update_time = t
        self.ideal_value = self.get_ideal_value(t)
        self.noisy_value = self.get_noisy_value(t)
        self.prev_noisy_value = self.noisy_value

    def needs_update(self, t: float) -> bool:
        """
        Check if enough time has passed since last update.

        Parameters
        ----------
        t : float
            Current simulation time in seconds

        Returns
        -------
        bool
            `True` if update for the sensor is needed and `False` if not enough time has passed since last update
        """
        if t >= self.last_update_time + self.sample_period:
            return True  # if sample period is completed the sensor needs an update
        return False  # if not enough time has passed to be updated

    def update(self, t: float, force: bool = False) -> bool:
        """
        Compute new sensor sample and reading if needed.

        Parameters
        ----------
        t : float
            Current simulation time in seconds

        force : bool, optinal
            Force measurement and reading update regardless of sample period, by default False

        Returns
        -------
        bool
            `True` if update was done or `False` if not enought time has passed since last update
        """
        if self.needs_update(t) or force:
            self.last_update_time = t
            self.ideal_value = self.get_ideal_value(t)
            self.prev_noisy_value = (
                self.noisy_value
            )  # store previous reading to simulate buffer delay
            self.noisy_value = self.get_noisy_value(t)
            return True
        else:
            return False

    @abstractmethod
    def get_ideal_value(self, t: float) -> np.ndarray:
        """Interface method to implement the sensor measurement model.
        The `ideal_value` is the ideal measurement calculated from the aircraft's state without noise.
        """

    @abstractmethod
    def get_noisy_value(self, t: float) -> np.ndarray:
        """Interface method to implement the sensor noise model.
        The `noisy_value` is the real measurement with noise and digitalization.
        """

    def read(self, t: float) -> np.ndarray:
        """
        Get the current reading of the sensor.
        If reading delay has passed since last update, returns the last reading (`noisy_value`).
        Otherwise, it returns the previous reading (`prev_noisy_value`).

        Parameters
        ----------
        t : float
            Current simulation time in seconds

        Returns
        -------
        np.ndarray
            The reading of the sensor considering reading delay
        """
        # if reading delay is completed
        if t >= self.last_update_time + self.reading_delay:
            if self.noisy_value is None:
                raise ValueError("Sensor noisy_value is not initialized.")
            return self.noisy_value
        
        # if not enough time has passed
        else:
            if self.prev_noisy_value is None:
                raise ValueError("Sensor prev_noisy_value is not initialized.")
            return self.prev_noisy_value
