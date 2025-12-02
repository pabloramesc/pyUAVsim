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


@dataclass(frozen=True)
class SensorReading:
    time: float
    data: np.ndarray
    is_new: bool = True
    sensor_name: str | None = None
    sensor_id: int | None = None
    reading_names: list[str] | None = None
    reading_units: list[str] | None = None


class Sensor(ABC):

    _count = 0

    def __init__(
        self,
        params: SensorParams,
        state: AircraftState,
        name: str | None = None,
        reading_names: list[str] | None = None,
        reading_units: list[str] | None = None,
    ) -> None:
        """General sensor class initialization.

        Parameters
        ----------
        params : SensorParams
            Parameters of the sensor
        state : AircraftState
            Aircraft's state class instance
        name : str, optional
            Name of the sensor, by default None
        reading_names : list[str], optional
            Names of each reading provided by the sensor, by default None
        reading_units : list[str], optional
            Units of each reading provided by the sensor, by default None
        """
        self.sample_rate = params.sample_rate
        self.sample_period = 1.0 / params.sample_rate
        self.reading_delay = params.reading_delay

        self.state = state

        Sensor._count += 1
        self.id = Sensor._count

        self.name = name if name is not None else f"{type(self).__name__}_{self.id}"
        self.reading_names = reading_names
        self.reading_units = reading_units

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

    def read(self, t: float) -> SensorReading:
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
        SensorReading
            Current sensor reading with timestamp and metadata
        """
        # if reading delay is completed
        if t >= self.last_update_time + self.reading_delay:
            if self.noisy_value is None:
                raise ValueError("Sensor noisy_value is not initialized.")
            data_value = self.noisy_value
            data_time = self.last_update_time
            is_new = True

        # if not enough time has passed
        else:
            if self.prev_noisy_value is None:
                raise ValueError("Sensor prev_noisy_value is not initialized.")
            data_value = self.prev_noisy_value
            data_time = self.last_update_time - self.sample_period
            is_new = False

        return SensorReading(
            time=data_time,
            data=data_value,
            is_new=is_new,
            sensor_name=self.name,
            sensor_id=self.id,
            reading_names=self.reading_names,
            reading_units=self.reading_units,
        )

    def get_data_time(self, t: float) -> float:
        """
        Returns the simulation timestamp of the data currently available
        to be read, accounting for the reading delay.

        Parameters
        ----------
        t : float
            Current simulation time in seconds

        Returns
        -------
        float
            Timestamp of the data currently available to be read
        """
        # If the delay has passed, we are seeing the latest update
        if t >= self.last_update_time + self.reading_delay:
            return self.last_update_time

        # Otherwise, we are still seeing the previous buffered value
        return self.last_update_time - self.sample_period
