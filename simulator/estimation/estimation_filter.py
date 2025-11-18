from abc import ABC, abstractmethod
from dataclasses import dataclass

from simulator.sensors.sensor_system import SensorReadings
from simulator.utils.types import FloatLike

import numpy as np
from numpy.typing import NDArray


@dataclass
class EstimatedState:
    roll: FloatLike
    pitch: FloatLike
    yaw: FloatLike
    p: FloatLike
    q: FloatLike
    r: FloatLike
    Va: FloatLike
    h: FloatLike
    pn: FloatLike
    pe: FloatLike
    pd: FloatLike
    Vg: FloatLike
    heading: FloatLike
    
    def as_array(self) -> NDArray[np.floating]:
        return np.array(
            [
                self.roll,
                self.pitch,
                self.yaw,
                self.p,
                self.q,
                self.r,
                self.Va,
                self.h,
                self.pn,
                self.pe,
                self.pd,
                self.Vg,
                self.heading,
            ],
            dtype=float,
        )


class EstimationFilter(ABC):
    @abstractmethod
    def update(self, readings: SensorReadings) -> EstimatedState:
        pass
