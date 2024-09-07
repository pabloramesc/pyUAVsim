"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

import numpy as np

from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.autopilot_status import AutopilotStatus
from simulator.math.angles import diff_angle_pi


class WaypointAction(ABC):
    def __init__(self, *params) -> None:
        self.params = params

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """
        Execute the action. This method should be overridden in each subclass.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(str, self.params))})"


class OrbitUnlimited(WaypointAction):
    def __init__(self, radius: float) -> None:
        super().__init__(radius)
        self.radius = radius

    def execute(self) -> None:
        return None
    
    def is_done(self) -> bool:
        return False


class OrbitTime(WaypointAction):
    def __init__(self, time: float, radius: float) -> None:
        super().__init__(time, radius)
        self.time = time
        self.radius = radius

        self._elapsed_time = 0.0

    def execute(self, dt: float) -> None:
        self._elapsed_time += dt

    def is_done(self) -> bool:
        return self._elapsed_time >= self.time


class OrbitTurns(WaypointAction):
    def __init__(self, turns: int, radius: float) -> None:
        super().__init__(turns, radius)
        self.turns = turns
        self.radius = radius

        self._previous_angular_position = 0.0
        self._accumulated_angular_position = 0.0

    def execute(self, ang_pos: float) -> None:
        ang_pos_inc = diff_angle_pi(ang_pos, self._previous_angular_position)
        self._accumulated_angular_position += ang_pos_inc

    def is_done(self) -> bool:
        return abs(self._accumulated_angular_position) >= 2.0 * self.turns * np.pi


class OrbitAlt(WaypointAction):
    def __init__(self, altitude: float, radius: float) -> None:
        super().__init__(altitude, radius)
        self.altitude = altitude
        self.radius = radius

        self._current_altitude = None

    def execute(self, alt: float) -> None:
        self._current_altitude = alt

    def is_done(self) -> bool:
        if self._current_altitude is None:
            return False
        return self._current_altitude >= self.altitude


class GoWaypoint(WaypointAction):
    def __init__(self, waypoint_id: int) -> None:
        super().__init__(waypoint_id)
        self.waypoint_id = waypoint_id

        self._is_done = False

    def execute(self) -> None:
        self._is_done = True

    def is_done(self) -> bool:
        return self._is_done


class SetAirspeed(WaypointAction):
    def __init__(self, airspeed: float) -> None:
        super().__init__(airspeed)
        self.airspeed = airspeed

        self._is_done = False

    def execute(self) -> None:
        self._is_done = True
    
    def is_done(self) -> bool:
        return self._is_done

