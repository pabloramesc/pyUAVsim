"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

import numpy as np

from simulator.math.angles import diff_angle_pi

DEFAULT_RADIUS = 100.0


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
    def restart(self) -> None:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def has_failed(self) -> bool:
        # TODO: implement waypoint actions' fail conditions
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(str, self.params))})"


class OrbitUnlimited(WaypointAction):
    def __init__(self, radius: float = DEFAULT_RADIUS, direction: int = 1) -> None:
        super().__init__(radius)
        self.radius = radius
        self.direction = direction

    def execute(self) -> None:
        return None

    def restart(self) -> None:
        return None

    def is_done(self) -> bool:
        return False

    def has_failed(self) -> bool:
        return False


class OrbitTime(WaypointAction):
    def __init__(
        self, time: float, radius: float = DEFAULT_RADIUS, direction=1
    ) -> None:
        super().__init__(time, radius)
        self.time = time
        self.radius = radius
        self.direction = direction

        self._elapsed_time = 0.0

    def execute(self, dt: float) -> None:
        self._elapsed_time += dt

    def restart(self) -> None:
        self._elapsed_time = 0.0

    def is_done(self) -> bool:
        return self._elapsed_time >= self.time

    def has_failed(self) -> bool:
        return False


class OrbitTurns(WaypointAction):
    def __init__(self, turns: int, radius: float = DEFAULT_RADIUS, direction=1) -> None:
        super().__init__(turns, radius)
        self.turns = turns
        self.radius = radius
        self.direction = direction

        self._previous_angular_position = 0.0
        self._accumulated_angular_position = 0.0
        self._completed_turns = 0.0

    def execute(self, ang_pos: float) -> None:
        ang_pos_inc = diff_angle_pi(ang_pos, self._previous_angular_position)
        self._accumulated_angular_position += ang_pos_inc
        self._completed_turns += abs(self._accumulated_angular_position) / (2.0 * np.pi)

    def restart(self) -> None:
        self._previous_angular_position = 0.0
        self._accumulated_angular_position = 0.0

    def is_done(self) -> bool:
        return self._completed_turns >= self.turns

    def has_failed(self) -> bool:
        return False


class OrbitAlt(WaypointAction):
    def __init__(
        self, altitude: float, radius: float = DEFAULT_RADIUS, direction=1
    ) -> None:
        super().__init__(altitude, radius)
        self.altitude = altitude
        self.radius = radius
        self.direction = direction

        self._current_altitude = None

    def execute(self, alt: float) -> None:
        self._current_altitude = alt

    def restart(self) -> None:
        self._current_altitude = None

    def is_done(self) -> bool:
        if self._current_altitude is None:
            return False
        return self._current_altitude >= self.altitude

    def has_failed(self) -> bool:
        return False


class GoWaypoint(WaypointAction):
    def __init__(self, wp_id: int, repeat: int = -1) -> None:
        super().__init__(wp_id)
        self.wp_id = wp_id
        self.repeat = repeat

        self._repeat_count = 0
        self._is_done = False

    def execute(self) -> None:
        self._repeat_count += 1
        self._is_done = True

    def restart(self) -> None:
        # never restart the counter
        # full repeat process can only be done once
        # self._repeat_count = 0
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def has_failed(self) -> bool:
        return False

    def has_pending_jumps(self) -> bool:
        if self.repeat < 0:
            return True  # do infinit jumps
        else:
            return self.repeat > self._repeat_count


class SetAirspeed(WaypointAction):
    def __init__(self, airspeed: float) -> None:
        super().__init__(airspeed)
        self.airspeed = airspeed

        self._is_done = False

    def execute(self) -> None:
        self._is_done = True

    def restart(self) -> None:
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def has_failed(self) -> bool:
        return False
