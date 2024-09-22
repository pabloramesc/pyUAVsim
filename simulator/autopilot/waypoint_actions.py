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
    """
    Abstract base class for waypoint actions.

    Attributes
    ----------
    params : tuple
        Stores the parameters for the action.
    """

    def __init__(self, *params) -> None:
        """
        Initialize the WaypointAction.

        Parameters
        ----------
        *params : tuple
            Parameters for the specific waypoint action.
        """
        self.params = params

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """
        Execute the action. Must be implemented in each subclass.
        """
        pass

    @abstractmethod
    def restart(self) -> None:
        """
        Restarts the action. This should reset any internal state.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Checks whether the action is complete.

        Returns
        -------
        bool
            True if the action is done, False otherwise.
        """
        pass

    @abstractmethod
    def has_failed(self) -> bool:
        """
        Checks if the action has failed.

        Returns
        -------
        bool
            True if the action has failed, False otherwise.
        """
        # TODO: implement waypoint actions' fail conditions
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(str, self.params))})"


class OrbitUnlimited(WaypointAction):
    """
    Orbit around a waypoint indefinitely.

    Attributes
    ----------
    radius : float
        The radius of the orbit.
    direction : int
        The direction of the orbit (1 for clockwise, -1 for counterclockwise).
    """

    def __init__(self, radius: float = DEFAULT_RADIUS, direction: int = 1) -> None:
        """
        Initialize the OrbitUnlimited action.

        Parameters
        ----------
        radius : float, optional
            The radius of the orbit (default is 100.0).
        direction : int, optional
            The direction of the orbit (1 for clockwise, -1 for counterclockwise; default is 1).
        """
        super().__init__(radius, direction)
        self.radius = radius
        self.direction = direction

    def execute(self) -> None:
        """Executes the orbit action. This method does nothing for unlimited orbit."""
        return None

    def restart(self) -> None:
        """Restarts the orbit action. This method does nothing for unlimited orbit."""
        return None

    def is_done(self) -> bool:
        """
        Checks whether the orbit action is complete.

        Returns
        -------
        bool
            Always returns False since it orbits indefinitely.
        """
        return False

    def has_failed(self) -> bool:
        """
        Checks if the orbit action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False


class OrbitTime(WaypointAction):
    """
    Orbit around a waypoint for a specified amount of time.

    Attributes
    ----------
    time : float
        The duration for which to orbit.
    radius : float
        The radius of the orbit.
    direction : int
        The direction of the orbit (1 for clockwise, -1 for counterclockwise).
    elapsed_time : float
        The accumulated time spent orbiting.
    """

    def __init__(
        self, time: float, radius: float = DEFAULT_RADIUS, direction: int = 1
    ) -> None:
        """
        Initialize the OrbitTime action.

        Parameters
        ----------
        time : float
            The duration for which to orbit.
        radius : float, optional
            The radius of the orbit (default is 100.0).
        direction : int, optional
            The direction of the orbit (1 for clockwise, -1 for counterclockwise; default is 1).
        """
        super().__init__(time, radius, direction)
        self.time = time
        self.radius = radius
        self.direction = direction

        self.elapsed_time = 0.0

    def execute(self, dt: float) -> None:
        """
        Updates the elapsed time for the orbit action.

        Parameters
        ----------
        dt : float
            The time step to add to the elapsed time.
        """
        self.elapsed_time += dt

    def restart(self) -> None:
        """Restarts the orbit action by resetting the elapsed time."""
        self.elapsed_time = 0.0

    def is_done(self) -> bool:
        """
        Checks whether the orbit action is complete.

        Returns
        -------
        bool
            True if the elapsed time exceeds the specified duration, False otherwise.
        """
        return self.elapsed_time >= self.time

    def has_failed(self) -> bool:
        """
        Checks if the orbit action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False


class OrbitTurns(WaypointAction):
    """
    Orbit around a waypoint for a specified number of turns.

    Attributes
    ----------
    turns : int
        The number of turns to complete.
    radius : float
        The radius of the orbit.
    direction : int
        The direction of the orbit (1 for clockwise, -1 for counterclockwise).
    completed_turns : float
        The number of completed turns.
    """

    def __init__(
        self, turns: float, radius: float = DEFAULT_RADIUS, direction: int = 1
    ) -> None:
        """
        Initialize the OrbitTurns action.

        Parameters
        ----------
        turns : float
            The number of turns to complete.
        radius : float, optional
            The radius of the orbit (default is 100.0).
        direction : int, optional
            The direction of the orbit (1 for clockwise, -1 for counterclockwise; default is 1).
        """
        super().__init__(turns, radius, direction)
        self.turns = turns
        self.radius = radius
        self.direction = direction

        self._prev_ang_pos: float = None
        self._cum_ang_pos = 0.0
        self.completed_turns = 0.0

    def execute(self, ang_pos: float) -> None:
        """
        Updates the accumulated angular position based on the current angular position.

        Parameters
        ----------
        ang_pos : float
            The current angular position of the vehicle.
        """
        if self._prev_ang_pos is not None:
            ang_pos_inc = diff_angle_pi(ang_pos, self._prev_ang_pos)
        else:
            ang_pos_inc = 0.0
        self._cum_ang_pos += ang_pos_inc
        self._prev_ang_pos = ang_pos
        self.completed_turns = abs(self._cum_ang_pos) / (2.0 * np.pi)

    def restart(self) -> None:
        """Restarts the orbit action by resetting the accumulated values."""
        self._prev_ang_pos = None
        self._cum_ang_pos = 0.0
        self.completed_turns = 0.0

    def is_done(self) -> bool:
        """
        Checks whether the orbit action is complete.

        Returns
        -------
        bool
            True if the number of completed turns meets or exceeds the specified number of turns, False otherwise.
        """
        return self.completed_turns >= self.turns

    def has_failed(self) -> bool:
        """
        Checks if the orbit action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False


class OrbitAlt(WaypointAction):
    """
    Orbit around a waypoint while maintaining a specified altitude.

    Attributes
    ----------
    altitude : float
        The altitude to maintain during the orbit.
    radius : float
        The radius of the orbit.
    direction : int
        The direction of the orbit (1 for clockwise, -1 for counterclockwise).
    current_altitude : float or None
        The current altitude of the vehicle.
    """

    def __init__(
        self, altitude: float, radius: float = DEFAULT_RADIUS, direction: int = 1
    ) -> None:
        """
        Initialize the OrbitAlt action.

        Parameters
        ----------
        altitude : float
            The altitude to maintain during the orbit.
        radius : float, optional
            The radius of the orbit (default is 100.0).
        direction : int, optional
            The direction of the orbit (1 for clockwise, -1 for counterclockwise; default is 1).
        """
        super().__init__(altitude, radius, direction)
        self.altitude = altitude
        self.radius = radius
        self.direction = direction

        self.current_altitude: float = None

    def execute(self, alt: float) -> None:
        """
        Updates the current altitude of the vehicle.

        Parameters
        ----------
        alt : float
            The current altitude of the vehicle.
        """
        self.current_altitude = alt

    def restart(self) -> None:
        """Restarts the orbit action by resetting the current altitude."""
        self.current_altitude = None

    def is_done(self) -> bool:
        """
        Checks whether the orbit action is complete.

        Returns
        -------
        bool
            True if the current altitude meets or exceeds the specified altitude, False otherwise.
        """
        if self.current_altitude is None:
            return False
        return (
            abs(self.current_altitude - self.altitude) < 1.0
        )  # altitude tolerance of 1 meter

    def has_failed(self) -> bool:
        """
        Checks if the orbit action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False


class GoWaypoint(WaypointAction):
    """
    Move to a specific waypoint.

    Attributes
    ----------
    wp_id : int
        The identifier of the waypoint.
    repeat : int
        The number of times to repeat the action (-1 for infinite repeats).
    repeat_count : int
        The number of repeats that have occurred.
    """

    def __init__(self, wp_id: int, repeat: int = -1) -> None:
        """
        Initialize the GoWaypoint action.

        Parameters
        ----------
        wp_id : int
            The identifier of the waypoint to go to.
        repeat : int, optional
            The number of times to repeat the action (-1 for infinite repeats; default is -1).
        """
        super().__init__(wp_id, repeat)
        self.wp_id = wp_id
        self.repeat = repeat

        self.repeat_count = 0
        self._is_done = False

    def execute(self) -> None:
        """Increment the repeat counter and mark action as done."""
        self.repeat_count += 1
        self._is_done = True

    def restart(self) -> None:
        """
        Restart the action. The counter is not reset, but the action is marked as not done.
        """
        self._is_done = False

    def is_done(self) -> bool:
        """
        Check if the action is complete.

        Returns
        -------
        bool
            True if the action is complete, False otherwise.
        """
        return self._is_done

    def has_failed(self) -> bool:
        """
        Check if the action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False

    def has_pending_jumps(self) -> bool:
        """
        Check if there are pending jumps to the waypoint.

        Returns
        -------
        bool
            True if there are pending jumps (infinite repeats or remaining repeats), False otherwise.
        """
        if self.repeat < 0:
            return True  # do infinite jumps
        else:
            return self.repeat > self.repeat_count


class SetAirspeed(WaypointAction):
    """
    Set the airspeed of the vehicle.

    Attributes
    ----------
    airspeed : float
        The target airspeed to set.
    """

    def __init__(self, airspeed: float) -> None:
        """
        Initialize the SetAirspeed action.

        Parameters
        ----------
        airspeed : float
            The target airspeed to set.
        """
        super().__init__(airspeed)
        self.airspeed = airspeed

        self._is_done = False

    def execute(self) -> None:
        """Executes the action and marks it as done."""
        self._is_done = True

    def restart(self) -> None:
        """Restarts the action by marking it as not done."""
        self._is_done = False

    def is_done(self) -> bool:
        """
        Checks whether the action is complete.

        Returns
        -------
        bool
            True if the action is complete, False otherwise.
        """
        return self._is_done

    def has_failed(self) -> bool:
        """
        Checks if the action has failed.

        Returns
        -------
        bool
            Always returns False since it cannot fail.
        """
        return False
