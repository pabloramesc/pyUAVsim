import numpy as np

from simulator.autopilot.flight_control import FlightCommand
from simulator.autopilot.path_follower import OrbitPathParams
from simulator.autopilot.path_navigator import PathNavCommand
from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.waypoint_actions import (
    GoWaypoint,
    OrbitAlt,
    OrbitTime,
    OrbitTurns,
    OrbitUnlimited,
    SetAirspeed,
)
from simulator.autopilot.waypoints import Waypoint

ACTIONS_MANAGER_STATUS = ["wait", "run", "done", "fail"]


class WaypointActionsManager:
    """
    Manages the execution of actions associated with waypoints, such as orbits or changes in airspeed.

    Attributes
    ----------
    route_manager : RouteManager
        The object managing the waypoints and the UAV route.
    flight_cmd : FlightCommand
        Object that contains flight control commands such as airspeed or attitude.
    path_cmd : PathNavCommand
        The path command that stores the current navigation command, e.g., path type and parameters.
    status : str
        The current status of the waypoint actions manager, one of ["wait", "run", "done", "fail"].
    active_action_code : str
        The code of the currently active action.
    active_action_status : str or None
        A string representing the current status of the active action, or None if no action is active.

    Methods
    -------
    reset():
        Resets the waypoint actions manager to its initial state.
    update(pos_ned, dt):
        Updates the current action based on the UAV's position and elapsed time.
    """

    def __init__(self, route_manager: RouteManager, flight_cmd: FlightCommand) -> None:
        """
        Initializes the WaypointActionsManager with the route manager and flight command.

        Parameters
        ----------
        route_manager : RouteManager
            The object responsible for managing waypoints and route logic.
        flight_cmd : FlightCommand
            The object managing flight control, such as airspeed, heading, and altitude commands.
        """
        self.route_manager = route_manager
        self.flight_cmd = flight_cmd
        self.path_cmd = PathNavCommand()
        self.status = "wait"
        self.active_action_code = "none"
        self.active_action_status: str = None

    def reset(self) -> None:
        """
        Resets the waypoint actions manager to its default state.
        """
        self.path_cmd = PathNavCommand()
        self.status = "wait"
        self.active_action_code = "none"
        self.active_action_status = None

    def update(self, pos_ned: np.ndarray, dt: float) -> PathNavCommand:
        """
        Updates the active waypoint action based on the current UAV position and time step.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the UAV in NED (North-East-Down) coordinates.
        dt : float
            The time step in seconds.

        Returns
        -------
        PathNavCommand
            The updated path command for navigation.
        """
        if self.status not in ACTIONS_MANAGER_STATUS:
            raise ValueError(
                f"Not valid waypoint actions manager status: {self.status}!"
            )

        if self.status == "done":
            self.reset()

        target_wp = self.route_manager.get_waypoint()

        if target_wp.action is not None:
            if self.status == "wait" and self.route_manager.is_on_waypoint(pos_ned):
                self._start_action(target_wp)

            if self.status == "run":
                self._execute_action(target_wp, pos_ned, dt)

            if target_wp.action.is_done():
                self._finish_action(target_wp, pos_ned)

        return self.path_cmd

    def _start_action(self, waypoint: Waypoint) -> None:
        """Starts the action."""
        waypoint.action.restart()
        if self._is_orbit_action(waypoint.action_code):
            self._set_orbit_path(waypoint)
        self.active_action_code = waypoint.action_code.lower()
        self.status = "run"

    def _finish_action(self, waypoint: Waypoint, pos_ned: np.ndarray) -> None:
        """Completes the action and advances the route."""
        if waypoint.action_code != "GO_WAYPOINT":
            self.route_manager.advance(pos_ned)
        self.status = "done"

    def _is_orbit_action(self, action_code: str) -> bool:
        """Check if the action is related to orbit path."""
        orbit_action_codes = ["ORBIT_UNLIM", "ORBIT_TIME", "ORBIT_TURNS", "ORBIT_ALT"]
        return action_code in orbit_action_codes

    def _set_orbit_path(self, wp: Waypoint) -> None:    
        """Sets the path command for orbit-type actions."""
        # use orbit unlim as base class for other orbit actions
        orbit_action: OrbitUnlimited = wp.action
        self.path_cmd.path_type = "orbit"
        self.path_cmd.path_params = OrbitPathParams(
            wp.ned_coords, orbit_action.radius, orbit_action.direction
        )

        if wp.action_code == "ORBIT_ALT":
            self._set_orbit_alt_path(wp)

        self.path_cmd.is_new_path = True

    def _set_orbit_alt_path(self, waypoint: Waypoint) -> None:
        """Set the orbit center and modify the waypoint's altitude."""
        orbit_alt_action: OrbitAlt = waypoint.action
        self.path_cmd.path_params.center[2] = -orbit_alt_action.altitude
        self.route_manager.set_waypoint_coords(self.path_cmd.path_params.center)

    def _execute_action(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        """Executes the appropriate action based on the action code."""
        action_map = {
            "ORBIT_UNLIM": self._execute_orbit_unlimited,
            "ORBIT_TIME": self._execute_orbit_time,
            "ORBIT_TURNS": self._execute_orbit_turns,
            "ORBIT_ALT": self._execute_orbit_alt,
            "GO_WAYPOINT": self._execute_go_waypoint,
            "SET_AIRSPEED": self._execute_set_airspeed,
        }
        if wp.action_code in action_map:
            action_map[wp.action_code](wp, pos_ned, dt)
        else:
            raise ValueError(
                f"Invalid action code: {wp.action_code}, for waypoint id: {wp.id}!"
            )

    def _execute_orbit_unlimited(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitUnlimited = wp.action
        action.execute()
        self.active_action_status = self._orbit_info(action.radius, action.direction)

    def _execute_orbit_time(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitTime = wp.action
        action.execute(dt)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit time: {action.time:.2f} s, Elapsed Time: {action.elapsed_time:.2f} s, "
            f"Progress: {action.elapsed_time/action.time*100.0:.2f}%"
        )

    def _execute_orbit_turns(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitTurns = wp.action
        r = pos_ned - wp.ned_coords
        ang_pos = np.arctan2(r[1], r[0])
        action.execute(ang_pos)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit turns: {action.turns}, Completed turns: {action.completed_turns:.1f}, "
            f"Progress: {action.completed_turns/action.turns*100.0:.2f}%"
        )

    def _execute_orbit_alt(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitAlt = wp.action
        alt = -pos_ned[2]
        action.execute(alt)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit altitude: {action.altitude:.1f} m, Current altitude: {alt:.1f} m, "
            f"Error: {alt - action.altitude:.2f} m"
        )

    def _execute_go_waypoint(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: GoWaypoint = wp.action
        if action.has_pending_jumps():
            self.route_manager.set_target_waypoint(action.wp_id)
        action.execute()
        self.active_action_status = (
            f"Repeat: {action.repeat}, Repeat Count: {action.repeat_count}"
        )

    def _execute_set_airspeed(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: SetAirspeed = wp.action
        self.flight_cmd.airspeed = action.airspeed
        action.execute()
        self.active_action_status = f"Target airspeed: {action.airspeed:.2f} m/s"

    def _orbit_info(self, radius: float, dir: int) -> str:
        dir_str = "CW" if dir > 0 else "CCW"
        return f"Radius: {radius:.1f} m, Direction: {dir_str}"
