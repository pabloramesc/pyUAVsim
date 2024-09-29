"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

"""
 Copyright (c) 2024 Pablo Ramirez Escudero

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.flight_control import FlightCommand
from simulator.autopilot.path_follower import (
    LinePathParams,
    OrbitPathParams,
    PathFollower,
)
from simulator.autopilot.path_navigator import (
    DubinPathNavigator,
    FilletPathNavigator,
    LinePathNavigator,
    PathNavCommand,
    PathNavigator,
)
from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.waypoint_actions_manager import WaypointActionsManager
from simulator.autopilot.waypoints import Waypoint, WaypointsList

NAV_PATH_TYPES = ["lines", "fillets", "dubins"]
MISSION_CONTROL_STATUS = ["wait", "init", "nav", "exe", "end", "fail"]


class MissionControl:
    """
    Manages the vehicle's path by handling waypoints and controlling the
    path-following behavior based on the selected path type.

    Attributes
    ----------
    config : AutopilotConfig
        Configuration parameters for the autopilot.
    nav_type : str
        The type of path navigation strategy ('lines', 'fillets', or 'dubins').
    """

    def __init__(
        self, dt: float, config: AutopilotConfig, nav_type: str = "lines"
    ) -> None:
        """
        Initialize the MissionControl with autopilot configuration and path type.

        Parameters
        ----------
        dt : float
            The time step of the simulation.
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        nav_type : str, optional
            The type of path navigation strategy ('lines', 'fillets', or 'dubins').
        """
        self.dt = dt
        self.config = config
        self.nav_type = nav_type

        self.flight_cmd = FlightCommand()
        self.path_cmd = PathNavCommand()
        
        self.path_follower = PathFollower(config)
        self.route_manager = RouteManager(config)
        self.path_navigator = self._create_path_navigator(nav_type)
        self.actions_manager = WaypointActionsManager(
            self.route_manager, self.flight_cmd
        )

        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False
        self.target_waypoint: Waypoint = None

        self.t: float = None
        self.pos_ned: np.ndarray = None
        self.course: float = None

    def reset(self) -> None:
        self._reset_managers()
        self._reset_states()

    def initialize(
        self, wps_list: WaypointsList, Va: float, h: float, chi: float
    ) -> FlightCommand:
        """
        Set waypoints and initialize mission control.

        Parameters
        ----------
        wps_list : WaypointsList
            The waypoints list
        Va : float
            Initial target airspeed.
        h : float
            Initial target altitude.
        chi : float
            Initial target course angle.

        Returns
        -------
        FlightCommand
            The initialized flight command.
        """
        self.reset()
        self.t = 0.0
        self._set_waypoints(wps_list)
        self._set_initial_flight_cmd(Va, h, chi)
        self.status = "init"
        return self.flight_cmd

    def update(
        self, pos_ned: np.ndarray, course: float, dt: float = None
    ) -> FlightCommand:
        """
        Update guidance commands based on current position and course.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in North-East-Down (NED) coordinates.
        course : float
            The current heading/course angle of the vehicle.
        dt : float, optional
            Time step for updates.

        Returns
        -------
        FlightCommand
            The updated flight command from path follower.
        """
        _dt = dt or self.dt
        self.t += _dt
        self.pos_ned, self.course = pos_ned, course

        self._validate_status()
        self._update_route_manager_if_needed(pos_ned)
        self._update_waypoint_target()
        self._manage_actions_and_navigation(pos_ned, course, _dt)

        course_ref, altitude_ref = self.path_follower.update(pos_ned, course)
        self._update_flight_command(course_ref, altitude_ref)
        return self.flight_cmd

    def enter_wait_orbit(self, pos_ned: np.ndarray) -> None:
        """
        Set the vehicle in a holding pattern at the desired position.

        Parameters
        ----------
        pos_ned : np.ndarray
            The orbit center position in North-East-Down (NED) coordinates.
        """
        if self.is_on_wait_orbit:
            return
        self._enter_orbit_path(pos_ned)
        self.is_on_wait_orbit = True

    def _reset_managers(self) -> None:
        """Reset all managers and command objects."""
        self.path_follower.reset()
        self.route_manager.reset()
        self.actions_manager.reset()
        self.flight_cmd.reset()
        self.path_cmd.reset()

    def _reset_states(self) -> None:
        """Reset internal state variables."""
        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False
        self.target_waypoint = None
        self.t = None
        self.pos_ned = None
        self.course = None

    def _set_waypoints(self, wps: WaypointsList) -> None:
        """Load waypoints into the route manager."""
        self.route_manager.set_waypoints(wps)

    def _set_initial_flight_cmd(self, Va: float, h: float, chi: float) -> None:
        """Set initial flight command parameters."""
        self.flight_cmd.altitude = h
        self.flight_cmd.course = chi
        self.flight_cmd.airspeed = Va

    def _validate_status(self) -> None:
        """Ensure mission status is valid."""
        if self.status not in MISSION_CONTROL_STATUS:
            raise ValueError(f"Invalid mission control status: {self.status}!")
        if self.status == "wait":
            raise Exception("Mission control must be initialized before update!")

    def _update_route_manager_if_needed(self, pos_ned: np.ndarray) -> None:
        """Restart route manager if in initialization phase."""
        if self.status == "init":
            self.route_manager.restart(pos_ned)

    def _update_waypoint_target(self) -> None:
        """Get the next target waypoint from the route manager."""
        self.target_waypoint = self.route_manager.get_waypoint()

    def _manage_actions_and_navigation(
        self, pos_ned: np.ndarray, course: float, dt: float
    ) -> None:
        """Handle both waypoint actions and path navigation based on current status."""
        self._update_actions_manager(pos_ned, dt)
        if not self.is_action_running:
            self._update_navigation(pos_ned, course)

    def _enter_orbit_path(self, pos_ned: np.ndarray) -> None:
        """Configure the path follower to enter an orbit mode."""
        orbit_center = pos_ned
        orbit_radius = self.config.wait_orbit_radius
        params = OrbitPathParams(orbit_center, orbit_radius)
        self.path_follower.follow_orbit(params)
        self._update_path_cmd("orbit", params)

    def _update_flight_command(self, course_ref: float, altitude_ref: float) -> None:
        """Update flight command with new course and altitude references."""
        self.flight_cmd.altitude = altitude_ref
        self.flight_cmd.course = course_ref

    def _create_path_navigator(self, nav_type: str) -> PathNavigator:
        """Create a line, fillet or dubin path navigator according to the type string provided."""
        if nav_type == "lines":
            return LinePathNavigator(self.config, self.route_manager)
        elif nav_type == "fillets":
            return FilletPathNavigator(self.config, self.route_manager)
        elif nav_type == "dubins":
            return DubinPathNavigator(self.config, self.route_manager)
        else:
            raise ValueError(f"Not valid path navigator type: {nav_type}!")

    def _update_navigation(self, pos_ned: np.ndarray, course: float) -> None:
        """Update navigation command based on the current route manager status."""

        if self.route_manager.status in ["init", "run"]:
            path_cmd = self.path_navigator.navigate_path(pos_ned, course)
            self._set_path_if_needed(path_cmd)
            self.status = "nav"

        elif self.route_manager.status in ["end", "fail"]:
            self.enter_wait_orbit(pos_ned)
            self.status = self.route_manager.status

        else:
            raise ValueError(
                f"Not valid route manager status: {self.route_manager.status}!"
            )

    def _update_actions_manager(self, pos_ned: np.ndarray, dt: float) -> None:
        """Update actions manager and set action running flag according to action status."""

        path_cmd = self.actions_manager.update(pos_ned, dt)

        if self.actions_manager.status == "wait":
            self.is_action_running = False

        elif self.actions_manager.status == "done":
            self.is_action_running = False
            # force set path for navigation
            path_cmd = self.path_navigator.navigate_path(pos_ned, 0.0)
            self._set_path_follower(path_cmd)

        elif self.actions_manager.status == "run":
            self.is_action_running = True
            self.status = "exe"
            self._set_path_if_needed(path_cmd)

        elif self.actions_manager.status == "fail":
            self.enter_wait_orbit(pos_ned)
            self.is_action_running = False
            self.status = "fail"

        else:
            raise ValueError(
                f"Not valid actions manager status: {self.actions_manager.status}!"
            )

    def _set_path_if_needed(self, path_cmd: PathNavCommand) -> None:
        """Set a new path for the path follower if necessary."""
        if path_cmd.is_new_path:
            self._set_path_follower(path_cmd)

    def _set_path_follower(self, path_cmd: PathNavCommand) -> None:
        """Configure the path follower based on the path type."""

        if path_cmd.path_type is None:
            self.route_manager.force_fail_mode()

        elif path_cmd.path_type == "line":
            params: LinePathParams = path_cmd.path_params
            self.path_follower.follow_line(params)
            self._update_path_cmd("line", params)

        elif path_cmd.path_type == "orbit":
            params: OrbitPathParams = path_cmd.path_params
            self.path_follower.follow_orbit(params)
            self._update_path_cmd("orbit", params)

        else:
            raise ValueError(f"Not valid path type: {path_cmd.path_type}!")

        path_cmd.is_new_path = False

    def _update_path_cmd(self, path_type: str, params) -> PathNavCommand:
        """Helper method to update the path command."""
        self.path_cmd.path_type = path_type
        self.path_cmd.path_params = params
        self.path_cmd.is_new_path = False
