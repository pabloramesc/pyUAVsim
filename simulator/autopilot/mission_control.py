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

        self.waypoints: WaypointsList = None
        self.flight_cmd: FlightCommand = None
        self.path_cmd: PathNavCommand = None

        self.path_follower = PathFollower(config)
        self.route_manager = RouteManager(config)
        self.actions_manager = WaypointActionsManager(
            self.route_manager, self.flight_cmd
        )
        self.path_navigator = self._create_path_navigator(nav_type)

        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False
        self.target_waypoint: Waypoint = None

        self.t: float = None
        self.pos_ned: np.ndarray = None
        self.course: float = None

    def reset(self) -> None:
        self.path_follower.reset()
        self.route_manager.reset()
        self.actions_manager.reset()

        self.waypoints: WaypointsList = None
        self.flight_cmd: FlightCommand = None
        self.path_cmd: PathNavCommand = None

        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False
        self.target_waypoint = None

        self.t = None
        self.pos_ned = None
        self.course = None

    def initialize(
        self, wps_list: WaypointsList, Va: float, h: float, chi: float
    ) -> FlightCommand:
        """
        Reset and initialize guidance commands and parameters.

        Parameters
        ----------
        wps_list : WaypointsList
            The waypoints list
        Va : float
            Target airspeed.
        h : float
            Target altitude.
        chi : float
            Target course angle.

        Returns
        -------
        FlightCommand
            The initialized flight command.
        """
        self.t = 0.0
        self.set_waypoints(wps_list)
        self.flight_cmd = FlightCommand(
            target_altitude=h, target_course=chi, target_airspeed=Va
        )
        self.path_cmd = PathNavCommand()
        self.status = "init"
        self.path_follower.reset()
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
        self.pos_ned = pos_ned
        self.course = course

        if self.status not in MISSION_CONTROL_STATUS:
            raise ValueError(f"Not valid mission control status: {self.status}!")

        elif self.status == "wait":
            raise Exception("Initialize mission control before update!")

        elif self.status == "init":
            self.route_manager.restart(pos_ned)

        self.target_waypoint = self._get_target_waypoint()

        self._update_actions_manager(self.target_waypoint, pos_ned, _dt)
        if not self.is_action_running:
            self._update_navigation(pos_ned, course)

        course_ref, altitude_ref = self.path_follower.update(pos_ned, course)
        self.flight_cmd.target_altitude = altitude_ref
        self.flight_cmd.target_course = course_ref
        return course_ref, altitude_ref

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

    def set_waypoints(self, wps_list: WaypointsList) -> None:
        """
        Load waypoints from a waypoint list and set them in the route manager.

        Parameters
        ----------
        wps_list : WaypointsList
            The waypoints list
        """
        wps = wps_list.get_waypoint_coords()
        self.route_manager.set_waypoints(wps)
        self.waypoints = wps_list

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
        """Update path command from path navigator or enter in wait orbit,
        depending on route manager status. Then update mission status."""

        if self.route_manager.status in ["init", "run"]:
            path_cmd = self.path_navigator.navigate_path(pos_ned, course)

            if path_cmd.is_new_path:
                self._set_path_follower(path_cmd)

            self.status = "nav"

        elif self.route_manager.status in ["end", "fail"]:
            self.enter_wait_orbit(pos_ned)
            self.status = self.route_manager.status

        else:
            raise ValueError(
                f"Not valid route manager status: {self.route_manager.status}!"
            )

    def _update_actions_manager(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        """Update actions manager and set action running flag according to action status"""

        self.actions_manager.update(wp, pos_ned, dt)

        if self.actions_manager.status in ["wait", "done"]:
            self.is_action_running = False

        elif self.actions_manager.status == "run":
            self.is_action_running = True
            self.status = "exe"

        elif self.actions_manager.status == "fail":
            self.enter_wait_orbit(pos_ned)
            self.is_action_running = False
            self.status = "fail"

        else:
            raise ValueError(
                f"Not valid actions manager status: {self.actions_manager.status}!"
            )

    def _enter_orbit_path(self, pos_ned: np.ndarray) -> None:
        """Set path follower in orbit mode and update path command."""

        orbit_center = pos_ned
        orbit_radius = self.config.wait_orbit_radius
        params = OrbitPathParams(orbit_center, orbit_radius)
        self.path_follower.follow_orbit(params)

        self.path_cmd = PathNavCommand()
        self.path_cmd.path_type = "orbit"
        self.path_cmd.path_params = params
        self.path_cmd.is_new_path = False

    def _set_path_follower(self, path_cmd: PathNavCommand) -> None:
        """Set path follower params for line or orbit following according to path command.
        Then update new path flag."""

        self.path_cmd = path_cmd

        if path_cmd.path_type is None:
            self.route_manager.force_fail_mode()

        elif path_cmd.path_type == "line":
            params: LinePathParams = path_cmd.path_params
            self.path_follower.follow_line(params)
            path_cmd.is_new_path = False

        elif path_cmd.path_type == "orbit":
            params: OrbitPathParams = path_cmd.path_params
            self.path_follower.follow_orbit(params)
            path_cmd.is_new_path = False

        else:
            raise ValueError(
                f"Not valid path navigator command path type: {path_cmd.path_type}!"
            )

    def _get_target_waypoint(self) -> Waypoint:
        if self.waypoints is None:
            raise Exception("No waypoints available!")

        if self.route_manager.wp_coords.shape[0] == len(self.waypoints):
            wp = self.waypoints[self.route_manager.wp_target]

        elif self.route_manager.wp_coords.shape[0] == len(self.waypoints) + 1:
            wp = self.waypoints[self.route_manager.wp_target - 1]

        else:
            raise Exception(
                f"Expected {len(self.waypoints)} waypoints in route manager "
                f"(or {len(self.waypoints) + 1} if aux initial waypoint is used), "
                f"but got {self.route_manager.wp_coords.shape[0]}!"
            )

        return wp
