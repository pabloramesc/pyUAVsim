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

from dataclasses import asdict, astuple, dataclass

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.path_follower import (
    PathFollower,
    BasePathFollower,
    BasePathParams,
    LinePathFollower,
    LinePathParams,
    OrbitPathFollower,
    OrbitPathParams,
)
from simulator.autopilot.path_navigator import (
    DubinPathNavigator,
    FilletPathNavigator,
    LinePathNavigator,
    PathNavCommand,
    PathNavigator,
)
from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.waypoint_actions import (
    GoWaypoint,
    OrbitAlt,
    OrbitTime,
    OrbitTurns,
    OrbitUnlimited,
    SetAirspeed,
)
from simulator.autopilot.waypoints import Waypoint, WaypointsList

NAV_PATH_TYPES = ["lines", "fillets", "dubins"]
MISSION_CONTROL_STATUS = ["wait", "init", "nav", "exe", "end", "fail"]
ACTIONS_MANAGER_STATUS = ["wait", "exe", "done", "fail"]


@dataclass
class FlightCommand:
    """
    Data structure for holding path following output commands.

    Attributes
    ----------
    target_altitude : float
        The target altitude for the vehicle.
    target_course : float
        The target course angle for the vehicle.
    target_airspeed : float
        The target airspeed for the vehicle.
    """

    target_altitude: float = None
    target_course: float = None
    target_airspeed: float = None


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

        self.path_follower = PathFollower(config)
        self.route_manager = RouteManager(config)
        self.path_navigator = self._create_path_navigator(nav_type)

        self.waypoints: WaypointsList = None
        self.flight_cmd: FlightCommand = None
        self.path_cmd: PathNavCommand = None

        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False
        self.active_waypoint: Waypoint = None

        self.pos_ned: np.ndarray = None
        self.course: float = None

    def reset(self) -> None:
        self.path_follower.reset()
        self.route_manager.reset()

        self.waypoints: WaypointsList = None
        self.flight_cmd: FlightCommand = None
        self.path_cmd: PathNavCommand = None

        self.status = "wait"
        self.is_on_wait_orbit = False
        self.is_action_running = False

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
        self.set_waypoints(wps_list)
        self.flight_cmd = FlightCommand(h, chi, Va)
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
        self.pos_ned = pos_ned
        self.course = course

        if self.status == "wait":
            raise Exception("Initialize mission control before update!")
        if self.status == "init":
            self.route_manager.restart(pos_ned)

        waypoint = (
            self.waypoints.get_waypoint(self.route_manager.wp_target + 1)
            if self.waypoints
            else None
        )
        if waypoint is None:
            raise Exception("Load waypoints before update!")
        else:
            self.active_waypoint = waypoint

        if self.is_action_running:
            self._execute_waypoint_action(waypoint, pos_ned, course, _dt)
            if waypoint.action.is_done():
                self.is_action_running = False
                self.route_manager.advance(pos_ned)
            self.status = "exe"
        else:
            self._update_navigation(pos_ned, course)
            self.status = "nav"

        if (waypoint.action is not None) and (
            self.route_manager.is_on_waypoint(pos_ned, waypoint.id)
        ):
            self.is_action_running = True

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
        if nav_type is None:
            raise ValueError("None is not a path type!")
        elif nav_type == "lines":
            return LinePathNavigator(self.config, self.route_manager)
        elif nav_type == "fillets":
            return FilletPathNavigator(self.config, self.route_manager)
        elif nav_type == "dubins":
            return DubinPathNavigator(self.config, self.route_manager)
        else:
            raise ValueError(f"Invalid path navigator type: {nav_type}!")

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

    def _update_navigation(self, pos_ned: np.ndarray, course: float) -> None:
        """Update path command from path navigator or enter in wait orbit,
        depending on route manager status"""
        if self.route_manager.status in ["init", "run"]:
            self.path_cmd = self.path_navigator.navigate_path(pos_ned, course)
            if self.path_cmd.is_new_path:
                self._execute_path_command(self.path_cmd)
        elif self.route_manager.status in ["end", "fail"]:
            self.enter_wait_orbit(pos_ned)
            self.status = self.route_manager.status
        else:
            raise ValueError(
                f"Not valid route manager status: {self.route_manager.status}!"
            )

    def _execute_path_command(self, path_cmd: PathNavCommand) -> None:
        """Set path follower params for line or orbit following according to path command.
        Then update new path flag."""
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

    def _execute_waypoint_action(
        self, wp: Waypoint, pos_ned: np.ndarray, course: float, dt: float
    ) -> None:
        action_map = {
            "ORBIT_UNLIM": self._execute_orbit_unlimited,
            "ORBIT_TIME": self._execute_orbit_time,
            "ORBIT_TURNS": self._execute_orbit_turns,
            "ORBIT_ALT": self._execute_orbit_alt,
            "GO_WAYPOINT": self._execute_go_waypoint,
            "SET_AIRSPEED": self._execute_set_airspeed,
        }
        action = wp.action
        action_code = wp.action_code
        if action_code in action_map:
            action_map[action_code](wp, action, pos_ned, course, dt)
        else:
            raise ValueError(
                f"Invalid action code: {action_code}, for waypoint id: {wp.id}!"
            )

    def _execute_orbit_unlimited(
        self,
        wp: Waypoint,
        action: OrbitUnlimited,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.enter_wait_orbit(wp.ned_coords)
        self.active_follower = self.orbit_follower
        self.status = "orbit_unlim"
        action.execute()

    def _execute_orbit_time(
        self,
        wp: Waypoint,
        action: OrbitTime,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.enter_wait_orbit(wp.ned_coords)
        self.active_follower = self.orbit_follower
        self.status = "orbit_time"
        action.execute(dt)

    def _execute_orbit_turns(
        self,
        wp: Waypoint,
        action: OrbitTurns,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.enter_wait_orbit(wp.ned_coords)
        self.active_follower = self.orbit_follower
        ang_pos = self.orbit_follower.get_angular_position(pos_ned)
        self.status = "orbit_turns"
        action.execute(ang_pos)

    def _execute_orbit_alt(
        self,
        wp: Waypoint,
        action: OrbitAlt,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.enter_wait_orbit(np.array([wp.pn, wp.pe, -action.altitude]))
        self.active_follower = self.orbit_follower
        alt = -pos_ned[2]
        self.status = "orbit_alt"
        action.execute(alt)

    def _execute_go_waypoint(
        self,
        wp: Waypoint,
        action: GoWaypoint,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.route_manager.set_target_waypoint(action.waypoint_id)
        self.status = "go_waypoint"
        action.execute()

    def _execute_set_airspeed(
        self,
        wp: Waypoint,
        action: SetAirspeed,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.flight_cmd.target_airspeed = action.airspeed
        self.status = "set_airspeed"
        action.execute()
