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

from dataclasses import dataclass

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.line_follower import LineFollower
from simulator.autopilot.orbit_follower import OrbitFollower
from simulator.autopilot.path_navigator import (
    DubinPathNavigator,
    FilletPathNavigator,
    LinePathNavigator,
    PathNavigator,
)
from simulator.autopilot.waypoints import Waypoint, WaypointsList
from simulator.autopilot.waypoint_actions import *
from simulator.autopilot.route_manager import RouteManager

PATH_TYPES = ["lines", "fillets", "dubins"]


@dataclass
class NavigationCommand:
    """
    Data structure for holding navigation commands.

    Attributes
    ----------
    target_altitude : float, optional
        The target altitude for the vehicle.
    target_course : float, optional
        The target course angle for the vehicle.
    target_airspeed : float, optional
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
    path_type : str
        The type of path management strategy ('lines', 'fillets', or 'dubins').
    route_manager : RouteManager
        Manages waypoints for the mission.
    orbit_follower : OrbitFollower
        Instance for orbit following guidance.
    path_navigator : PathNavigator
        The navigator object corresponding to the selected path type.
    waypoints : WaypointsList, optional
        List of waypoints to be followed during the mission.
    is_action_running : bool
        Flag indicating if an action is currently being executed.
    dt : float
        Time step for updates.
    nav_cmd : NavigationCommand
        Navigation commands including target altitude, course, and airspeed.
    is_on_wait_orbit : bool
        Flag indicating if the vehicle is in a holding orbit.
    """


    def __init__(self, config: AutopilotConfig, path_type: str = "lines") -> None:
        """
        Initialize the MissionControl with autopilot configuration and path type.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        path_type : str, optional
            The type of path management strategy ('lines', 'fillets', or 'dubins').
        """
        self.config = config
        self.path_type = path_type

        # self.line_follower = LineFollower(config)
        self.orbit_follower = OrbitFollower(config)
        self.is_on_wait_orbit = False

        self.route_manager = RouteManager(config)
        self.path_navigator = self._build_path_navigator(path_type)

        self.waypoints: WaypointsList = None
        self.is_action_running = False

        self.dt = 0.0
        self.nav_cmd = NavigationCommand()

    def initialize(
        self, dt: float, Va: float, h: float, course: float
    ) -> NavigationCommand:
        """
        Initialize navigation commands and parameters.

        Parameters
        ----------
        dt : float
            Time step for updates.
        Va : float
            Target airspeed.
        h : float
            Target altitude.
        course : float
            Target course angle.

        Returns
        -------
        NavigationCommand
            The initialized navigation command.
        """
        self.dt = dt
        self.nav_cmd = NavigationCommand(
            target_airspeed=Va, target_altitude=h, target_course=course
        )
        return self.nav_cmd

    def update(
        self, pos_ned: np.ndarray, course: float, dt: float = None
    ) -> NavigationCommand:
        """
        Update navigation commands based on current position and course.

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
        NavigationCommand
            The updated navigation command.
        """
        _dt = dt or self.dt
        waypoint = (
            self.waypoints.get_waypoint(self.route_manager.wp_target)
            if self.waypoints
            else None
        )

        if waypoint and self.is_action_running:
            self._execute_waypoint_action(waypoint, pos_ned, course, _dt)
            if waypoint.action.is_done():
                self.is_action_running = False
        else:
            self.nav_cmd.target_course, self.nav_cmd.target_altitude = (
                self.update_path_navigation(pos_ned, course)
            )

        return self.nav_cmd

    def update_path_navigation(
        self, pos_ned: np.ndarray, course: float
    ) -> tuple[float, float]:
        """
        Update the path navigation based on the vehicle's current position and course.

        Depending on the waypoints manager status, it either follows the path
        using the navigator or enters a holding pattern using an orbit.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in North-East-Down (NED) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float] or None
            The reference course angle and altitude for path following
            as `(course_ref, altitude_ref)`.
        """
        if self.route_manager.status == "run":
            return self.path_navigator.navigate_path(pos_ned, course)

        elif self.route_manager.status in ["end", "fail"]:
            if not self.is_on_wait_orbit:
                self.enter_wait_orbit()
            return self.orbit_follower.guidance(pos_ned, course)

        else:
            raise ValueError("not valid route manager status!")

    def enter_wait_orbit(self, pos_ned: np.ndarray = None) -> None:
        """
        Set the vehicle in a holding pattern at the desired position.

        Parameters
        ----------
        pos_ned : np.ndarray, optional
            The current position of the obrit center in North-East-Down (NED) coordinates.
            If not provided, the method uses the target waypoint coordinates from the
            route manager.
        """
        orbit_center = pos_ned or self.route_manager.get_waypoint_coords()
        orbit_radius = self.config.wait_orbit_radius
        self.orbit_follower.set_path(orbit_center, orbit_radius)
        self.is_on_wait_orbit = True

    def load_waypoints(self, wps_list: WaypointsList) -> None:
        """
        Load waypoints from a waypoint list and set them in the route manager.

        Parameters
        ----------
        wps_list : WaypointsList
            The waypoints list
        """
        wps = np.array([[wp.pn, wp.pe, wp.h] for wp in wps_list.waypoints])
        self.route_manager.set_waypoints(wps)
        self.waypoints = wps_list

    def _build_path_navigator(self, path_type: str) -> PathNavigator:
        if path_type == "lines":
            return LinePathNavigator(self.config, self.route_manager)
        elif path_type == "fillets":
            return FilletPathNavigator(self.config, self.route_manager)
        elif path_type == "dubins":
            return DubinPathNavigator(self.config, self.route_manager)
        else:
            raise ValueError(f"Invalid path type: {path_type}")

    def _execute_waypoint_action(
        self, waypoint: Waypoint, pos_ned: np.ndarray, course: float, dt: float
    ) -> None:
        action_map = {
            "ORBIT_UNLIM": self._execute_orbit_unlimited,
            "ORBIT_TIME": self._execute_orbit_time,
            "ORBIT_TURNS": self._execute_orbit_turns,
            "ORBIT_ALT": self._execute_orbit_alt,
            "GO_WAYPOINT": self._execute_go_waypoint,
            "SET_AIRSPEED": self._execute_set_airspeed,
        }
        action = waypoint.action
        action_code = waypoint.action_code
        if action_code in action_map:
            action_map[action_code](waypoint, action, pos_ned, course, dt)
        else:
            raise ValueError(
                f"Invalid action code: {action_code}, for waypoint id: {waypoint.id}"
            )

    def _execute_orbit_unlimited(
        self,
        waypoint: Waypoint,
        action: OrbitUnlimited,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        if not self.is_on_wait_orbit:
            self.enter_wait_orbit(waypoint.ned_coords)
        self.nav_cmd.target_course, self.nav_cmd.target_altitude = (
            self.orbit_follower.guidance(pos_ned, course)
        )
        action.execute()

    def _execute_orbit_time(
        self,
        waypoint: Waypoint,
        action: OrbitTime,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        if not self.is_on_wait_orbit:
            self.enter_wait_orbit(waypoint.ned_coords)
        self.nav_cmd.target_course, self.nav_cmd.target_altitude = (
            self.orbit_follower.guidance(pos_ned, course)
        )
        action.execute(dt)

    def _execute_orbit_turns(
        self,
        waypoint: Waypoint,
        action: OrbitTurns,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        if not self.is_on_wait_orbit:
            self.enter_wait_orbit(waypoint.ned_coords)
        self.nav_cmd.target_course, self.nav_cmd.target_altitude = (
            self.orbit_follower.guidance(pos_ned, course)
        )
        ang_pos = self.orbit_follower.get_angular_position(pos_ned)
        action.execute(ang_pos)

    def _execute_orbit_alt(
        self,
        waypoint: Waypoint,
        action: OrbitAlt,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        if not self.is_on_wait_orbit:
            self.enter_wait_orbit(
                np.array([waypoint.pn, waypoint.pe, -action.altitude])
            )
        self.nav_cmd.target_course, self.nav_cmd.target_altitude = (
            self.orbit_follower.guidance(pos_ned, course)
        )
        alt = -pos_ned[2]
        action.execute(alt)

    def _execute_go_waypoint(
        self,
        waypoint: Waypoint,
        action: GoWaypoint,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.route_manager.set_target_waypoint(action.waypoint_id)
        action.execute()

    def _execute_set_airspeed(
        self,
        waypoint: Waypoint,
        action: SetAirspeed,
        pos_ned: np.ndarray,
        course: float,
        dt: float,
    ) -> None:
        self.nav_cmd.target_airspeed = action.airspeed
        action.execute()
