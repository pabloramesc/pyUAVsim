"""
 Copyright (c) 2024 Pablo Ramirez Escudero

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import json

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
from simulator.autopilot.waypoints import WaypointsList
from simulator.autopilot.waypoints_manager import WaypointsManager


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
    wps_manager : WaypointsManager
        Manages waypoints for the mission.
    line_follower : LineFollower
        Instance for line following guidance.
    orbit_follower : OrbitFollower
        Instance for orbit following guidance.
    path_navigator : PathNavigator
        The navigator object corresponding to the selected path type.
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
        self.line_follower = LineFollower(config)
        self.orbit_follower = OrbitFollower(config)
        self.wps_manager = WaypointsManager(config)
        self.path_navigator = self._build_path_navigator(path_type)

    def _build_path_navigator(self, path_type: str) -> PathNavigator:
        """
        Construct the appropriate path navigator based on the path type.

        Parameters
        ----------
        path_type : str
            The type of path management strategy ('lines', 'fillets', or 'dubins').

        Returns
        -------
        PathNavigator
            An instance of a PathNavigator subclass corresponding to the path type.

        Raises
        ------
        ValueError
            If the provided path type is not valid.
        """
        if path_type == "lines":
            return LinePathNavigator(self.config, self.wps_manager)
        elif path_type == "fillets":
            return FilletPathNavigator(self.config, self.wps_manager)
        elif path_type == "dubins":
            return DubinPathNavigator(self.config, self.wps_manager)
        else:
            raise ValueError(f"Invalid path type: {path_type}")

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
        if self.wps_manager.status == "run":
            return self.path_navigator.navigate_path(pos_ned, course)
        elif self.wps_manager.status in ["end", "fail"]:
            self.enter_wait_orbit()

    def enter_wait_orbit(self, pos_ned: np.ndarray = None) -> None:
        """
        Set the vehicle in a holding pattern at the current waypoint.

        This method is used when the waypoints manager is in the 'end' or 'fail' state.
        It sets an orbit pattern using the last known waypoint as the center.

        Parameters
        ----------
        pos_ned : np.ndarray, optional
            The current position of the vehicle in North-East-Down (NED) coordinates.
            If not provided, the method uses the target waypoint coordinates from the
            waypoints manager.
        """
        orbit_center = pos_ned or self.wps_manager.get_waypoint_coords(
            self.wps_manager.wp_target
        )
        orbit_radius = self.config.wait_orbit_radius
        self.orbit_follower.set_path(orbit_center, orbit_radius)

    def load_waypoints_from_txt(self, filename: str) -> None:
        """
        Load waypoints from a text file and set them in the WaypointsManager.

        Parameters
        ----------
        filename : str
            The path to the text file containing waypoints data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file content is not in the expected format.
        """
        waypoints_list = WaypointsList()
        waypoints_list.load_from_txt(filename)

        # Convert the list of Waypoints to a format suitable for WaypointsManager
        waypoints = np.array(
            [[wp.pn, wp.pe, wp.h] for wp in waypoints_list.get_waypoints()]
        )

        # Pass the waypoints to WaypointsManager
        self.wps_manager.set_waypoints(waypoints)
