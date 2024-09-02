"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.path_following import LineFollower, OrbitFollower


class PathManager:
    """
    Manages the vehicle's path by handling waypoints and controlling the
    path following behavior based on the selected path type.

    Attributes
    ----------
    config : AutopilotConfig
        Configuration parameters for the autopilot.
    path_types : str
        The type of path management strategy ('lines', 'fillets', or 'dubins').
    line_follower : LineFollower
        Instance for line following guidance.
    orbit_follower : OrbitFollower
        Instance for orbit following guidance.
    waypoints_coords : np.ndarray
        Array of waypoints' coordinates.
    target_waypoint : int
        Index of the current target waypoint.
    current_mode : str
        Current operating mode of the path manager ('init', 'run', or 'end').
    """

    def __init__(self, config: AutopilotConfig, path_types: str = "lines") -> None:
        self.config = config
        self.path_types = path_types

        self.line_follower = LineFollower(config)
        self.orbit_follower = OrbitFollower(config)

        self.waypoints_coords: np.ndarray = None
        self.target_waypoint: int = 0  # initial waypoint index
        self.current_mode: str = "init"

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        """
        Set the waypoints for the path manager.

        Parameters
        ----------
        waypoints : np.ndarray
            A N-by-3 numpy array representing waypoints' coordinates,
            where N is the number of waypoints.

        Raises
        ------
        ValueError
            If `waypoints` is not a N-by-3 array or if fewer than 3 waypoints are provided.
        """
        if (
            not isinstance(waypoints, np.ndarray)
            or waypoints.ndim != 2
            or waypoints.shape[1] != 3
        ):
            raise ValueError(
                "waypoints array must be a N-by-3 size numpy.ndarray (where N is the number of WPs)!"
            )
        if waypoints.shape[0] < 3:
            raise ValueError("there must be provided 3 waypoints at least!")

        self.waypoints_coords = np.zeros((waypoints.shape[0] + 1, 3))
        self.waypoints_coords[1:] = waypoints
        self.target_waypoint = 1
        self.current_mode = "init"

    def set_target_waypoint(self, waypoint_id: int) -> None:
        """
        Set the target waypoint index.

        Parameters
        ----------
        waypoint_id : int
            The index of the target waypoint (1-based).

        Raises
        ------
        ValueError
            If `waypoint_id` is out of the valid range.
        """
        if waypoint_id < 1 or waypoint_id > self.waypoints_coords.shape[0]:
            raise ValueError(
                f"waypoint_id must be between 1 and {self.waypoints_coords.shape[0]}"
            )
        self.target_waypoint = waypoint_id

    def restart_mission(self) -> None:
        """
        Restart the mission, resetting the path manager to initial conditions.
        """
        self.target_waypoint = 1
        self.waypoints_coords[0] = np.zeros(3)  # reset the aux initial waypoint
        self.current_mode = "init"

    def manage_paths(self, pos_ned: np.ndarray, course: float) -> tuple:
        """
        Manage the paths based on the selected path type.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            Guidance values (e.g., course and altitude references) for the vehicle.
        """
        if self.path_types is "lines":
            self._manage_line_paths(pos_ned, course)
        elif self.path_types is "fillets":
            self._manage_fillet_paths(pos_ned, course)
        elif self.path_types is "dubins":
            self._manage_line_paths(pos_ned, course)

    def update(self, pos_ned: np.ndarray, course: float) -> tuple:
        """
        Update the path manager based on the current vehicle position and course.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            Guidance values (e.g., course and altitude references) based on the current mode.
        """
        if self.current_mode == "init":
            if self.is_on_waypoint(1, pos_ned):
                self.target_waypoint = 2
                self.current_mode = "run"
            else:
                self.target_waypoint = 1
                self.waypoints_coords[0] = pos_ned
                self.current_mode = "run"

        if self.current_mode == "run":
            if self.target_waypoint >= self.waypoints_coords.shape[0]:
                self.current_mode = "end"
            else:
                return self.manage_paths(pos_ned, course)

        if self.current_mode == "end":
            self.orbit_follower.set_orbit(
                self.waypoints_coords[-1, :], self.config.wait_orbit_radius, 1
            )
            return self.orbit_follower.update(pos_ned, course)

    def get_status_string(self, pos_ned: np.ndarray) -> str:
        """
        Get a status string representing the current mode and distance to the target waypoint.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.

        Returns
        -------
        str
            A string representing the current mode and distance to the target waypoint.
        """
        if (
            self.current_mode == "end"
            or self.target_waypoint >= self.waypoints_coords.shape[0]
        ):
            return f"END R{self.config.wait_orbit_radius}"
        return f"WP{self.target_waypoint} ({self.get_3D_distance_to_waypoint(pos_ned):.0f}m)"

    def get_3D_distance_to_waypoint(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the 3D distance to the current target waypoint.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.

        Returns
        -------
        float
            The 3D distance to the target waypoint.
        """
        return np.linalg.norm(self.waypoints_coords[self.target_waypoint, :] - pos_ned)

    def get_2D_distance_to_waypoint(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the 2D distance to the current target waypoint.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.

        Returns
        -------
        float
            The 2D distance to the target waypoint.
        """
        return np.linalg.norm(
            self.waypoints_coords[self.target_waypoint, 0:2] - pos_ned[0:2]
        )

    def get_waypoint_on(self, pos_ned: np.ndarray) -> float:
        """
        Check if the vehicle is within the radius of the current target waypoint.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.

        Returns
        -------
        float
            The index of the waypoint if the vehicle is within the waypoint radius, otherwise -1.
        """
        if (
            self.get_2D_distance_to_waypoint(pos_ned)
            < self.config.waypoint_default_radius
        ):
            return self.target_waypoint
        return -1

    def is_on_waypoint(self, waypoint_id: int, pos_ned: np.ndarray) -> bool:
        """
        Determine if the vehicle is within the radius of a specific waypoint.

        Parameters
        ----------
        waypoint_id : int
            The index of the waypoint to check.
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.

        Returns
        -------
        bool
            True if the vehicle is within the waypoint radius, otherwise False.
        """
        dist = np.linalg.norm(self.waypoints_coords[waypoint_id, :] - pos_ned)
        return dist < self.config.waypoint_default_radius

    def _manage_line_paths(
        self, pos_ned: np.ndarray, course: float
    ) -> tuple[float, float]:
        """
        Manage and provide guidance for line path following.

        This method calculates the guidance values for the vehicle to follow a line path
        between waypoints. It updates the target waypoint if the vehicle has reached the
        current target waypoint. For the last waypoint, it uses the direction of the line
        segment to provide guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for line path following
            as `(course_ref, altitude_ref)`.
        """
        wp0 = self.waypoints_coords[self.target_waypoint - 1, :]
        wp1 = self.waypoints_coords[self.target_waypoint, :]
        dir_wp0_to_wp1 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))

        # if the target waypoint is not the last
        if self.target_waypoint < self.waypoints_coords.shape[0] - 1:
            wp2 = self.waypoints_coords[self.target_waypoint + 1, :]
            dir_wp1_to_wp2 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))
            normal_dir = (dir_wp0_to_wp1 + dir_wp1_to_wp2) / np.linalg.norm(
                (dir_wp0_to_wp1 + dir_wp1_to_wp2)
            )
        else:
            # if the target waypoint is the last
            normal_dir = dir_wp0_to_wp1

        # if the target waypoint is reached
        if normal_dir.dot(pos_ned - wp1) > 0:
            self.target_waypoint += 1

        line_orig = wp0
        line_dir = dir_wp0_to_wp1
        self.line_follower.set_line(line_orig, line_dir)
        return self.line_follower.guidance(pos_ned, course)

    def _manage_fillet_paths(
        self, pos_ned: np.ndarray, course: float
    ) -> tuple[float, float]:
        """
        Manage and provide guidance for fillet path following.

        This method is a placeholder for managing and providing guidance for fillet
        paths. The fillet paths involve transitions between straight line segments with
        curved segments.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for fillet path following
            as `(course_ref, altitude_ref)`.
        """
        # TODO: implement fillet paths
        raise NotImplementedError("Fillet paths management is not yet implemented.")

    def _manage_dubin_paths(
        self, pos_ned: np.ndarray, course: float
    ) -> tuple[float, float]:
        """
        Manage and provide guidance for Dubins path following.

        This method is a placeholder for managing and providing guidance for Dubins
        paths. Dubins paths involve transitions between straight line segments and circular
        arcs.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for dubin path following
            as `(course_ref, altitude_ref)`.
        """
        # TODO: implement dubin paths
        raise NotImplementedError("Dubins paths management is not yet implemented.")
