"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig

ROUTE_MANAGER_STATUS = ["wait", "init", "run", "end", "fail"]


class RouteManager:
    def __init__(self, config: AutopilotConfig = None) -> None:
        """
        Initialize the RouteManager.

        Parameters
        ----------
        config : AutopilotConfig, optional
            Configuration parameters for the autopilot, by default None.
        """
        self.config = config or AutopilotConfig()
        self.wp_coords: np.ndarray = None
        self.wp_target: int = 0 # initial waypoint index
        self.status: str = "wait"

    def reset(self) -> None:
        """
        Reset the route manager and deletes waypoints.

        Deletes all waypoints and set the first waypoint to the origin `[0, 0, 0]`.
        Then, resets the target index to `0` and set status to `init`.
        """
        self.wp_coords = None
        self.wp_target = 0  # initial waypoint index
        self.status = "wait"

    def restart(self, pos_ned: np.ndarray) -> None:
        """
        Restart and initialize the route manager based on the current position.

        Sets the initial state for waypoint navigation based on whether the
        vehicle is within the first waypoint area. If inside, advances to
        the second waypoint; otherwise, sets the first waypoint at the
        current position.

        Parameters
        ----------
        pos_ned : np.ndarray
            Current position of the vehicle in North-East-Down (NED) coordinates.
        """
        self._check_status()

        if self.wp_coords is None or self.status == "wait":
            raise Exception("Set waypoints before restart!")
        
        # if the aircraft is not inside the WP0 area
        if not self.is_on_waypoint(pos_ned, wp_id=0):
            # set an initial waypoint at current aircraft position
            self.wp_coords = np.vstack((pos_ned, self.wp_coords))

        self.wp_target = 1
        self.status = "run"

    def advance(self, pos_ned: np.ndarray, check_area: bool = False) -> None:
        """
        Advance to the next waypoint by incrementing `wp_target`.

        If `check_area` flag is True it updates the target waypoint
        when the vehicle is within the current waypoint area.
        If not within the area, `status` is set to 'fail'.
        If the target waypoint is the last, `status` is set to 'end'.

        Parameters
        ----------
        pos_ned : np.ndarray
            Current position of the vehicle in North-East-Down (NED) coordinates.
        """
        self._check_status()

        if self.status != "run":
            return

        self.dist_target = self.get_distance_to_waypoint(pos_ned)
        
        # if the aircraft is not inside the target waypoint area
        if check_area and not self.is_on_waypoint(pos_ned, self.wp_target, is_3d=False):
            self.status = "fail"

        # if the target waypoint is the last
        elif self.is_target_last():
            self.status = "end"

        # advance to the next waypoint
        else:
            self.wp_target += 1

    def set_waypoints(self, wps: np.ndarray) -> None:
        """
        Set the waypoints for navigation.

        Restarts the route manager to initial conditions and set the new waypoints.
        The waypoints array must be a 2D array with three columns representing
        NED coordinates (North, East, Down).

        Parameters
        ----------
        wps : np.ndarray
            A N-by-3 numpy array representing waypoints' coordinates,
            where N is the number of waypoints.

        Raises
        ------
        ValueError
            If `wps` is not a N-by-3 array or if fewer than 3 waypoints are provided.
        """
        if not isinstance(wps, np.ndarray) or wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("'wps' array must be a N-by-3 size np.ndarray!")

        if wps.shape[0] < 3:
            raise ValueError("'wps' must contain at least 3 waypoints!")

        self.reset()
        self.wp_coords = wps
        self.status = "init"

    def set_target_waypoint(self, wp_id: int) -> None:
        """
        Set the target waypoint index.

        Parameters
        ----------
        wp_id : int
            The index of the target waypoint (1-based).

        Raises
        ------
        ValueError
            If `wp_id` is out of the valid range of waypoints.
        """
        if wp_id < 1 or wp_id > self.wp_coords.shape[0]:
            raise ValueError(f"'wp_id' must be between 1 and {self.wp_coords.shape[0]}")
        self.wp_target = wp_id

    def set_waypoint_coords(self, coords: np.ndarray, wp_id: int = None) -> None:
        """
        Set coordinates for a specific waypoint.

        Parameters
        ----------
        coords : np.ndarray
            3-size array with the coordinates for the selected
            waypoint refered to NED frame in meters.
        wp_id : int, optional
            The index of the waypoint. If not provided, it uses the target waypoint.
            Default is None.
        """
        _index = wp_id if wp_id is not None else self.wp_target
        self.wp_coords[_index, :] = coords

    def get_waypoint_coords(self, wp_id: int = None) -> np.ndarray:
        """
        Get coordinates from a specific waypoint.

        Parameters
        ----------
        wp_id : int, optional
            The index of the waypoint. If not provided, it uses the target waypoint.
            Default is None.

        Returns
        -------
        np.ndarray
            3-size array with the coordinates for the selected
            waypoint refered to NED frame in meters.
        """
        _index = wp_id if wp_id is not None else self.wp_target
        return self.wp_coords[_index, :]

    def get_path_waypoints(self) -> tuple:
        """
        Get coordinates of waypoints related to the current path.

        Provides the coordinates of the waypoint before the target,
        the target waypoint, and the waypoint after the target.

        Returns
        -------
        tuple
            A tuple containing:
            - wp0: Waypoint before the target (or None if not applicable)
            - wp1: The current target waypoint
            - wp2: Waypoint after the target (or None if the target is the last)
        """
        wp0 = (
            self.get_waypoint_coords(self.wp_target - 1) if self.wp_target > 0 else None
        )
        wp1 = self.get_waypoint_coords(self.wp_target)
        wp2 = (
            self.get_waypoint_coords(self.wp_target + 1)
            if not self.is_target_last()
            else None
        )
        return wp0, wp1, wp2

    def get_distance_to_waypoint(
        self, pos_ned: np.ndarray, wp_id: int = None, is_3d: bool = True
    ) -> float:
        """
        Calculate the distance to a specific waypoint.

        This method computes either the 2D or 3D distance between the vehicle's
        current position and the target waypoint, depending on the value of
        the `is_3d` parameter.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in North-East-Down (NED) coordinates.
        wp_id : int, optional
            The index of the waypoint. If not provided, it uses target waypoint.
            Default is None.
        is_3d : bool, optional
            Whether to calculate the 3D distance (including the vertical dimension).
            If False, the calculation will be limited to 2D (horizontal plane only).
            Default is True.

        Returns
        -------
        float
            The distance to the target waypoint in meters, computed in 2D or 3D as specified.
        """
        _index = wp_id if wp_id is not None else self.wp_target
        r = self.wp_coords[_index, :] - pos_ned
        return np.linalg.norm(r if is_3d else r[0:2])

    def is_on_waypoint(
        self, pos_ned: np.ndarray, wp_id: int = None, is_3d: bool = True
    ) -> bool:
        """
        Determine if the vehicle is within the radius of a specific waypoint.

        This method checks whether the vehicle's current position is within the specified
        radius of the given waypoint, taking into account either 2D or 3D distances based on
        the `is_3d` parameter.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in North-East-Down (NED) coordinates.
        wp_id : int, optional
            The index of the waypoint. If not provided, it uses target waypoint.
            Default is None.
        is_3d : bool, optional
            Whether to calculate the 3D distance (including the vertical dimension).
            If False, the calculation will be limited to 2D (horizontal plane only).
            Default is True.

        Returns
        -------
        bool
            True if the vehicle is within the specified radius of the waypoint, otherwise False.
        """
        _index = wp_id if wp_id is not None else self.wp_target
        dist = self.get_distance_to_waypoint(pos_ned, _index, is_3d)
        return dist < self.config.wp_default_radius

    def is_target_last(self) -> bool:
        """
        Check if the current target waypoint is the last in the sequence.

        Returns
        -------
        bool
            `True` if the current target waypoint is the last, otherwise `False`.
        """
        return self.wp_target >= self.wp_coords.shape[0] - 1

    def force_fail_mode(self) -> None:
        self.status = "fail"

    def _check_status(self) -> None:
        if self.status not in ROUTE_MANAGER_STATUS:
            raise ValueError(f"Not valid route manager status: {self.status}!")