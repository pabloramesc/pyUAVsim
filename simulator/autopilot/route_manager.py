"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.waypoints import Waypoint, WaypointsList

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
        self.waypoints: WaypointsList = None
        self.target_index: int = 0  # initial waypoint index
        self.status: str = "wait"

    def reset(self) -> None:
        """
        Reset the route manager and deletes waypoints.

        Deletes all waypoints and set the first waypoint to the origin `[0, 0, 0]`.
        Then, resets the target index to `0` and set status to `init`.
        """
        self.waypoints = None
        self.target_index = 0  # initial waypoint index
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
            3-size array with aircrfat's position in NED (North, East, Down) coordinates.
        """
        self._check_status()
        self._check_pos_ned(pos_ned)

        if self.waypoints is None or self.status == "wait":
            raise Exception("Set waypoints before restart!")

        # if the aircraft is not inside the WP1 area
        if not self.is_on_waypoint(pos_ned, wp_index=0):
            # set an WP0 (auxiliar initial waypoint) at current aircraft position
            wp = Waypoint(0, *pos_ned)
            self.waypoints.add_waypoint(wp)

        self.target_index = 1
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
        self._check_pos_ned(pos_ned)

        if self.status != "run":
            return

        # if the aircraft is not inside the target waypoint area
        if check_area and not self.is_on_waypoint(
            pos_ned, self.target_index, is_3d=False
        ):
            self.status = "fail"

        # if the target waypoint is the last
        elif self.is_target_last():
            self.status = "end"

        # advance to the next waypoint
        else:
            self.target_index += 1

    def set_target_waypoint(self, wp_id: int) -> None:
        """
        Set the target waypoint index.

        Parameters
        ----------
        wp_id : int
            The ID of the target waypoint.

        Raises
        ------
        ValueError
            If `wp_index` is out of the valid range of waypoints.
        """
        wp_index = self.waypoints.get_waypoint_index(wp_id)
        if not (1 <= wp_index < len(self.waypoints)):
            raise ValueError(
                f"'wp_index' must be between 1 and {len(self.waypoints) - 1}"
            )
        self.target_index = wp_index

    def get_waypoint(self, wp_index: int = None) -> Waypoint:
        """
        Get the desired waypoint.

        Parameters
        ----------
        wp_index : int, optional
            The index of the waypoint. If not provided, it uses the target waypoint.
            Default is None.

        Returns
        -------
        Waypoint
            The desired waypoint.
        """
        _index = wp_index if wp_index is not None else self.target_index
        wp = self.waypoints[_index]
        return wp

    def set_waypoints(self, wps: WaypointsList) -> None:
        """
        Load waypoints from a waypoint list and set them in the route manager.

        Parameters
        ----------
        wps : WaypointsList
            The waypoints list

        Raises
        ------
        ValueError
            If fewer than 3 waypoints are provided.
        """
        if len(wps) < 3:
            raise ValueError("'wps' must contain at least 3 waypoints!")

        self.reset()
        self.waypoints = wps
        self.status = "init"

    def get_waypoint_coords(self, wp_index: int = None) -> np.ndarray:
        """
        Get coordinates from a specific waypoint.

        Parameters
        ----------
        wp_index : int, optional
            The index of the waypoint. If not provided, it uses the target waypoint.
            Default is None.

        Returns
        -------
        np.ndarray
            3-size array with the coordinates for the selected
            waypoint refered to NED frame in meters.
        """
        _index = wp_index if wp_index is not None else self.target_index
        wp = self.waypoints[_index]
        return wp.ned_coords

    def set_waypoint_coords(self, coords: np.ndarray, wp_index: int = None) -> None:
        """
        Set coordinates for a specific waypoint.

        Parameters
        ----------
        coords : np.ndarray
            3-size array with the coordinates for the selected
            waypoint refered to NED frame in meters.
        wp_index : int, optional
            The index of the waypoint. If not provided, it uses the target waypoint.
            Default is None.
        """
        _index = wp_index if wp_index is not None else self.target_index
        wp = self.waypoints[_index]
        wp.ned_coords = coords

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
            self.get_waypoint_coords(self.target_index - 1)
            if self.target_index > 0
            else None
        )
        wp1 = self.get_waypoint_coords(self.target_index)
        wp2 = (
            self.get_waypoint_coords(self.target_index + 1)
            if not self.is_target_last()
            else None
        )
        return wp0, wp1, wp2

    def get_distance_to_waypoint(
        self, pos_ned: np.ndarray, wp_index: int = None, is_3d: bool = True
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
        wp_index : int, optional
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
        _index = wp_index if wp_index is not None else self.target_index
        r = self.get_waypoint_coords(_index) - pos_ned
        return np.linalg.norm(r if is_3d else r[0:2])

    def is_on_waypoint(
        self, pos_ned: np.ndarray, wp_index: int = None, is_3d: bool = True
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
        wp_index : int, optional
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
        _index = wp_index if wp_index is not None else self.target_index
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
        return self.target_index >= len(self.waypoints) - 1

    def force_fail_mode(self) -> None:
        self.status = "fail"

    def _check_status(self) -> None:
        if self.status not in ROUTE_MANAGER_STATUS:
            raise ValueError(f"Not valid route manager status: {self.status}!")

    def _check_pos_ned(self, pos_ned: np.ndarray) -> None:
        if not isinstance(pos_ned, np.ndarray):
            raise ValueError(f"Position must be a numpy array!")
        if pos_ned.shape != (3,):
            raise ValueError(f"Position array must have (3,) shape!")
