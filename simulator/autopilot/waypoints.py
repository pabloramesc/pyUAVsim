"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import List, Dict, Iterator

import numpy as np

from simulator.autopilot.waypoint_actions import (
    WaypointAction,
    OrbitUnlimited,
    OrbitTime,
    OrbitTurns,
    OrbitAlt,
    GoWaypoint,
    SetAirspeed,
)

WAYPOINT_ACTION_CLASSES: Dict[str, WaypointAction] = {
    "NONE": None,
    "ORBIT_UNLIM": OrbitUnlimited,
    "ORBIT_TIME": OrbitTime,
    "ORBIT_TURNS": OrbitTurns,
    "ORBIT_ALT": OrbitAlt,
    "GO_WAYPOINT": GoWaypoint,
    "SET_AIRSPEED": SetAirspeed,
}

WAYPOINT_ACTION_PARAM_TYPES: Dict[str, List[type]] = {
    "NONE": [],
    "ORBIT_UNLIM": [float, int],  # radius (optional), direction (optional)
    "ORBIT_TIME": [
        float,
        float,
        int,
    ],  # wait time, radius (optional), direction (optional)
    "ORBIT_TURNS": [
        float,
        float,
        int,
    ],  # turns, radius (optional), direction (optional)
    "ORBIT_ALT": [
        float,
        float,
        int,
    ],  # altitude, radius (optional), direction (optional)
    "GO_WAYPOINT": [int, int],  # waypoint id, times to repeat (optional)
    "SET_AIRSPEED": [float],  # airspeed
}

WAYPOINT_ACTION_REQUIRED_PARAMS: Dict[str, int] = {
    "NONE": 0,
    "ORBIT_UNLIM": 0,  # radius (optional), direction (optional)
    "ORBIT_TIME": 1,  # wait time, radius (optional), direction (optional)
    "ORBIT_TURNS": 1,  # turns, radius (optional), direction (optional)
    "ORBIT_ALT": 1,  # altitude, radius (optional), direction (optional)
    "GO_WAYPOINT": 1,  # waypoint id, times to repeat (optional)
    "SET_AIRSPEED": 1,  # airspeed
}


class Waypoint:
    """
    Represents a waypoint with position (pn, pe, h) and an optional action.

    Attributes
    ----------
    id : int
        The unique ID of the waypoint.
    pn : float
        North coordinate of the waypoint.
    pe : float
        East coordinate of the waypoint.
    h : float
        Altitude of the waypoint.
    ned_coords : np.ndarray
        NED (North, East, Down) coordinates of the waypoint.
    action : WaypointAction
        The action associated with this waypoint, if any.
    action_code : str
            Code representing the action to be performed at this waypoint. Default is None.
    params : tuple
        Parameters for the specified action.
    """

    def __init__(
        self, id: int, pn: float, pe: float, h: float, action_code: str = None, *params
    ) -> None:
        """
        Initializes a Waypoint object with position and optional action.

        Parameters
        ----------
        id : int
            Unique identifier for the waypoint.
        pn : float
            North position of the waypoint.
        pe : float
            East position of the waypoint.
        h : float
            Altitude of the waypoint.
        action_code : str, optional
            Code representing the action to be performed at this waypoint. Default is None.
        *params : tuple
            Parameters for the specified action.
        """
        self.id = id
        self.pn = pn
        self.pe = pe
        self.h = h
        self.action_code = action_code
        self.params = params

        self.ned_coords = np.array([pn, pe, -h])
        self.action = self._build_action(action_code, *params)

    def _build_action(self, action_code: str, *params) -> WaypointAction:
        if action_code is None:
            return None
        elif action_code in WAYPOINT_ACTION_CLASSES:
            action_class = WAYPOINT_ACTION_CLASSES[action_code]
            if action_class is None:
                return None
            else:
                return action_class(*params)
        else:
            raise ValueError(f"Not valid action code: {action_code}!")

    def __repr__(self) -> str:
        return (
            f"Waypoint(id={self.id}, pn={self.pn}, pe={self.pe}, h={self.h}, "
            f"action_code={self.action_code}, params={self.params})"
        )


class WaypointsList:
    """
    Manages a collection of waypoints.

    Attributes
    ----------
    waypoints : dict of int: Waypoint
        Dictionary of waypoints, where the key is the waypoint ID and the value is the Waypoint object.
    """

    def __init__(self) -> None:
        # Store waypoints in a dictionary, using their id as the key
        self.waypoints: Dict[int, Waypoint] = {}

    def __iter__(self) -> Iterator[Waypoint]:
        return iter(self.waypoints.values())

    def __getitem__(self, index: int) -> Waypoint:
        # Access waypoint by index (optional, if needed)
        return list(self.waypoints.values())[index]

    def __len__(self) -> int:
        return len(self.waypoints)

    def add_waypoint(self, waypoint: Waypoint) -> None:
        self._validate_waypoint(waypoint)
        # Add the waypoint to the dictionary, using its id as the key
        self.waypoints[waypoint.id] = waypoint

    def get_waypoint(self, id: int) -> Waypoint:
        """
        Retrieves a waypoint by its ID.

        Parameters
        ----------
        id : int
            The ID of the waypoint.

        Returns
        -------
        Waypoint
            The waypoint with the specified ID.

        Raises
        ------
        ValueError
            If the waypoint with the specified ID is not found.
        """
        try:
            # Directly access the waypoint by id
            return self.waypoints[id]
        except KeyError:
            raise ValueError(f"Waypoint with ID {id} not found!")

    def get_waypoint_coords(self) -> np.ndarray:
        """
        Gets the NED (North, East, Down) coordinates of all waypoints in the list.

        Returns
        -------
        np.ndarray
            N-by-3 size array with the NED coordinates of all waypoints,
            where N is the number of waypoints.
        """
        wps = np.zeros((self.__len__(), 3))
        for k, wp in enumerate(self.waypoints.values()):
            wps[k, :] = wp.ned_coords
        return wps

    def _validate_waypoint(self, waypoint: Waypoint) -> None:
        action = waypoint.action_code
        params = waypoint.params

        if waypoint.id <= 0:
            raise ValueError(f"Waypoint ID {waypoint.id} must be greater than zero!")
        
        if waypoint.id in self.waypoints:
            raise ValueError(f"Waypoint ID {waypoint.id} already exists in the list!")

        if action not in WAYPOINT_ACTION_PARAM_TYPES:
            raise ValueError(
                f"Invalid action '{action}' for Waypoint ID: {waypoint.id}! "
                f"Valid actions are: {list(WAYPOINT_ACTION_PARAM_TYPES.keys())}."
            )

        expected_types = WAYPOINT_ACTION_PARAM_TYPES[action]
        required_params = WAYPOINT_ACTION_REQUIRED_PARAMS[action]

        if len(params) < required_params:
            raise ValueError(
                f"Action '{action}' for Waypoint ID {waypoint.id}"
                f"expects at least {required_params} parameter(s), but got {len(params)}!"
            )

        # zip discards extra params if there are more than expected type
        # also discards expected types ir there are more than provided params*
        # *provided params have been already checked to be the minimum required
        for i, (param, expected_type) in enumerate(zip(params, expected_types)):
            if not isinstance(param, expected_type):
                raise ValueError(
                    f"Parameter {i+1} for action '{action}' on Waypoint ID {waypoint.id}"
                    f"should be of type {expected_type.__name__}, but got {type(param).__name__}!"
                )


def load_waypoints_from_txt(filename: str) -> WaypointsList:
    """
    Loads waypoints from a text file and returns a WaypointsList object.

    Parameters
    ----------
    filename : str
        Path to the text file containing the waypoints.

    Returns
    -------
    WaypointsList
        A list of waypoints loaded from the file.

    Raises
    ------
    ValueError
        If a waypoint line is invalid or if the action code is not valid.
    """
    wps_list = WaypointsList()

    with open(filename, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip lines that are comments or empty
        if not line or line.startswith("#"):
            continue

        # Remove comments that start with '#'
        line = line.split("#")[0].strip()

        # Check waypoint number of parameters
        parts = line.strip().split(",")
        if len(parts) < 4:
            raise ValueError(f"Invalid waypoint format: {line}, in line {i+1}!")

        id = int(parts[0].strip())
        pn = float(parts[1].strip())
        pe = float(parts[2].strip())
        h = float(parts[3].strip())

        action_code = parts[4].strip() if len(parts) > 4 else "NONE"
        action_code = action_code or "NONE"
        params = []

        if action_code in WAYPOINT_ACTION_PARAM_TYPES:
            expected_types = WAYPOINT_ACTION_PARAM_TYPES[action_code]
            for i, param_str in enumerate(parts[5:]):
                param_type = expected_types[i]
                params.append(param_type(param_str.strip()))
        else:
            raise ValueError(
                f"Invalid waypoint action code: {action_code}, in line {i+1}!"
            )

        waypoint = Waypoint(id, pn, pe, h, action_code, *params)
        wps_list.add_waypoint(waypoint)

    return wps_list
