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

WAYPOINT_ACTION_PARAMS: Dict[str, List[type]] = {
    "NONE": [],
    "ORBIT_UNLIM": [float],  # Expects: radius (float)
    "ORBIT_TIME": [float, float],  # Expects: wait time (float), radius (float)
    "ORBIT_TURNS": [float, float],  # Expects: turns (float), radius (float)
    "ORBIT_ALT": [float, float],  # Expects: altitude (float), radius (float)
    "GO_WAYPOINT": [int],  # Expects: waypoint id (int)
    "SET_AIRSPEED": [float],  # Expects: airspeed (float)
}


class Waypoint:
    def __init__(
        self, id: int, pn: float, pe: float, h: float, action_code: str = None, *params
    ) -> None:
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
            raise ValueError("not valid action code!")

    def __repr__(self) -> str:
        return (
            f"Waypoint(id={self.id}, pn={self.pn}, pe={self.pe}, h={self.h}, "
            f"action_code={self.action_code}, params={self.params})"
        )


class WaypointsList:
    def __init__(self) -> None:
        self.waypoints: List[Waypoint] = []

    def __iter__(self) -> Iterator[Waypoint]:
        return iter(self.waypoints)

    def __getitem__(self, index: int) -> Waypoint:
        return self.waypoints[index]

    def __len__(self) -> int:
        return len(self.waypoints)

    def add_waypoint(self, waypoint: Waypoint) -> None:
        self.validate_waypoint(waypoint)
        self.waypoints.append(waypoint)

    def get_waypoint(self, id: int) -> Waypoint:
        wp = self.waypoints[id - 1]
        if wp.id == id:
            return wp
        else:
            raise Exception("waypoints list is not sorted by id!")

    def validate_waypoint(self, waypoint: Waypoint) -> None:
        action = waypoint.action_code
        params = waypoint.params

        if waypoint.id <= 0:
            raise ValueError(f"Waypoint ID {waypoint.id} must be positive.")
        if any(wp.id == waypoint.id for wp in self.waypoints):
            raise ValueError(f"Waypoint ID {waypoint.id} already exists in the list.")

        if action not in WAYPOINT_ACTION_PARAMS:
            raise ValueError(
                f"Invalid action '{action}' for Waypoint ID {waypoint.id}."
                f"Valid actions are: {list(WAYPOINT_ACTION_PARAMS.keys())}."
            )

        expected_types = WAYPOINT_ACTION_PARAMS[action]

        if len(params) != len(expected_types):
            raise ValueError(
                f"Action '{action}' for Waypoint ID {waypoint.id}"
                f"expects {len(expected_types)} parameter(s), but got {len(params)}."
            )

        for i, (param, expected_type) in enumerate(zip(params, expected_types)):
            if not isinstance(param, expected_type):
                raise ValueError(
                    f"Parameter {i+1} for action '{action}' on Waypoint ID {waypoint.id}"
                    f"should be of type {expected_type.__name__}, but got {type(param).__name__}."
                )

    def load_from_txt(self, filename: str) -> None:
        with open(filename, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip lines that are comments or empty
            if not line or line.startswith("#"):
                continue

            # Remove comments that start with '#'
            line = line.split("#")[0].strip()

            parts = line.strip().split(",")
            if len(parts) < 4:
                raise ValueError(f"Invalid waypoint format: {line}, in line {i+1}")

            id = int(parts[0].strip())
            pn = float(parts[1].strip())
            pe = float(parts[2].strip())
            h = float(parts[3].strip())

            action = parts[4].strip() if len(parts) > 4 else "NONE"
            params = []

            if action in WAYPOINT_ACTION_PARAMS:
                expected_types = WAYPOINT_ACTION_PARAMS[action]
                for i, param_type in enumerate(expected_types):
                    param_type = expected_types[i]
                    param_str = str(parts[5 + i].strip())
                    params.append(param_type(param_str))
            else:
                raise ValueError(f"Invalid waypoint action: {action}, in line {i+1}")

            waypoint = Waypoint(id, pn, pe, h, action, *params)
            self.add_waypoint(waypoint)
