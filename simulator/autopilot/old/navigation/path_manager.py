"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from enum import Enum
import numpy as np
from simulator.autopilot.autopilot_configuration import AutopilotConfiguration

from simulator.autopilot.guidance.line_follower import LineFollower
from simulator.autopilot.guidance.orbit_follower import OrbitFollower


class PathManagerModes(Enum):
    INIT = 0  # initialized mode: check if an aux waypoint (at current vehicle position) needs to be placed
    RUN = 1  # running mode: follow the paths between waypoints
    END = 2  # end mode: orbit around last waypoint


class PathManager:
    def __init__(self, params: AutopilotConfiguration) -> None:
        self.params = params
        self.line_follower = LineFollower(params)
        self.orbit_follower = OrbitFollower(params)

        self.waypoints_coords: np.ndarray = None
        self.target_waypoint: int = 0  # index of waypoints (0 correspond to aux waypoint at vehicle position)
        self.current_mode: PathManagerModes.INIT

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        if not isinstance(waypoints, np.ndarray) or waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints array must be a N-by-3 size numpy.ndarray (where N is the number of WPs)!")
        if waypoints.shape[0] < 3:
            raise ValueError("there must be provided 3 waypoints at least!")
        self.waypoints_coords = np.zeros((waypoints.shape[0] + 1, 3))
        self.waypoints_coords[1:] = waypoints
        self.target_waypoint = 1
        self.current_mode = PathManagerModes.INIT

    def set_target_waypoint(self, waypoint_id: int) -> None:
        if waypoint_id < 1 or waypoint_id > self.waypoints_coords.shape[0]:
            raise ValueError(f"waypoint_id must be between 1 and {self.waypoints_coords.shape[0]}")
        self.target_waypoint = waypoint_id

    def restart_mission(self) -> None:
        self.target_waypoint = 1
        self.waypoints_coords[0] = np.zeros(3)  # reset the aux initial waypoint
        self.current_mode = PathManagerModes.INIT

    def manager_interface(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> tuple:
        pass  # interface method

    def update(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> tuple:
        if self.current_mode == PathManagerModes.INIT:
            # if the WP1 is too close to vehicle current position set target as WP2 (WP1 to WP2 path)
            if self.is_on_waypoint(1, aircraft_pos_ned):
                self.target_waypoint = 2
                self.current_mode = PathManagerModes.RUN
            # if the WP1 is far from the vehicle current position set target as WP1 (WP0 to WP1 path)
            else:
                self.target_waypoint = 1
                self.waypoints_coords[0] = aircraft_pos_ned
                self.current_mode = PathManagerModes.RUN

        if self.current_mode == PathManagerModes.RUN:
            # if the last waypoint is reached change to END mode
            if self.target_waypoint >= self.waypoints_coords.shape[0]:
                self.current_mode = PathManagerModes.END
            else:
                return self.manager_interface(aircraft_pos_ned, aircraft_course)

        if self.current_mode == PathManagerModes.END:
            # if manager is in END mode do orbits around last waypoint
            self.orbit_follower.set_orbit_path(self.waypoints_coords[-1, :], self.params.wait_orbit_radius, 1)
            return self.orbit_follower.update(aircraft_pos_ned, aircraft_course)

    def get_target_waypoint(self) -> int:
        return self.target_waypoint

    def get_status_string(self, aircraft_pos_ned: np.ndarray) -> str:
        if self.current_mode == PathManagerModes.END or self.target_waypoint >= self.waypoints_coords.shape[0]:
            return f"END R{self.params.wait_orbit_radius}"
        return f"WP{self.target_waypoint} ({self.get_3D_distance_to_waypoint(aircraft_pos_ned):.0f}m)"

    def get_3D_distance_to_waypoint(self, aircraft_pos_ned: np.ndarray) -> float:
        return np.linalg.norm(self.waypoints_coords[self.target_waypoint, :] - aircraft_pos_ned)

    def get_2D_distance_to_waypoint(self, aircraft_pos_ned: np.ndarray) -> float:
        return np.linalg.norm(self.waypoints_coords[self.target_waypoint, 0:2] - aircraft_pos_ned[0:2])
    
    def get_waypoint_on(self, aircraft_pos_ned: np.ndarray) -> float:
        if self.get_2D_distance_to_waypoint(aircraft_pos_ned) < self.params.waypoint_default_radius:
            return self.target_waypoint
        return -1
    
    def is_on_waypoint(self, waypoint_id: int, aircraft_pos_ned: np.ndarray) -> bool:
        dist = np.linalg.norm(self.waypoints_coords[waypoint_id, 0:2] - aircraft_pos_ned[0:2])
        if dist < self.params.waypoint_default_radius:
            return True
        return False
