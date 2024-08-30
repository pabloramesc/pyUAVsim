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
from simulator.autopilot.navigation_v2.waypoint import Waypoint, WaypointCommand
from simulator.utils.rotation import diff_angle_pi


class WaypointsManagerState(Enum):
    INIT = 0  # waypoints manager loaded or reset, set path to initial waypoint
    REACH = 1  # target waypoint is reached, waypoint must be analyzed 
    LINE = 2  # currently following a line path to target waypoint
    ORBIT = 3  # currently orbiting the target waypoint until end condition is triggered
    END = 4  # last waypoint is reached
    ERROR = 5  # an error ocurred and mission cannot be completed
    WAIT = 6  # orbiting a point until path manager is reset


class WaypointsManager:
    def __init__(self, config: AutopilotConfiguration) -> None:
        self.config = config

        self.orbit_follower = OrbitFollower(self.config)
        self.line_follower = LineFollower(self.config)

        self.target_airspeed: float = 0.0
        self.target_heading: float = 0.0
        self.target_altitude: float = 0.0

        self.waypoints: list = []
        self.waypoints_num = 0

        self.home_geo = np.zeros(3)  # geographic reference coordinates to waypoint coords conversion to NED frame

        self.mission_name: str

        self.state: WaypointsManagerState = WaypointsManagerState.INIT
        self.target_waypoint_id: int = 0
        self.target_waypoint: Waypoint = None

        self.wp0_pos_ned: np.ndarray = None
        self.wp1_pos_ned: np.ndarray = None
        self.wp0_wp1_dir: np.ndarray = None

        self.timer = 0.0  # timer for timing actions
        self.turns_counter = 0.0  # angular counter for turns (float)
        self.reached_waypoints_ids = []  # id list of each waypoint reached

        self.prev_angular_position = 0.0

    def load_mission_file(self, file_name: str) -> None:
        with open(file_name, "r") as file:
            file_lines = file.readlines()

            for kk, line in enumerate(file_lines[1:]):
                line = line.split(" ")

                wp_id = int(line[0])
                if wp_id != kk + 1:
                    raise ValueError("WP index is invalid! Must start with 1 and ascend.")
                waypoint = Waypoint(
                    id=wp_id,
                    command=WaypointCommand(int(line[1])),
                    latitude=float(line[2]),
                    longitude=float(line[3]),
                    altitude=float(line[4]),
                    radius=float(line[5]),
                    param1=float(line[6]),
                    param2=float(line[7]),
                    param3=float(line[8]),
                )

                if waypoint.radius <= 0.0:
                    waypoint.radius = self.config.waypoint_default_radius

                # check ORBIT waypoints params
                if waypoint.command == WaypointCommand.ORBIT_TIME or waypoint.command == WaypointCommand.ORBIT_TURN:
                    if waypoint.param1 == 0.0:
                        raise ValueError(f"WP{waypoint.id} param1 (direction) can not be 0!")
                    waypoint.param1 = np.sign(waypoint.param1)  # direction
                    waypoint.param2 = float(waypoint.param1)  # timer or turns
                    waypoint.param3 = None

                # check JUMP waypoints params
                if waypoint.command == WaypointCommand.JUMP:
                    if 0 >= waypoint.param1 > len(file_lines[1:]):
                        raise ValueError(f"WP{waypoint.id} param1 (waypoint) does not point to any valid waypoint!")
                    waypoint.param1 = int(waypoint.param1)  # jump waypoint
                    waypoint.param2 = int(waypoint.param2)  # loop times
                    waypoint.param3 = None

                self.waypoints.append(waypoint)
            self.waypoints_num = len(self.waypoints)

    def set_home(self, geo_coords: np.ndarray) -> None:
        self.home_geo = geo_coords

    def get_3D_distance_to_waypoint(self, aircraft_pos_ned: np.ndarray, waypoint_pos_ned: np.ndarray) -> float:
        return np.linalg.norm(waypoint_pos_ned - aircraft_pos_ned)

    def get_2D_distance_to_waypoint(self, aircraft_pos_ned: np.ndarray, waypoint_pos_ned: np.ndarray) -> float:
        return np.linalg.norm(waypoint_pos_ned[0:2] - aircraft_pos_ned[0:2])

    def _build_orbit(self, waypoint: Waypoint) -> None:
        center = waypoint.get_coords_NED(self.home_geo)
        radius = waypoint.param1
        direction = waypoint.param2  # +1 if current_waypoint.param2 >= 0 else -1
        self.orbit_follower.set_orbit_path(center, radius, direction)
        # store for later use
        self.wp0_pos_ned = center

    def _build_line(self) -> None:
        wp0 = self.wp0_pos_ned
        wp1 = self.wp1_pos_ned
        dir = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))
        self.line_follower.set_line_path(wp0, dir)
        # store calculated direction for later use
        self.wp0_wp1_dir = dir

    def _is_wp1_reached(self, aircraft_pos_ned: np.ndarray) -> bool:
        # if the aircraft is inside the radius of the waypoint
        if self.get_3D_distance_to_waypoint(aircraft_pos_ned, self.wp1_pos_ned) < self.target_waypoint.radius:
            return True
        # if the aircraft has past away the waypoint without enter in its radius
        if self.wp0_wp1_dir.dot(aircraft_pos_ned - self.wp1_pos_ned) > 0:
            return True
        return False

    def reset(self) -> None:
        self.state = WaypointsManagerState.INIT
        self.target_waypoint_id = 0
        self.target_waypoint = self.waypoints[0]

    def update(self, time: float, position: np.ndarray, course: float) -> tuple:

        # if in initialization state set line path from current position to first waypoint
        if self.state == WaypointsManagerState.INIT:
            waypoint_0: Waypoint = self.waypoints[0]
            wp0_pos_ned = waypoint_0.get_coords_NED(self.home_geo)
            self.wp0_pos_ned = position
            self.wp1_pos_ned = wp0_pos_ned
            self._build_line()
            self.state = WaypointsManagerState.LINE
            self.target_waypoint_id = 0
            self.target_waypoint = waypoint_0

        # if the target waypoint is reached analyze the target waypoint to do corresponding actions
        if self.state == WaypointsManagerState.REACH:
            # if the target waypoint command is WAYPOINT set a line path to next waypoint
            if self.target_waypoint.command == WaypointCommand.WAYPOINT:
                self.target_waypoint_id += 1  # increase target waypoint index
                waypoint0: Waypoint = self.target_waypoint 
                waypoint1: Waypoint = self.waypoints[self.target_waypoint_id]
                self.target_waypoint = waypoint1  # update new target waypoint
                self.wp0_pos_ned = waypoint0.get_coords_NED(self.home_geo)
                self.wp1_pos_ned = waypoint1.get_coords_NED(self.home_geo)
                self._build_line()
            # if the target waypoint command is ORBIT set an orbit path centered on target waypoint
            if self.target_waypoint.command == WaypointCommand.ORBIT_TIME \
                or self.target_waypoint.command == WaypointCommand.ORBIT_TURNS:
                pass
 
        # if in line follow state check if the target waypoint is reached
        if self.state == WaypointsManagerState.LINE:
            if self._is_wp1_reached(position):
                self.state = WaypointsManagerState.REACH
            return self.line_follower.update(position, course)
        
        # if in orbit follow state check if end condition is triggered
        if self.state == WaypointsManagerState.ORBIT:
            return self.orbit_follower.update(position, course)