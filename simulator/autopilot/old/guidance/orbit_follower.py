"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""


import numpy as np
from simulator.autopilot.autopilot_configuration import AutopilotConfiguration
from simulator.common.constants import PI


class OrbitFollower:
    def __init__(self, config: AutopilotConfiguration) -> None:
        self.k_orbit = 1.0
        self.min_turn_radius = config.min_turn_radius

        self.orbit_center = 0.0
        self.orbit_radius = 0.0
        self.orbit_direction = 0
        self.orbit_altitude = 0.0

    def wrap_course(self, path_course: float, aircraft_course: float):
        while path_course - aircraft_course > +PI:
            path_course = path_course - 2.0 * PI
        while path_course - aircraft_course < -PI:
            path_course = path_course + 2.0 * PI
        return path_course

    def set_orbit_path(self, orbit_center: np.ndarray, orbit_radius: float, orbit_direction: int) -> None:
        if not isinstance(orbit_center, np.ndarray) or orbit_center.shape != (3,):
            raise ValueError("orbit_center must be a numpy.ndarray with shape (3,)!")
        if orbit_radius <= 0:
            raise ValueError("orbit radius muest be greatter than 0!")
        if not orbit_direction in (-1, 1):
            raise ValueError("orbit direction must be 1 (clockwise) or -1 (counter-clockwise)!")
        self.orbit_center = orbit_center
        self.orbit_radius = orbit_radius
        self.orbit_direction = orbit_direction
        self.orbit_altitude = -orbit_center[2]
        self.k_orbit = self.orbit_radius / self.min_turn_radius

    def lateral_path_following(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> np.ndarray:
        pn, pe, _ = aircraft_pos_ned
        cn, ce, _ = self.orbit_center
        dist_error = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        ang_position = np.arctan2(pe - ce, pn - cn)
        ang_position = FlightControl.wrap_course(ang_position, aircraft_course)
        course_cmd = ang_position + self.orbit_direction * (
            0.5 * PI + np.arctan(self.k_orbit * (dist_error - self.orbit_radius) / self.orbit_radius)
        )
        return course_cmd
    
    def get_angular_position(self, aircraft_pos_ned: np.ndarray) -> float:
        pn, pe, _ = aircraft_pos_ned
        cn, ce, _ = self.orbit_center
        ang_position = np.arctan2(pe - ce, pn - cn)
        return ang_position

    def get_field(self, density: int = 10) -> tuple:
        cn = self.orbit_center[0]
        ce = self.orbit_center[1]
        r = self.orbit_radius
        pn, pe = np.meshgrid(np.linspace(cn - 2 * r, cn + 2 * r, density), np.linspace(ce - 2 * r, ce + 2 * r, density))
        d = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        psi = np.arctan2(pe - ce, pn - cn)
        # ang_position = self.wrap_course(ang_position, aircraft_course)
        ang = psi + self.orbit_direction * (0.5 * PI + np.arctan(self.k_orbit * (d - r) / r))
        qn = np.cos(ang)
        qe = np.sin(ang)
        return (pe, pn, qe, qn)

    def update(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> tuple:
        course_cmd = self.lateral_path_following(aircraft_pos_ned, aircraft_course)
        altitude_cmd = self.orbit_altitude
        return (course_cmd, altitude_cmd)
