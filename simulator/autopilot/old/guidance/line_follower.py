"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.autopilot.autopilot_configuration import AutopilotConfiguration
from simulator.common.constants import DEG2RAD, PI, RAD2DEG
from simulator.utils.rotation import rot_matrix_zyx


class LineFollower:
    def __init__(self, params: AutopilotConfiguration) -> None:

        self.course_inf = params.course_inf
        self.k_path = 1.0 / params.min_turn_radius
        self.max_path_slope = params.max_path_slope
        self.min_path_slope = params.min_path_slope

        self.path_origin = np.zeros(3)
        self.path_direction = np.zeros(3)
        self.path_course = 0.0
        self.R_ip = np.eye(3)  # rotation matrix from inertial frame to path frame

    def set_line_path(self, path_origin: np.ndarray, path_direction: np.ndarray) -> None:
        qn, qe, qd = path_direction
        path_slope = np.arctan(-qd / np.sqrt(qn**2 + qe**2))
        if path_slope > self.max_path_slope or path_slope < self.min_path_slope:
            raise ValueError(
                f"Impossible path direction! Path slope is {path_slope*RAD2DEG:.2f} deg but must be between"
                f" {self.min_path_slope*RAD2DEG:.2f} and {self.max_path_slope*RAD2DEG:.2f} deg."
            )
        self.path_origin = path_origin
        self.path_direction = path_direction
        self.path_course = np.arctan2(path_direction[1], path_direction[0])
        self.R_ip = rot_matrix_zyx(np.array([0.0, 0.0, self.path_course]))

    def wrap_course(self, path_course: float, aircraft_course: float):
        while path_course - aircraft_course > +PI:
            path_course = path_course - 2.0 * PI
        while path_course - aircraft_course < -PI:
            path_course = path_course + 2.0 * PI
        return path_course

    def lateral_path_following(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> float:
        e_py = self.R_ip.dot(aircraft_pos_ned - self.path_origin)[1]
        path_course_wrapped = self.wrap_course(self.path_course, aircraft_course)
        course_cmd = path_course_wrapped - self.course_inf * (2.0 / PI) * np.arctan(self.k_path * e_py)
        return course_cmd

    def longitudinal_path_following(self, aircraft_pos_ned: np.ndarray) -> float:
        # error from aircraft to path origin (in NED inertial frame)
        e_ip = aircraft_pos_ned - self.path_origin
        # normal vector to vertical path plane
        q_cross_ki = np.cross(np.array([0.0, 0.0, 1.0]), self.path_direction)
        n_qk = q_cross_ki / np.linalg.norm(q_cross_ki)
        # projection of error on the vertical path plane
        s_n, s_e, s_d = e_ip - e_ip.dot(n_qk) * n_qk
        r_d = self.path_origin[2]
        q_n, q_e, q_d = self.path_direction
        altitude_cmd = -r_d - np.sqrt(s_n**2 + s_e**2) * (q_d / np.sqrt(q_n**2 + q_e**2))
        return altitude_cmd
    
    def get_field(self, density: int = 10, distance: float = 100.0) -> tuple:
        pn, pe = np.meshgrid(np.linspace(-distance, distance, density), np.linspace(-distance, distance, density))
        ang = 0.0 - (PI / 2.0) * (2.0 / PI) * np.arctan(self.k_path * pe)
        qn = np.cos(ang)
        qe = np.sin(ang)
        return (pe, pn, qe, qn)

    def update(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> tuple:
        """_summary_

        Parameters
        ----------
        aircraft_pos_ned : np.ndarray
            _description_
        aircraft_course : float
            _description_

        Returns
        -------
        tuple
            a tuple with commanded (course, altitude)
        """
        course_cmd = self.lateral_path_following(aircraft_pos_ned, aircraft_course)
        altitude_cmd = self.longitudinal_path_following(aircraft_pos_ned)
        return (course_cmd, altitude_cmd)
