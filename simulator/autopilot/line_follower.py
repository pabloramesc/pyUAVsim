"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.base_follower import BasePathFollower, BasePathParams
from simulator.math.angles import wrap_angle
from simulator.math.rotation import rot_matrix_zyx


@dataclass
class LinePathParams(BasePathParams):
    """Parameters for line path following."""

    origin: np.ndarray
    direction: np.ndarray


class LinePathFollower(BasePathFollower):
    """Class for straight line path following."""

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the LineFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        super().__init__(config)
        self.k_path = 1.0 / self.config.min_turn_radius
        self.path_origin: np.ndarray = None
        self.path_direction: np.ndarray = None
        self.path_slope: float = None
        self.path_course: float = None
        self.R_ip = np.eye(3)  # rotation matrix from NED (inertia frame) to path frame

    def set_path(self, origin: np.ndarray, direction: np.ndarray) -> None:
        """
        Set the line path with a given origin and direction.

        Parameters
        ----------
        origin : np.ndarray
            A 3-element array representing the origin of the path.
        direction : np.ndarray
            A 3-element array representing the direction of the path.

        Raises
        ------
        ValueError
            If `origin` or `direction` does not have the shape (3,).
        """
        if np.all(origin == self.path_origin) and np.all(
            direction == self.path_direction
        ):
            return  # no calculation needed

        if origin.shape != (3,):
            raise ValueError("origin parameter must be a np.ndarray with shape (3,)")
        else:
            self.path_origin = origin

        if direction.shape != (3,):
            raise ValueError("direction parameter must be a np.ndarray with shape (3,)")
        elif np.linalg.norm(direction) <= 1e-6:
            raise ValueError("direction cannot be a null array")
        else:
            self.path_direction = direction / np.linalg.norm(direction)

        qn, qe, qd = self.path_direction
        slope = np.arctan(-qd / np.sqrt(qn**2 + qe**2))
        self.path_slope = np.clip(
            slope, self.config.min_path_slope, self.config.max_path_slope
        )
        self.path_course = np.arctan2(qe, qn)
        self.R_ip = rot_matrix_zyx(np.array([0.0, 0.0, self.path_course]))

    def lateral_guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> float:
        """
        Calculate the lateral guidance for line following.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        float
            The reference course angle for line following.
        """
        e_py = self.get_lateral_distance(pos_ned)  # path error 2nd component
        path_course_wrapped = wrap_angle(self.path_course, course)
        course_ref = (
            path_course_wrapped
            - self.config.course_inf * 2.0 / np.pi * np.arctan(self.k_path * e_py)
        )
        return course_ref

    def longitudinal_guidance(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the longitudinal guidance for line following.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The reference altitude for line following.
        """
        # error from aircraft to path origin (in NED inertial frame)
        e_ip = pos_ned - self.path_origin
        # normal vector to vertical path plane
        q_cross_ki = np.cross(np.array([0.0, 0.0, 1.0]), self.path_direction)
        n_qk = q_cross_ki / np.linalg.norm(q_cross_ki)
        # projection of error on the vertical path plane
        sn, se, sd = e_ip - e_ip.dot(n_qk) * n_qk
        rd = self.path_origin[2]
        qn, qe, qd = self.path_direction
        h_ref = -rd - np.sqrt(sn**2 + se**2) * (qd / np.sqrt(qn**2 + qe**2))
        return h_ref

    def get_lateral_distance(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the lateral distance of the aircraft to the linear path.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The lateral distance in meters.
        """
        lat_dist = self.R_ip.dot(pos_ned - self.path_origin)[1]
        return lat_dist
