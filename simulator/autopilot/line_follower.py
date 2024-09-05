"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.flight_control import FlightControl
from simulator.autopilot.path_follower import PathFollower
from simulator.math.rotation import rot_matrix_zyx


class LineFollower(PathFollower):
    """Class for line following guidance."""

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
        self.path_origin = np.zeros(3)
        self.path_direction = np.zeros(3)
        self.path_slope = 0.0
        self.path_course = 0.0
        self.R_ip = np.eye(3)  # rotation matrix from NED (inertia frame) to path frame

    def set_line(self, origin: np.ndarray, direction: np.ndarray) -> None:
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
        if origin == self.path_origin and direction == self.path_direction:
            return # no calculation needed

        if origin.shape != (3,):
            raise ValueError("origin parameter must be a np.ndarray with shape (3,)")
        else:
            self.path_origin = origin

        if direction.shape != (3,):
            raise ValueError("direction parameter must be a np.ndarray with shape (3,)")
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
        path_course_wrapped = FlightControl.wrap_course(self.path_course, course)
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

    def plot_course_field(self, density: int = 20, distance: float = 100.0) -> None:
        """
        Plot the course field for line following.

        Parameters
        ----------
        density : int, optional
            The density of the grid for plotting (default is 20).
        distance : float, optional
            The distance range from the path origin for plotting (default is 100.0).
        """
        # Extract initial position and direction
        pn0, pe0, _ = self.path_origin
        qn0, qe0, _ = self.path_direction

        # Create position grids for pn and pe
        pn = np.linspace(pn0 - distance, pn0 + distance, density)
        pe = np.linspace(pe0 - distance, pe0 + distance, density)
        pn, pe = np.meshgrid(pn, pe)

        # Compute the course for each grid point
        course = np.array(
            [
                self.lateral_guidance(np.array([pni, pei, 0.0]))
                for pni, pei in zip(pn.flatten(), pe.flatten())
            ]
        ).reshape(density, density)

        # Calculate the direction vectors
        qn, qe = np.cos(course), np.sin(course)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("Line Following Course Field")
        ax.set_ylabel("North position (m)")
        ax.set_xlabel("East position (m)")
        ax.set_aspect("equal")
        ax.plot(pe0, pn0, "ro")  # Plot initial position
        ax.plot(
            [pe0, pe0 + distance * qe0], [pn0, pn0 + distance * qn0], "r-"
        )  # Line path
        ax.plot(
            [pe0, pe0 - distance * qe0], [pn0, pn0 - distance * qn0], "r--"
        )  # Opposite line path
        ax.quiver(pe, pn, qe, qn)  # Plot vector arrows
        plt.show()

