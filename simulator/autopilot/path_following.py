"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft.aircraft_state import AircraftState
from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.flight_control import FlightControl
from simulator.math.rotation import rot_matrix_zyx


class LineFollower:
    """Class for line following guidance."""

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the LineFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        self.config = config
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
        e_py = self.R_ip.dot(pos_ned - self.path_origin)[1]  # path error 2nd component
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

    def guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> tuple[float, float]:
        """
        Provide both lateral and longitudinal guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for line following
            as `(course_ref, altitude_ref)`.
        """
        course_ref = self.lateral_guidance(pos_ned, course)
        altitude_ref = self.longitudinal_guidance(pos_ned)
        return course_ref, altitude_ref

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


class OrbitFollower:
    """Class for orbit following guidance."""

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the OrbitFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        self.config = config
        self.k_orbit = 1.0
        self.orbit_center = np.zeros(3)
        self.orbit_radius = 0.0
        self.orbit_direction = 0  # 1 (clockwise) or -1 (counter-clockwise)

    def set_orbit(self, center: np.ndarray, radius: float, direction: int) -> None:
        """
        Set the orbit with a given center, radius, and direction.

        Parameters
        ----------
        center : np.ndarray
            A 3-element array representing the center of the orbit.
        radius : float
            The radius of the orbit.
        direction : int
            The direction of the orbit: 1 for clockwise, -1 for counter-clockwise.

        Raises
        ------
        ValueError
            If `center` does not have the shape (3,) or if `radius` is not greater than zero.
        """
        if center.shape != (3,):
            raise ValueError("center parameter must be a np.ndarray with shape (3,)")
        else:
            self.orbit_center = center

        if radius <= 0.0:
            raise ValueError("radius parameter must be a float greater than zero")
        else:
            self.orbit_radius = radius

        self.orbit_direction = np.sign(direction)
        self.k_orbit = self.orbit_radius / self.config.min_turn_radius

    def lateral_guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> float:
        """
        Calculate the lateral guidance for orbit following.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        float
            The reference course angle for orbit following.
        """
        pn, pe, _ = pos_ned
        cn, ce, _ = self.orbit_center
        dist_error = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        ang_position = np.arctan2(pe - ce, pn - cn)
        ang_position = FlightControl.wrap_course(ang_position, course)
        course_ref = ang_position + self.orbit_direction * (
            0.5 * np.pi
            + np.arctan(
                self.k_orbit * (dist_error - self.orbit_radius) / self.orbit_radius
            )
        )
        return course_ref

    def get_angular_position(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the angular position of the aircraft relative to the orbit center.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The angular position in radians.
        """
        pn, pe, _ = pos_ned
        cn, ce, _ = self.orbit_center
        ang_position = np.arctan2(pe - ce, pn - cn)
        return ang_position

    def plot_course_field(self, density: int = 20) -> None:
        """
        Plot the course field for orbit following.

        Parameters
        ----------
        density : int, optional
            The density of the grid for plotting (default is 20).
        """
        # Extract orbit center and radius
        cn, ce, _ = self.orbit_center
        r = self.orbit_radius

        # Create position grids for pn and pe
        pn, pe = np.meshgrid(
            np.linspace(cn - 2 * r, cn + 2 * r, density),
            np.linspace(ce - 2 * r, ce + 2 * r, density),
        )
        d = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        psi = np.arctan2(pe - ce, pn - cn)

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
        ax.set_title("Orbit Following Course Field")
        ax.set_ylabel("North position (m)")
        ax.set_xlabel("East position (m)")
        ax.set_aspect("equal")
        ax.plot(ce, cn, "ro")  # Plot initial position
        ang = np.linspace(-np.pi, +np.pi, 100)
        ax.plot(r * np.cos(ang) + ce, r * np.sin(ang) + cn, "r--")  # Orbit path
        ax.quiver(pe, pn, qe, qn)  # Plot vector arrows
        plt.show()
