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


@dataclass
class OrbitPathParams(BasePathParams):
    """Parameters for orbit path following."""

    center: np.ndarray
    radius: float
    direction: int = 1


class OrbitPathFollower(BasePathFollower):
    """Class for circular orbit path following."""

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the OrbitFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        super().__init__(config)
        self.k_orbit = 1.0
        self.orbit_center: np.ndarray = None
        self.orbit_radius: float = None
        self.orbit_direction = 0  # 1 (clockwise) or -1 (counter-clockwise)
        self.orbit_altitude: float = None

    def set_path(self, center: np.ndarray, radius: float, direction: int = 1) -> None:
        """
        Set the orbit with a given center, radius, and direction.

        Parameters
        ----------
        center : np.ndarray
            A 3-element array representing the center of the orbit.
        radius : float
            The radius of the orbit.
        direction : int, optional
            The direction of the orbit: 1 for clockwise, -1 for counter-clockwise.
            By default 1 (clockwise).

        Raises
        ------
        ValueError
            If `center` does not have the shape (3,) or if `radius` is not greater than zero.
        """
        if (
            np.all(center == self.orbit_center)
            and radius == self.orbit_radius
            and direction == self.orbit_direction
        ):
            return  # no calculation needed

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
        self.orbit_altitude = -self.orbit_center[2]

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
        dist_error = self.get_lateral_distance(pos_ned)
        ang_position = self.get_angular_position(pos_ned)
        ang_position = wrap_angle(ang_position, course)
        course_ref = ang_position + self.orbit_direction * (
            0.5 * np.pi
            + np.arctan(
                self.k_orbit * (dist_error - self.orbit_radius) / self.orbit_radius
            )
        )
        return course_ref

    def longitudinal_guidance(self, pos_ned: np.ndarray) -> float:
        """
        Orbit has a constant altitude. No calculation is needed.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The orbit altitude.
        """
        return -self.orbit_center[2]  # orbit altitude

    def get_lateral_distance(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the lateral distance of the aircraft to the orbit center.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The lateral distance in meters.
        """
        pn, pe, _ = pos_ned
        cn, ce, _ = self.orbit_center
        lat_dist = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        return lat_dist

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
