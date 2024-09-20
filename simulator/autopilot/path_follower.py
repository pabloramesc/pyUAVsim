"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.math.rotation import rot_matrix_zyx
from simulator.math.angles import wrap_angle


class BasePathFollower(ABC):
    """Base class for path following guidance."""

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the PathFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        self.config = config

    @abstractmethod
    def set_path(self, *args, **kwargs):
        """
        Set the path parameters. To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def lateral_guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> float:
        """
        Calculate the lateral guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        float
            The reference course angle.
        """
        pass

    @abstractmethod
    def longitudinal_guidance(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the longitudinal guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The reference altitude.
        """
        pass

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
            The reference course angle and altitude for path following as `(course_ref, altitude_ref)`.
        """
        course_ref = self.lateral_guidance(pos_ned, course)
        altitude_ref = self.longitudinal_guidance(pos_ned)
        return course_ref, altitude_ref


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


@dataclass
class BasePathParams(ABC):
    """Base class for path parameters."""
    pass


@dataclass
class LinePathParams(BasePathParams):
    """Parameters for line path following."""
    origin: np.ndarray
    direction: np.ndarray


@dataclass
class OrbitPathParams(BasePathParams):
    """Parameters for orbit path following."""
    center: np.ndarray
    radius: float
    direction: int = 1

PATH_FOLLOWER_STATUS = ["wait", "init", "follow"]

class PathFollower:
    """
    Manages switching between line following and orbit following for path guidance.

    Attributes
    ----------
    line_follower : LinePathFollower
        Instance of the `LinePathFollower` to handle line path guidance.
    orbit_follower : OrbitPathFollower
        Instance of the `OrbitPathFollower` to handle orbit path guidance.
    active_follower : BasePathFollower
        The currently active path follower (either line or orbit follower).
    status : str
        Path follower status, one ('wait', 'init', 'follow').
    active_follower_type : str
        Type of the active path ('line', 'orbit', or 'none').
    active_follower_info : str
        Information about the current path being followed.
    active_follower_status : str
        Status info such as lateral error and angular position.
    """

    def __init__(self, config: AutopilotConfig) -> None:
        """
        Initialize the PathFollower with autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        """
        self.line_follower = LinePathFollower(config)
        self.orbit_follower = OrbitPathFollower(config)
        self.active_follower: BasePathFollower = None
        self.status = "wait"
        self.active_follower_type = "none"
        self.active_follower_info = "none"
        self.active_follower_status = "none"

    def reset(self) -> None:
        """
        Reset the path follower, clearing the active follower and setting status to 'wait'.
        """
        self.active_follower = None
        self.status = "wait"
        self.active_follower_type = "none"
        self.active_follower_info = "none"
        self.active_follower_status = "none"

    def follow_line(self, line_params: LinePathParams) -> None:
        """
        Start following a straight-line path.

        Parameters
        ----------
        line_params : LinePathParams
            Parameters defining the line path (origin and direction).
        """
        self.line_follower.set_path(**asdict(line_params))
        self.active_follower = self.line_follower
        self.status = "init"
        self.active_follower_type = "line"
        self.active_follower_info = (
            f"Slope: {np.rad2deg(self.line_follower.path_slope):.1f} deg, "
            f"Course: {np.rad2deg(self.line_follower.path_course):.1f} deg"
        )
        self.active_follower_status = "initialized"

    def follow_orbit(self, orbit_params: OrbitPathParams) -> None:
        """
        Start following a circular orbit path.

        Parameters
        ----------
        orbit_params : OrbitPathParams
            Parameters defining the orbit path, including center, radius, and direction.
        """
        self.orbit_follower.set_path(**asdict(orbit_params))
        self.active_follower = self.orbit_follower
        self.status = "init"
        self.active_follower_type = "orbit"
        self.active_follower_info = (
            f"Radius: {self.orbit_follower.orbit_radius:.1f} m, "
            f"Altitude: {self.orbit_follower.orbit_altitude:.1f} m, "
            f"Direction: {"ccw" if orbit_params.direction else "cw"}"
        )
        self.active_follower_status = "initialized"

    def update(self, pos_ned: np.ndarray, course: float = 0.0) -> tuple[float, float]:
        """
        Update the guidance for the current path based on the aircraft's position and course.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            The current course angle of the aircraft (default is 0.0).

        Returns
        -------
        tuple[float, float]
            The reference course and altitude for path following as `(course_ref, altitude_ref)`.

        Raises
        ------
        Exception
        
            If no active path follower is set when calling update.
        ValueError
            If the `status` is not a valid path follower status.
        """
        if self.status == "wait":
            raise Exception("Select a follower before update!")
        elif self.status in ["init", "follow"]:
            self.status = "follow"
        else:
            raise ValueError(f"Not valid path follower status: {self.status}!")

        course_ref, altitude_ref = self.active_follower.guidance(pos_ned, course)

        if self.active_follower_type == "line":
            lat_dev = self.line_follower.get_lateral_distance(pos_ned)
            self.active_follower_status = f"Lateral error: {lat_dev:.1f} m"
        
        elif self.active_follower_type == "orbit":
            lat_dev = self.orbit_follower.get_lateral_distance(pos_ned) - self.orbit_follower.orbit_radius
            ang_pos = self.orbit_follower.get_angular_position(pos_ned)
            self.active_follower_status = f"Lateral error: {lat_dev:.1f} m, "
            f"Angular position: {np.rad2deg(ang_pos):.1f} deg"

        else:
            raise ValueError(f"Not valid active follower type: {self.active_follower_type}!")

        return course_ref, altitude_ref
