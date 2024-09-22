"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import asdict

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.base_follower import BasePathFollower, BasePathParams
from simulator.autopilot.line_follower import LinePathFollower, LinePathParams
from simulator.autopilot.orbit_follower import OrbitPathFollower, OrbitPathParams

PATH_FOLLOWER_TYPES = ["none", "line", "orbit"]
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
        self.active_follower_info: str = None
        self.active_follower_status: str = None

    def reset(self) -> None:
        """
        Reset the path follower, clearing the active follower and setting status to 'wait'.
        """
        self.active_follower = None
        self.status = "wait"
        self.active_follower_type = "none"
        self.active_follower_info = None
        self.active_follower_status = None

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
            f"Direction: {"CW" if orbit_params.direction > 0 else "CWW"}"
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

        if self.active_follower_type == "none":
            raise Exception("Select a follower before update!")
        
        elif self.active_follower_type == "line":
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
