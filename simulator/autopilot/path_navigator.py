from abc import ABC, abstractmethod

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.line_follower import LineFollower
from simulator.autopilot.orbit_follower import OrbitFollower
from simulator.autopilot.waypoints_manager import WaypointsManager


class PathNavigator(ABC):
    """
    Abstract base class for path navigators that provide guidance for different path following strategies.
    """

    def __init__(self, config: AutopilotConfig, wps_manager: WaypointsManager) -> None:
        """
        Initialize the PathNavigator with configuration and waypoints manager.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        wps_manager : WaypointsManager
            Manager for handling waypoint navigation.
        """
        self.config = config
        self.wps_manager = wps_manager
        self.line_follower = LineFollower(config)
        self.orbit_follower = OrbitFollower(config)
        self.follower_needs_update = True

    @abstractmethod
    def navigate_path(self, pos_ned: np.ndarray, course: float) -> tuple[float, float]:
        """
        Navigate and provide guidance for a specific path navigating strategy.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for path following
            as `(course_ref, altitude_ref)`.
        """
        pass


class LinePathNavigator(PathNavigator):
    """
    Path navigator for line paths between waypoints.
    """

    def navigate_path(self, pos_ned: np.ndarray, course: float) -> tuple[float, float]:
        """
        Provide guidance for following a line path between waypoints.

        This method calculates the guidance values for the vehicle to follow a line path
        between waypoints. It updates the target waypoint if the vehicle has reached the
        current target waypoint. For the last waypoint, it uses the direction of the line
        segment to provide guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for line path following
            as `(course_ref, altitude_ref)`.
        """
        wp0, wp1, wp2 = self.wps_manager.get_path_waypoints()
        dir_wp0_wp1 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))

        if not self.wps_manager.is_target_last():
            dir_wp1_wp2 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))
            normal_dir = (dir_wp0_wp1 + dir_wp1_wp2) / np.linalg.norm(
                (dir_wp0_wp1 + dir_wp1_wp2)
            )
        else:
            normal_dir = dir_wp0_wp1

        # if the transition frontier is reached
        if normal_dir.dot(pos_ned - wp1) > 0:
            self.wps_manager.advance(pos_ned)
            self.follower_needs_update = True

        if self.follower_needs_update:
            self.line_follower.set_line(wp0, dir_wp0_wp1)
            self.follower_needs_update = False

        return self.line_follower.guidance(pos_ned, course)


class FilletPathNavigator(PathNavigator):

    def __init__(self, config: AutopilotConfig, wps_manager: WaypointsManager) -> None:
        super().__init__(config, wps_manager)
        self.on_fillet = False

    def navigate_path(self, pos_ned: np.ndarray, course: float) -> tuple[float, float]:
        """
        Navigate and provide guidance for fillet path following.

        This method is a placeholder for managing and providing guidance for fillet
        paths. The fillet paths involve transitions between straight line segments with
        curved segments.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float
            The current heading/course angle of the vehicle.

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for fillet path following
            as `(course_ref, altitude_ref)`.
        """
        # TODO: test fillet paths

        wp0, wp1, wp2 = self.wps_manager.get_path_waypoints()
        dir_wp0_wp1 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))
        fillet_radius = self.config.min_turn_radius

        if not self.wps_manager.is_target_last():
            dir_wp1_wp2 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))
            fillet_angle = np.acos(-dir_wp0_wp1.T @ dir_wp1_wp2)

        # Determine whether to switch to a fillet or line path
        if not self.on_fillet:
            transition_dist = wp1 - fillet_radius / np.tan(fillet_angle / 2)
            # if the transition frontier from path to the fillet is reached
            if dir_wp0_wp1.dot(pos_ned - transition_dist) > 0.0:
                self.on_fillet = True
                self.follower_needs_update = True
        else:
            transition_dist = wp1 + fillet_radius / np.tan(fillet_angle / 2)
            # if the transition frontier from fillet to next path is reached
            if dir_wp1_wp2.dot(pos_ned - transition_dist) > 0.0:
                self.on_fillet = False
                self.follower_needs_update = True

        # Update the appropriate follower based on current segment
        if self.on_fillet:
            if self.follower_needs_update:
                orbit_center = wp1 - fillet_radius / np.sin(fillet_angle / 2) * (
                    dir_wp0_wp1 - dir_wp1_wp2
                ) / np.linalg.norm(dir_wp0_wp1 - dir_wp1_wp2)
                self.orbit_follower.set_orbit(orbit_center, fillet_radius, 1)
            return self.orbit_follower.guidance(pos_ned, course)
        else:
            if self.follower_needs_update:
                self.line_follower.set_line(wp0, dir_wp0_wp1)
            return self.line_follower.guidance(pos_ned, course)


class DubinPathNavigator(PathNavigator):

    def navigate_path(self, pos_ned: np.ndarray, course: float) -> tuple[float, float]:
        raise NotImplementedError
