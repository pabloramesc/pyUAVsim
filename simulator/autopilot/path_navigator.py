from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.path_follower import BasePathParams
from simulator.autopilot.line_follower import LinePathParams
from simulator.autopilot.orbit_follower import OrbitPathParams
from simulator.autopilot.route_manager import RouteManager


@dataclass
class PathCommand:
    """
    Data structure for holding path navigation output commands.

    Attributes
    ----------
    path_type : str
        The type of path to follow ('line' or 'orbit').
    path_params : BasePathParams
        Parameters for the line or orbit path follower.
    is_new_path : bool
        Flag indicating whether the path is new or an update to an existing path.
    """
    path_type: str = None  # 'line' or 'orbit'
    path_params: BasePathParams = None
    is_new_path: bool = True


class PathNavigator(ABC):
    """
    Abstract base class for path navigators that provide guidance for different path following strategies.
    """

    def __init__(
        self, config: AutopilotConfig, route_manager: RouteManager = None
    ) -> None:
        """
        Initialize the path navigator with configuration and waypoints manager.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration parameters for the autopilot.
        route_manager : RouteManager, optional
            Manager for handling waypoint navigation, by default None
        """
        self.config = config
        self.route_manager = route_manager or RouteManager(config)
        self.nav_cmd = PathCommand()
        self.nav_cmd.is_new_path = True

    @abstractmethod
    def navigate_path(self, pos_ned: np.ndarray, course: float) -> PathCommand:
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
        PathCommand
        """
        pass


class LinePathNavigator(PathNavigator):
    """
    Path navigator for line paths between waypoints.
    """

    def navigate_path(self, pos_ned: np.ndarray, course: float = None) -> PathCommand:
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
        course : float <-- NOT USED!
            The current heading/course angle of the vehicle.

        Returns
        -------
        PathCommand
        """
        wp0, wp1, wp2 = self.route_manager.get_path_waypoints()
        q01 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))  # wp0 to wp1 direction

        if not self.route_manager.is_target_last():
            q12 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))  # wp1 to wp2 direction
            normal_dir = (q01 + q12) / np.linalg.norm((q01 + q12))
        else:
            q12 = q01  # if target wp is the last, follow wp0 to wp1
            normal_dir = q01

        # if the transition frontier is reached, set line path from wp1 to wp2
        if normal_dir.dot(pos_ned - wp1) > 0:
            self.nav_cmd.path_type = "line"
            self.nav_cmd.path_params = LinePathParams(wp1, q12)  # go from wp1 to wp2
            self.nav_cmd.is_new_path = True
            self.route_manager.advance(pos_ned)  # increment target wp

        # if not, keep following line path from wp0 to wp1
        else:
            self.nav_cmd.path_type = "line"
            self.nav_cmd.path_params = LinePathParams(wp0, q01)  # go from wp0 to wp1

        return self.nav_cmd


class FilletPathNavigator(PathNavigator):

    def __init__(self, config: AutopilotConfig, route_manager: RouteManager) -> None:
        super().__init__(config, route_manager)
        self.on_fillet = False

    def navigate_path(self, pos_ned: np.ndarray, course: float = None) -> PathCommand:
        """
        Navigate and provide guidance for fillet path following.

        This method is a placeholder for managing and providing guidance for fillet
        paths. The fillet paths involve transitions between straight line segments with
        curved segments.

        Parameters
        ----------
        pos_ned : np.ndarray
            The current position of the vehicle in NED (North-East-Down) coordinates.
        course : float <-- NOT USED!
            The current heading/course angle of the vehicle.

        Returns
        -------
        PathCommand
        """
        # TODO: test fillet paths

        wp0, wp1, wp2 = self.route_manager.get_path_waypoints()
        q01 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))  # wp0 to wp1 direction
        fillet_radius = self.config.min_turn_radius

        if not self.route_manager.is_target_last():
            q12 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))  # wp1 to wp2 direction
            fillet_angle = np.acos(-q01.T @ q12)
            fillet_center = wp1 - fillet_radius / np.sin(fillet_angle / 2) * (
                q01 - q12
            ) / np.linalg.norm(q01 - q12)
        else:
            q12 = q01  # if target wp is the last, follow wp0 to wp1

        # if following line path from wp0 to wp1, before the fillet
        if not self.on_fillet:
            transition_dist = wp1 - fillet_radius / np.tan(fillet_angle / 2)

            # if the transition frontier from line path to the fillet is reached, set orbit path
            if q01.dot(pos_ned - transition_dist) > 0.0:
                self.on_fillet = True
                self.nav_cmd.path_type = "orbit"
                self.nav_cmd.path_params = OrbitPathParams(fillet_center, fillet_radius)
                self.nav_cmd.is_new_path = True

            # if not, keep following line path from wp0 to wp1
            else:
                self.on_fillet = False
                self.nav_cmd.path_type = "line"
                self.nav_cmd.path_params = LinePathParams(wp0, q01)

        # if following orbit path (fillet) around wp1
        else:
            transition_dist = wp1 + fillet_radius / np.tan(fillet_angle / 2)

            # if the transition frontier from fillet to next line path is reached
            if q12.dot(pos_ned - transition_dist) > 0.0:
                self.on_fillet = False
                self.nav_cmd.path_type = "line"
                self.nav_cmd.path_params = LinePathParams(wp1, q12)
                self.nav_cmd.is_new_path = True
                self.route_manager.advance()  # increment target wp

            # if not, keep following orbit path (fillet)
            else:
                self.on_fillet = True
                self.nav_cmd.path_type = "orbit"
                self.nav_cmd.path_params = OrbitPathParams(fillet_center, fillet_radius)

        return self.nav_cmd


class DubinPathNavigator(PathNavigator):

    def navigate_path(self, pos_ned: np.ndarray, course: float) -> PathCommand:
        # TODO: implement dubins path navigator
        raise NotImplementedError
