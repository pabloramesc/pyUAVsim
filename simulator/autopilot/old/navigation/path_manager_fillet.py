"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.autopilot.autopilot_configuration import AutopilotConfiguration
from simulator.autopilot.navigation.path_manager import PathManager


class PathManagerFillet(PathManager):
    def __init__(self, params: AutopilotConfiguration, end_action="orbit", loop_wp=0) -> None:
        super().__init__(params, end_action, loop_wp)
        
    def manager_interface(self, aircraft_pos_ned: np.ndarray, aircraft_course: float) -> tuple:
        wp0 = self.waypoints_coords[self.target_waypoint - 1, :]
        wp1 = self.waypoints_coords[self.target_waypoint, :]
        dir_wp0_to_wp1 = (wp1 - wp0) / np.linalg.norm((wp1 - wp0))

        # if the target waypoint is not the last
        if self.target_waypoint < self.waypoints_coords.shape[0] - 1:
            wp2 = self.waypoints_coords[self.target_waypoint + 1, :]
            dir_wp1_to_wp2 = (wp2 - wp1) / np.linalg.norm((wp2 - wp1))
            normal_dir = (dir_wp0_to_wp1 + dir_wp1_to_wp2) / np.linalg.norm((dir_wp0_to_wp1 + dir_wp1_to_wp2))

        # if the target waypoint is the last
        else:
            normal_dir = dir_wp0_to_wp1

        # if the target waypoint is reached
        if normal_dir.dot(aircraft_pos_ned - wp1) > 0:
            self.target_waypoint += 1

        line_orig = wp0
        line_dir = dir_wp0_to_wp1
        self.line_follower.set_line_path(line_orig, line_dir)
        return self.line_follower.update(aircraft_pos_ned, aircraft_course)