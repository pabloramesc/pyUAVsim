import numpy as np

from simulator.autopilot.flight_control import FlightCommand
from simulator.autopilot.path_follower import OrbitPathParams
from simulator.autopilot.path_navigator import PathNavCommand
from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.waypoint_actions import (
    GoWaypoint,
    OrbitAlt,
    OrbitTime,
    OrbitTurns,
    OrbitUnlimited,
    SetAirspeed,
)
from simulator.autopilot.waypoints import Waypoint

ACTIONS_MANAGER_STATUS = ["wait", "run", "done", "fail"]


class WaypointActionsManager:

    def __init__(self, route_manager: RouteManager, flight_cmd: FlightCommand) -> None:
        self.route_manager = route_manager
        self.flight_cmd = flight_cmd
        self.path_cmd = PathNavCommand()
        self.status = "wait"
        self.active_action_code = "none"
        self.active_action_status: str = None

    def reset(self) -> None:
        self.path_cmd = PathNavCommand()
        self.status = "wait"
        self.active_action_code = "none"
        self.active_action_status = None

    def update(self, pos_ned: np.ndarray, dt: float) -> PathNavCommand:

        if self.status not in ACTIONS_MANAGER_STATUS:
            raise ValueError(f"Not valid waypoint actions manager status: {self.status}!")

        if self.status == "done":
            self.status = "wait"
            self.active_action_code = "none"
            self.active_action_status = None

        target_wp = self.route_manager.get_waypoint()

        if target_wp.action is not None:
            if self.status == "wait" and self.route_manager.is_on_waypoint(pos_ned):
                target_wp.action.restart()
                if self._is_orbit_action(target_wp.action_code):
                    self._set_orbit_path(target_wp)
                self.active_action_code = target_wp.action_code.lower()
                self.status = "run"

            if self.status == "run":
                self._execute_action(target_wp, pos_ned, dt)
                self.status = "run"

            if target_wp.action.is_done():
                if target_wp.action_code != "GO_WAYPOINT":
                    self.route_manager.advance(pos_ned)
                self.status = "done"

        return self.path_cmd

    def _is_orbit_action(self, action_code: str) -> bool:
        orbit_action_codes = ["ORBIT_UNLIM", "ORBIT_TIME", "ORBIT_TURNS", "ORBIT_ALT"]
        return action_code in orbit_action_codes

    def _set_orbit_path(self, wp: Waypoint) -> None:
        # use orbit unlim as base class for other orbit actions
        orbit_action: OrbitUnlimited = wp.action
        self.path_cmd.path_type = "orbit"
        self.path_cmd.path_params = OrbitPathParams(
            wp.ned_coords, orbit_action.radius, orbit_action.direction
        )

        # modify orbit center and waypoint coordinates altitude
        if wp.action_code == "ORBIT_ALT":
            orbit_alt_action: OrbitAlt = wp.action
            self.path_cmd.path_params.center[2] = -orbit_alt_action.altitude
            self.route_manager.set_waypoint_coords(self.path_cmd.path_params.center)

        self.path_cmd.is_new_path = True

    def _execute_action(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action_map = {
            "ORBIT_UNLIM": self._execute_orbit_unlimited,
            "ORBIT_TIME": self._execute_orbit_time,
            "ORBIT_TURNS": self._execute_orbit_turns,
            "ORBIT_ALT": self._execute_orbit_alt,
            "GO_WAYPOINT": self._execute_go_waypoint,
            "SET_AIRSPEED": self._execute_set_airspeed,
        }
        if wp.action_code in action_map:
            action_map[wp.action_code](wp, pos_ned, dt)
        else:
            raise ValueError(
                f"Invalid action code: {wp.action_code}, for waypoint id: {wp.id}!"
            )

    def _orbit_info(self, radius: float, dir: int) -> str:
        dir_str = "CW" if dir > 0 else "CCW"
        return f"Radius: {radius:.1f} m, Direction: {dir_str}"

    def _execute_orbit_unlimited(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitUnlimited = wp.action
        action.execute()
        self.active_action_status = self._orbit_info(action.radius, action.direction)

    def _execute_orbit_time(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitTime = wp.action
        action.execute(dt)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit time: {action.time:.2f} s, Elapsed Time: {action.elapsed_time:.2f} s, "
            f"Progress: {action.elapsed_time/action.time*100.0:.2f}%"
        )

    def _execute_orbit_turns(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitTurns = wp.action
        r = pos_ned - wp.ned_coords
        ang_pos = np.arctan2(r[1], r[0])
        action.execute(ang_pos)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit turns: {action.turns}, Completed turns: {action.completed_turns:.1f}, "
            f"Progress: {action.completed_turns/action.turns*100.0:.2f}%"
        )

    def _execute_orbit_alt(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitAlt = wp.action
        alt = -pos_ned[2]
        action.execute(alt)
        self.active_action_status = self._orbit_info(action.radius, action.direction)
        self.active_action_status += (
            f", Orbit altitude: {action.altitude:.1f} m, Current altitude: {alt:.1f} m, "
            f"Error: {alt - action.altitude:.2f} m"
        )

    def _execute_go_waypoint(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: GoWaypoint = wp.action
        if action.has_pending_jumps():
            self.route_manager.set_target_waypoint(action.wp_id)
        action.execute()
        self.active_action_status = (
            f"Repeat: {action.repeat}, Repeat Count: {action.repeat_count}"
        )

    def _execute_set_airspeed(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: SetAirspeed = wp.action
        self.flight_cmd.target_airspeed = action.airspeed
        action.execute()
        self.active_action_status = f"Target airspeed: {action.airspeed:.2f} m/s"
