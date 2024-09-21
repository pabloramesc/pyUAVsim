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
        self.active_action = "none"
        self.action_status = "none"

    def reset(self) -> None:
        self.path_cmd = PathNavCommand()
        self.status = "wait"
        self.active_action = "none"
        self.action_status = "none"

    def update(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> PathNavCommand:

        if self.status not in ACTIONS_MANAGER_STATUS:
            raise ValueError(f"Not valid waypoint actions manager status: {self.status}!")

        if self.status == "done":
            self.status = "wait"
            self.active_action = "none"
            self.action_status = "none"

        if self.status == "wait":
            if wp is not None and self.route_manager.is_on_waypoint(pos_ned, wp.id):
                wp.action.restart()
                if self._is_orbit_action(wp.action_code):
                    self._set_orbit_path(wp)
                self.active_action = wp.action_code.lower()
                self.status == "run"

        if self.status == "run":
            self._execute_action(wp, pos_ned, dt)

        if wp.action.is_done():
            self.status == "done"
            self.route_manager.advance()

        return self.path_cmd

    def _is_orbit_action(self, action_code: str) -> bool:
        orbit_action_codes = ["ORBIT_UNLIM", "ORBIT_TIME", "ORBIT_TURNS", "ORBIT_ALT"]
        return action_code in orbit_action_codes

    def _set_orbit_path(self, wp: Waypoint) -> None:
        action: OrbitUnlimited = wp.action
        self.path_cmd.path_type = "orbit"
        self.path_cmd.path_params = OrbitPathParams(
            wp.ned_coords, action.radius, action.direction
        )
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

    def _orbit_info(radius: float, dir: int) -> str:
        dir_str = "CCW" if dir > 0 else "CW"
        return f"Radius: {radius:.1f} m, Direction: {dir_str}"

    def _execute_orbit_unlimited(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitUnlimited = wp.action
        action.execute()
        self.action_status = self._orbit_info(action.radius, action.direction)

    def _execute_orbit_time(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitTime = wp.action
        action.execute(dt)
        self.action_status = self._orbit_info(action.radius, action.direction)
        self.action_status += (
            f"Orbit time: {action.time:.2f} s, Elapsed Time: {action._elapsed_time:.2f} s"
            f"Progress: {action._elapsed_time/action.time*100.0:.2f}%"
        )

    def _execute_orbit_turns(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: OrbitTurns = wp.action
        r = pos_ned - wp.ned_coords
        ang_pos = np.arctan2(r[1], r[0])
        action.execute(ang_pos)
        self.action_status = self._orbit_info(action.radius, action.direction)
        self.action_status += (
            f"Orbit turns: {action.turns}, Completed turns: {action._completed_turns:.1f}, "
            f"Progress: {action._completed_turns/action.turns*100.0:.2f}%"
        )

    def _execute_orbit_alt(self, wp: Waypoint, pos_ned: np.ndarray, dt: float) -> None:
        action: OrbitAlt = wp.action
        alt = -pos_ned[2]
        action.execute(alt)
        self.action_status = self._orbit_info(action.radius, action.direction)
        self.action_status += (
            f"Orbit altitude: {action.altitude:.1f} m, Current altitude: {alt:.1f} m, "
            f"Progress: {alt/action.altitude*100.0:.2f}%"
        )

    def _execute_go_waypoint(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: GoWaypoint = wp.action
        if action.has_pending_jumps():
            self.route_manager.set_target_waypoint(action.waypoint_id)
        action.execute()
        self.action_status = (
            f"Repeat: {action.repeat}, Repeat Count: {action._repeat_count}"
        )

    def _execute_set_airspeed(
        self, wp: Waypoint, pos_ned: np.ndarray, dt: float
    ) -> None:
        action: SetAirspeed = wp.action
        self.flight_cmd.target_airspeed = action.airspeed
        action.execute()
        self.action_status = f"Target airspeed: {action.airspeed:.2f} m/s"
