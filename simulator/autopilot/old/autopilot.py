"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""


from enum import Enum

import numpy as np
from simulator.autopilot.autopilot_messages import (
    AutopilotCommands,
    AutopilotCommandsHistory,
    AutopilotDeltas,
    AutopilotDeltasHistory,
    TransmitterData,
)
from simulator.autopilot.autopilot_configuration import AutopilotConfiguration


from simulator.autopilot.control.lateral_control import LateralControl
from simulator.autopilot.control.longitudinal_control import LongitudinalControl
from simulator.autopilot.filters.filter import Filter, FilterEstimations
from simulator.autopilot.guidance.line_follower import LineFollower
from simulator.autopilot.guidance.orbit_follower import OrbitFollower
from simulator.autopilot.navigation.path_manager_line import PathManagerLine
from simulator.common.constants import DEFAULT_HOME_COORDS
from simulator.sensors.state_handler import StateHandler


class Modes(Enum):
    STANDBY = 0  # aicraft off
    MANUAL = 1  # full control to transmitter
    FBW = 2  # stabilized natural mode for planes (transmitter control changes on target attitude)
    STABILIZE = 3  # wing-leveling on stick release (transmitter control target attitude) <-- not recommended for planes
    AUTO = 4  # follow loaded waypoints plan
    ORBIT = 5
    TAKE_OFF = 6
    CLIMB = 7
    LAND = 8


TEST_WAYPOINTS_NED = np.array(
    [
        [0.0, 0.0, 0.0],  # WP0
        [+500.0, 0.0, -25.0],  # WP1
        [+1000.0, 0.0, -25.0],  # WP2
        [+1000.0, +500.0, -25.0],  # WP3
        [+0.0, +500.0, -25.0],  # WP4
    ]
)


class Autopilot:
    def __init__(self, config: AutopilotConfiguration, filter: Filter) -> None:
        self.config = config

        self.current_mode = Modes.STANDBY
        self.waypoints: list = None
        self.target_waypoint = 0

        self.target_airspeed = 0.0
        self.default_orbit_radius = 100.0

        self.state_filter = filter
        self.state_estimation: FilterEstimations = None
        self.home_coords = DEFAULT_HOME_COORDS

        self.commands = AutopilotCommands()
        self.commands_history = AutopilotCommandsHistory()

        self.deltas = AutopilotDeltas()
        self.deltas_history = AutopilotDeltasHistory()

        ### AIRCRAFT CONTROL (target vars to deltas) ###
        self.longitudinal_control = LongitudinalControl(self.config)
        self.lateral_control = LateralControl(self.config)

        ### AIRCRAFT GUIDANCE (target path to target vars) ###
        self.line_follower = LineFollower(self.config)
        self.orbit_follower = OrbitFollower(self.config)

        ### AICRAFT NAVIGATION (waypoints to target path) ###
        self.path_manager = PathManagerLine(self.config)

    def set_mode(self, mode: str) -> bool:
        self.current_mode = Modes[mode]
        if self.current_mode is Modes.ORBIT:
            self.orbit_follower.set_orbit_path(
                self.state_estimation.position_ned, self.default_orbit_radius, orbit_direction=1
            )
        if self.current_mode is Modes.AUTO:
            if self.waypoints is None:
                self.path_manager.set_waypoints(TEST_WAYPOINTS_NED)
            else:
                self.path_manager.set_waypoints(self.waypoints)
                
        return True
    
    def set_homde_coords(self, home_coords: np.ndarray):
        self.home_coords = home_coords

    def load_waypoints(self, waypoints: list) -> bool:
        self.waypoints = waypoints
        if self.waypoints is None:
            self.path_manager.set_waypoints(TEST_WAYPOINTS_NED)
        else:
            self.path_manager.set_waypoints(self.waypoints)
        return True

    def set_target_waypoint(self, waypoint_id: int) -> bool:
        self.target_waypoint = waypoint_id
        if waypoint_id > 0 and isinstance(waypoint_id, int):
            self.path_manager.target_waypoint = waypoint_id
        else:
            raise ValueError("waypoint id must be and integer greater than 0!")
        return True

    def set_home(self, home_coords: tuple) -> bool:
        self.home_coords = home_coords
        return True

    def set_target_airspeed(self, airspeed: float) -> bool:
        if airspeed < 0.0:
            raise ValueError("airspeed must be positive!")
        self.target_airspeed = airspeed
        return True

    def get_status_string(self) -> str:
        if self.current_mode == Modes.STANDBY:
            return "STANDBY"
        if self.current_mode == Modes.MANUAL:
            return "MANUAL"
        if self.current_mode == Modes.FBW:
            return "FLY-BY-WIRE"
        if self.current_mode == Modes.STABILIZE:
            return "STABILIZE"
        if self.current_mode == Modes.ORBIT:
            return f"ORBIT R{self.orbit_follower.orbit_radius}"
        if self.current_mode == Modes.AUTO:
            return f"AUTO {self.path_manager.get_status_string(self.state_estimation.position_ned)}"
            
    def update(self, tx_data: TransmitterData, sim_time: float, sim_dt: float) -> AutopilotDeltas:
        if sim_time == 0.0:
            press = self.state_filter.barometer.read(sim_time)[0]
            self.state_filter.set_baro_reference(press)
        self.state_estimation = self.state_filter.estimate(sim_time, sim_dt)
        
        tx_commands = tx_data.to_commands()

        if self.current_mode == Modes.STANDBY:
            self.deltas.elevator = 0.0
            self.deltas.aileron = 0.0
            self.deltas.rudder = 0.0
            self.deltas.throttle = 0.0

        if self.current_mode == Modes.MANUAL:
            self.deltas.elevator = tx_commands["elevator"]
            self.deltas.aileron = tx_commands["aileron"]
            self.deltas.rudder = tx_commands["rudder"]
            self.deltas.throttle = tx_commands["throttle"]

        if self.current_mode == Modes.FBW:
            ### COMMANDS
            self.commands.pitch += tx_commands["elevator"] * (self.config.fbw_max_pitch_rate * sim_dt)
            self.commands.roll += tx_commands["aileron"] * (self.config.fbw_max_roll_rate * sim_dt)
            ### DELTAS
            self.deltas.elevator = self.longitudinal_control.pitch_hold_with_elevator(
                self.commands.pitch, self.state_estimation.pitch, self.state_estimation.q
            )
            self.deltas.aileron = self.lateral_control.roll_hold_with_aileron(
                self.commands.roll, self.state_estimation.roll, self.state_estimation.p
            )
            self.deltas.rudder = tx_commands["rudder"]
            self.deltas.throttle = tx_commands["throttle"]

        if self.current_mode == Modes.STABILIZE:
            ### COMMANDS
            self.commands.pitch = tx_commands["elevator"]
            self.commands.roll = tx_commands["aileron"]
            ### DELTAS
            self.deltas.elevator = self.longitudinal_control.pitch_hold_with_elevator(
                self.commands.pitch, self.state_estimation.pitch, self.state_estimation.q
            )
            self.deltas.aileron = self.lateral_control.roll_hold_with_aileron(
                self.commands.roll, self.state_estimation.roll, self.state_estimation.p
            )
            self.deltas.rudder = tx_commands["rudder"]
            self.deltas.throttle = tx_commands["throttle"]

        if self.current_mode == Modes.ORBIT:
            ### COMMANDS
            self.commands.course, self.commands.altitude = self.orbit_follower.update(
                self.state_estimation.position_ned, self.state_estimation.course
            )
            self.commands.pitch = self.longitudinal_control.altitude_hold_with_pitch(
                self.commands.altitude, self.state_estimation.altitude, sim_dt
            )
            self.commands.roll = self.lateral_control.course_hold_with_roll(
                self.commands.course, self.state_estimation.course_angle, sim_dt
            )
            self.commands.airspeed = self.target_airspeed
            ### DELTAS
            self.deltas.elevator = self.longitudinal_control.pitch_hold_with_elevator(
                self.commands.pitch, self.state_estimation.pitch, self.state_estimation.q
            )
            self.deltas.aileron = self.lateral_control.roll_hold_with_aileron(
                self.commands.roll, self.state_estimation.roll, self.state_estimation.p
            )
            self.deltas.rudder = self.lateral_control.yaw_damper_with_rudder(self.state_estimation.r, sim_dt)
            self.deltas.throttle = self.longitudinal_control.airspeed_hold_with_throttle(
                self.commands.airspeed, self.state_estimation.airspeed, sim_dt
            )

        if self.current_mode == Modes.AUTO:
            ### COMMANDS
            self.commands.course, self.commands.altitude = self.path_manager.update(
                self.state_estimation.position_ned, self.state_estimation.course
            )
            self.commands.pitch = self.longitudinal_control.altitude_hold_with_pitch(
                self.commands.altitude, self.state_estimation.altitude, sim_dt
            )
            self.commands.roll = self.lateral_control.course_hold_with_roll(
                self.commands.course, self.state_estimation.course, sim_dt
            )
            self.commands.airspeed = self.target_airspeed
            ### DELTAS
            self.deltas.elevator = self.longitudinal_control.pitch_hold_with_elevator(
                self.commands.pitch, self.state_estimation.pitch, self.state_estimation.q
            )
            self.deltas.aileron = self.lateral_control.roll_hold_with_aileron(
                self.commands.roll, self.state_estimation.roll, self.state_estimation.p
            )
            self.deltas.rudder = self.lateral_control.yaw_damper_with_rudder(self.state_estimation.r, sim_dt)
            self.deltas.throttle = self.longitudinal_control.airspeed_hold_with_throttle(
                self.commands.airspeed, self.state_estimation.airspeed, sim_dt
            )

        self.commands_history.update(self.commands)
        self.deltas_history.update(self.deltas)

        return self.deltas
