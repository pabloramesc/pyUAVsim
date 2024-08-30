"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_configuration import AutopilotConfiguration
from simulator.autopilot.control.pid_control import PID_control
from simulator.sensors.state_handler import StateHandler


class LongitudinalControl:
    def __init__(self, config: AutopilotConfiguration, state: StateHandler = None) -> None:

        self.state = state

        self.time_prev = 0.0

        ##### PITCH CONTROL WITH ELEVATOR #####
        self.kp_pitch_elevator = config.kp_pitch_elevator
        self.kd_pitch_elevator = config.kd_pitch_elevator
        # self.max_elevator = params.max_elevator
        # self.min_elevator = params.min_elevator
        self.saturate_elevator = lambda x: np.clip(x, config.min_elevator, config.max_elevator)

        ##### ALTITUDE CONTROL WITH ELEVATOR #####
        # self.kp_altitude_pitch = params.kp_altitude_pitch
        # self.ki_altitude_pitch = params.ki_altitude_pitch
        self.PID_altitude_pitch = PID_control(kp=config.kp_altitude_pitch, ki=config.ki_altitude_pitch)
        self.saturate_pitch = lambda x: np.clip(x, config.min_pitch, config.max_pitch)

        ##### AIRSPEED CONTROL WITH THROTTLE #####
        # self.kp_airspeed_throttle = params.kp_airspeed_throttle
        # self.ki_airspeed_throttle = params.ki_airspeed_throttle
        self.PID_airspeed_throttle = PID_control(kp=config.kp_airspeed_throttle, ki=config.ki_airspeed_throttle)
        self.max_throttle = config.max_throttle
        self.min_throttle = config.min_throttle
        self.saturate_throttle = lambda x: np.clip(x, config.min_throttle, config.max_throttle)

        ##### AIRSPEED CONTROL WITH PITCH #####
        # self.kp_airspeed_pitch = params.kp_airspeed_pitch
        # self.ki_airspeed_pitch = params.ki_airspeed_pitch
        self.PID_airspeed_pitch = PID_control(kp=config.kp_airspeed_pitch, ki=config.ki_airspeed_pitch)

    def pitch_hold_with_elevator(
        self, commanded_pitch: float, aircraft_pitch: float, aircraft_pitch_rate: float
    ) -> float:
        delta_elevator = (
            self.kp_pitch_elevator * (commanded_pitch - aircraft_pitch) - self.kd_pitch_elevator * aircraft_pitch_rate
        )
        delta_elevator = self.saturate_elevator(delta_elevator)
        return delta_elevator

    def altitude_hold_with_pitch(self, commanded_altitude: float, aircraft_altitude: float, dt: float) -> float:
        commanded_pitch = self.PID_altitude_pitch.update(commanded_altitude, aircraft_altitude, dt)
        commanded_pitch = self.saturate_pitch(commanded_pitch)
        return commanded_pitch

    def airspeed_hold_with_throttle(
        self, commanded_airspeed: float, aircraft_airspeed: float, dt: float, throttle_trim: float = 0.0
    ) -> float:
        delta_throttle = self.PID_airspeed_throttle.update(commanded_airspeed, aircraft_airspeed, dt)
        delta_throttle = self.saturate_throttle(delta_throttle + throttle_trim)
        return delta_throttle

    def airspeed_hold_with_pitch(self, commanded_airspeed: float, aircraft_airspeed: float, dt: float) -> float:
        commanded_pitch = self.PID_airspeed_pitch.update(commanded_airspeed, aircraft_airspeed, dt)
        commanded_pitch = self.saturate_pitch(commanded_pitch)
        return commanded_pitch
