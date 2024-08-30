"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_configuration import AutopilotConfiguration
from simulator.autopilot.control.pid_control import PID_control
from simulator.common.constants import PI
from simulator.sensors.state_handler import StateHandler


class LateralControl:
    def __init__(self, config: AutopilotConfiguration, state: StateHandler = None) -> None:

        self.state = state

        self.time_prev = 0.0

        ##### ROLL CONTROL WITH AILERON #####
        self.kp_roll_aileron = config.kp_roll_aileron
        self.kd_roll_aileron = config.kd_roll_aileron
        self.saturate_aileron = lambda x: np.clip(x, config.min_aileron, config.max_aileron)

        ##### COURSE CONTROL WITH ROLL #####
        # self.kp_course_roll = params.kp_course_roll
        # self.ki_course_roll = params.ki_course_roll
        self.PID_course_roll = PID_control(kp=config.kp_course_roll, ki=config.ki_course_roll)
        self.saturate_roll = lambda x: np.clip(x, config.min_roll, config.max_roll)

        ##### SIDESLIP CONTROL WITH RUDDER #####
        # self.kp_sideslip_rudder = params.kp_sideslip_rudder
        # self.ki_sideslip_rudder = params.ki_sideslip_rudder
        self.PID_sideslip_rudder = PID_control(kp=config.kp_sideslip_rudder, ki=config.ki_sideslip_rudder)
        self.saturate_rudder = lambda x: np.clip(x, config.min_rudder, config.max_rudder)

        ##### YAW DAMPER WITH RUDDER #####
        self.xi = 0.0
        self.Ts = config.Ts_damper
        self.kr = config.kr_damper
        self.p_wo = config.p_wo

    def wrap_course(self, commanded_course: float, aircraft_course: float):
        while commanded_course - aircraft_course > +PI:
            commanded_course = commanded_course - 2.0 * PI
        while commanded_course - aircraft_course < -PI:
            commanded_course = commanded_course + 2.0 * PI
        return commanded_course

    def roll_hold_with_aileron(self, commanded_roll: float, aircraft_roll: float, aircraf_roll_rate: float) -> float:
        delta_aileron = (
            self.kp_roll_aileron * (commanded_roll - aircraft_roll) - self.kd_roll_aileron * aircraf_roll_rate
        )
        delta_aileron = self.saturate_aileron(delta_aileron)
        return delta_aileron

    def course_hold_with_roll(self, commanded_course: float, aircraft_course: float, dt: float) -> float:
        commanded_course = self.wrap_course(commanded_course, aircraft_course)
        commanded_roll = self.PID_course_roll.update(commanded_course, aircraft_course, dt)
        commanded_roll = self.saturate_roll(commanded_roll)
        return commanded_roll

    def sideslip_hold_with_rudder(self, aircraft_sideslip: float, dt: float) -> float:
        delta_rudder = self.PID_sideslip_rudder.update(0.0, aircraft_sideslip, dt)
        delta_rudder = self.saturate_rudder(delta_rudder)
        return delta_rudder

    def yaw_damper_with_rudder(self, aircraft_yaw_rate: float, dt: float) -> float:
        if self.Ts is None:
            Ts = 5.0 * dt
        else:
            Ts = self.Ts
        self.xi = self.xi + Ts * (-self.p_wo * self.xi + self.kr * aircraft_yaw_rate)
        delta_rudder = -self.p_wo * self.xi + self.kr * aircraft_yaw_rate
        return delta_rudder

    def control(self, time: float, commanded_course: float) -> tuple:
        dt = time - self.time_prev
        self.time_prev = time

        aircraft_course = self.state.aircraft.course_angle
        commanded_roll = self.course_hold_with_roll(commanded_course, aircraft_course, dt)

        aircraft_roll = self.state.aircraft.attitude[0]
        aircraft_roll_rate = self.state.aircraft.omega[0]
        delta_aileron = self.roll_hold_with_aileron(commanded_roll, aircraft_roll, aircraft_roll_rate)

        aircraft_sideslip = self.state.aircraft.side_slip_angle
        aircraft_yaw_rate = self.state.aircraft.omega[2]
        delta_rudder = 0.0
        # delta_rudder += self.sideslip_hold_with_rudder(aircraft_sideslip, dt)
        delta_rudder += self.yaw_damper_with_rudder(aircraft_yaw_rate, dt)

        return (delta_aileron, delta_rudder)
