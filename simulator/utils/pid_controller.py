"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.common.constants import EPS


class pid_controller:
    def __init__(
        self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0, **kwargs
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        max_output = kwargs.get("max_output")
        min_output = kwargs.get("min_output")
        if max_output is None and min_output is None:
            self.saturate = lambda x: x
        else:
            self.saturate = lambda x: np.clip(x, min_output, max_output)

        self.tau = kwargs.get("tau")

        self.prev_error = 0.0
        self.prev_intg = 0.0
        self.prev_diff = 0.0

    @staticmethod
    def integrate(
        error: float, error_prev: float, intg_prev: float, dt: float
    ) -> float:
        return intg_prev + dt / 2.0 * (error + error_prev)

    @staticmethod
    def differenciate(
        error: float, error_prev: float, diff_prev: float, dt: float, tau: float = None
    ) -> float:
        if tau is None:
            tau = 5.0 * dt
        return ((2.0 * tau - dt) / (2.0 * tau + dt)) * diff_prev + (
            2.0 / (2.0 * tau + dt)
        ) * (error - error_prev)

    def update(self, command: float, state: float, dt: float) -> float:

        error = command - state

        # proportional term
        control = self.kp * error  # kp*e

        # integral term
        if np.abs(self.ki) > EPS:
            error_intg = pid_controller.integrate(
                error, self.prev_error, self.prev_intg, dt
            )
            control += self.ki * error_intg  # ki/s*e
            # self.prev_intg = error_intg

        # derivative term
        if np.abs(self.kd) > EPS:
            error_diff = pid_controller.differenciate(
                error, self.prev_error, self.prev_diff, dt, self.tau
            )
            control += self.kd * error_diff  # kd*s*e
            self.prev_diff = error_diff

        self.prev_error = error

        # control action (control = kp*e + ki/s*e + kd*s*e)
        control_unsat = control
        control = self.saturate(control)  # saturate the output to its range

        # integrator anti-windup
        if np.abs(self.ki) > EPS:
            self.prev_intg = self.prev_intg + dt / self.ki * (control - control_unsat)

        return control

    def reset(self) -> None:
        self.prev_error = 0.0
        self.prev_intg = 0.0
        self.prev_diff = 0.0
