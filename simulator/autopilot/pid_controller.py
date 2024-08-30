"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.common.constants import EPS


class PIDController:
    """
    A PID (Proportional-Integral-Derivative) controller implementation.

    Attributes
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    tau : float
        Time constant for derivative filtering.
    saturate : function
        Function to apply output saturation.
    prev_error : float
        Previous error value for derivative calculation.
    prev_intg : float
        Previous integral value for integral calculation.
    prev_diff : float
        Previous derivative value for derivative calculation.

    Methods
    -------
    update(command, state, dt)
        Updates the PID controller and computes the control output.
    reset()
        Resets the PID controller state.
    """

    def __init__(
        self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0, **kwargs
    ) -> None:
        """Initialize the PIDController class.

        Parameters
        ----------
        kp : float, optional
            Proportional gain, by default 0.0.
        ki : float, optional
            Integral gain, by default 0.0.
        kd : float, optional
            Derivative gain, by default 0.0.
        tau : float, optional
            Time constant for derivative filtering, by default 5 times the time step (dt).
        max_output : float, optional
            Maximum output value for saturation, by default None (no saturation).
        min_output : float, optional
            Minimum output value for saturation, by default None (no saturation).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau = kwargs.get("tau")
        self.max_output = kwargs.get("max_output")
        self.min_output = kwargs.get("min_output")

        self.reset()

    @staticmethod
    def integrate(error: float, error_prev: float, intg_prev: float, dt: float):
        """
        Computes the integral of the error using the trapezoidal rule.

        Parameters
        ----------
        error : float
            Current error value.
        error_prev : float
            Previous error value.
        intg_prev : float
            Previous integral value.
        dt : float
            Time step.

        Returns
        -------
        float
            Updated integral value.
        """
        return intg_prev + dt / 2.0 * (error + error_prev)

    @staticmethod
    def differentiate(
        error: float, error_prev: float, diff_prev: float, dt: float, tau: float = None
    ):
        """
        Computes the derivative of the error with filtering.

        Parameters
        ----------
        error : float
            Current error value.
        error_prev : float
            Previous error value.
        diff_prev : float
            Previous derivative value.
        dt : float
            Time step.
        tau : float, optional
            Time constant for filtering, by default calculated as 5 times dt.

        Returns
        -------
        float
            Updated derivative value.
        """
        if tau is None:
            tau = 5.0 * dt
        alpha = (2.0 * tau - dt) / (2.0 * tau + dt)
        return alpha * diff_prev + (2.0 / (2.0 * tau + dt)) * (error - error_prev)

    def update(self, x_ref: float, x: float, dt):
        """
        Updates the PID controller with a new command and state, computes the control output.

        Parameters
        ----------
        x_ref : float
            Desired setpoint.
        x : float
            Current process variable.
        dt : float
            Time step.

        Returns
        -------
        float
            The control output.
        """
        error = x_ref - x

        # Proportional term
        u = self.kp * error

        # Integral term
        if np.abs(self.ki) > EPS:
            self.prev_intg = self.integrate(error, self.prev_error, self.prev_intg, dt)
            u += self.ki * self.prev_intg

        # Derivative term
        if np.abs(self.kd) > EPS:
            self.prev_diff = self.differentiate(
                error, self.prev_error, self.prev_diff, dt, self.tau
            )
            u += self.kd * self.prev_diff

        self.prev_error = error

        # Saturate output
        u_unsat = u
        u = np.clip(u, self.min_output, self.max_output)

        # Anti-windup
        if np.abs(self.ki) > EPS:
            self.prev_intg += dt / self.ki * (u - u_unsat)

        return u

    def reset(self):
        """
        Resets the PID controller state (error, integral, derivative).
        """
        self.prev_error = 0.0
        self.prev_intg = 0.0
        self.prev_diff = 0.0
