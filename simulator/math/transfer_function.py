"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np

from simulator.math.lti_systems import normalize_tf, tf2ccf, tf2zpk
from simulator.math.numeric_integration import rk4


class TransferFunction:
    """
    A class to represent a transfer function.

    Attributes
    ----------
    num : np.ndarray
        1D array representing the numerator coefficients of the transfer
        function.
    den : np.ndarray
        1D array representing the denominator coefficients of the transfer
        function.
    order : int
        The order of the transfer function, defined as the degree of the
        denominator polynomial.
    A : np.ndarray
        State matrix in controllable canonical form (n-by-n), where n is
        the order of the system.
    B : np.ndarray
        Input matrix in controllable canonical form (n-by-1).
    C : np.ndarray
        Output matrix in controllable canonical form (1-by-n).
    D : np.ndarray
        Feedthrough matrix in controllable canonical form (1-by-1).
    """

    def __init__(self, num: np.ndarray, den: np.ndarray):
        """
        Initialize the Transfer Function with numerator and denominator
        coefficients.

        Parameters
        ----------
        num : np.ndarray
            1D array representing the numerator coefficients of the transfer
            function.
        den : np.ndarray
            1D array representing the denominator coefficients of the transfer
            function.

        Raises
        ------
        ValueError
            If the numerator or denominator array is empty.
        """
        if len(num) == 0 or len(den) == 0:
            raise ValueError("Numerator and Denominator cannot be empty.")

        self.num, self.den = normalize_tf(num, den)
        self.order = len(self.den) - 1  # Order of the system

        # Calculate the State Space Model (SS) using the Controllable Canonical Form (CCF)
        self.A, self.B, self.C, self.D = tf2ccf(self.num, self.den)

        # Calculate the Zeros-Poles-Gain representation
        self.zeros, self.poles, self.gain = tf2zpk(self.num, self.den)

    def state_equation(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the state derivative dx = Ax + Bu.

        Parameters
        ----------
        x : np.ndarray
            1D array representing the state vector with shape (n,), where n is
            the order of the system.
        u : np.ndarray
            1D array representing the control input with shape (1,).

        Returns
        -------
        np.ndarray
            1D array representing the state derivative with shape (n,).
        """
        x = self._reshape_state(x)
        u = self._reshape_input(u)
        return self.A @ x + self.B @ u

    def output_equation(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the output y = Cx + Du.

        Parameters
        ----------
        x : np.ndarray
            1D array representing the state vector with shape (n,), where n is
            the order of the system.
        u : np.ndarray
            1D array representing the control input with shape (1,).

        Returns
        -------
        np.ndarray
            1D array representing the output with shape (1,).
        """
        x = self._reshape_state(x)
        u = self._reshape_input(u)
        return self.C @ x + self.D @ u

    def simulate(
        self,
        u: np.ndarray,
        t: np.ndarray,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """
        Simulate the system's response to a given control input.

        Parameters
        ----------
        u : np.ndarray
            1D array representing the control input over time with shape (n,),
            where n is the number of samples.
        t : np.ndarray
            1D array representing the time array with the same shape as `u`.
        x0 : np.ndarray, optional
            1D array representing the initial state of the system with shape
            (n,). Defaults to a zero vector.

        Returns
        -------
        y : np.ndarray
            1D array representing the output of the system over time.
        """

        u = np.atleast_1d(u)
        t = np.atleast_1d(t)

        if t.size != u.size:
            raise ValueError(
                "Length of time array 't' must match the length of control input 'u'."
            )

        if x0 is None:
            x0 = np.zeros(self.order)

        self._check_state(x0)

        y = np.zeros(u.size)
        x = np.zeros((self.order, u.size))
        x[:, 0] = x0

        for k in range(1, u.size):
            _u = u[k-1]  # take previous input (impulse response only works like that)
            dt = t[k] - t[k - 1]
            _t0 = t[k - 1]
            _x0 = x[:, k - 1]
            _, x[:, k], y[k] = self.sim_step(_u, dt, _t0, _x0)

        return y

    def sim_step(
        self, u: float, dt: float, t0: float = 0.0, x0: np.ndarray = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Perform a single simulation step to update the system's state and output 
        for a given control input.

        Parameters
        ----------
        u : float
            Control input applied during this step.
        dt : float
            Time increment for this step.
        t0 : float, optional
            Initial time at the beginning of the step, default is 0.0.
        x0 : np.ndarray, optional
            State vector at the start of the step with shape (n,),
            where n is the order of the system. Defaults to a zero vector.

        Returns
        -------
        t : float
            Updated time after the step.
        x : np.ndarray
            Updated state vector after the step with shape (n,), where n is the 
            order of the system.
        y : np.ndarray
            Output of the system after the step with shape (1,).
        """
        if x0 is None:
            x0 = np.zeros(self.order)
        dx = lambda t, x: self.state_equation(x, u)
        t = t0 + dt
        x = x0 + rk4(dx, t0, x0, dt)  # Update state using RK4
        y = self.output_equation(x, u)
        return t, x, y

    def step(
        self, x0: np.ndarray = None, t: np.ndarray = None, n: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the step response of the system.

        Parameters
        ----------
        x0 : np.ndarray, optional
            1D array representing the initial state of the system with shape
            (n,). Defaults to a zero vector.
        t : np.ndarray, optional
            1D array representing the time array. If None, a time array from 0
            to 1 seconds is generated.
        n : int, optional
            Number of samples for the simulation if `t` is not provided.
            Default is 100.

        Returns
        -------
        t : np.ndarray
            1D array representing the time array used in the simulation.
        y : np.ndarray
            1D array representing the output of the system over time.
        """
        if x0 is None:
            x0 = np.zeros((self.order,))

        self._check_state(x0)

        if t is None:
            t = np.linspace(0, 1, n)

        u = np.ones(len(t))
        y = self.simulate(u, t, x0)
        return t, y

    def impulse(
        self, x0: np.ndarray = None, t: np.ndarray = None, n: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the impulse response of the system.

        Parameters
        ----------
        x0 : np.ndarray, optional
            1D array representing the initial state of the system with shape
            (n,). Defaults to a zero vector.
        t : np.ndarray, optional
            1D array representing the time array. If None, a time array from 0
            to 1 seconds is generated.
        n : int, optional
            Number of samples for the simulation if `t` is not provided.
            Default is 100.

        Returns
        -------
        t : np.ndarray
            1D array representing the time array used in the simulation.
        y : np.ndarray
            1D array representing the output of the system over time.
        """
        if x0 is None:
            x0 = np.zeros((self.order,))

        self._check_state(x0)

        if t is None:
            t = np.linspace(0, 1, n)
        dt = np.diff(t)

        u = np.zeros(len(t))
        u[0] = 1 / dt[0]  # Impulse at t=0
        y = self.simulate(u, t, x0)
        return t, y

    def bode(self, w: np.ndarray = None):
        """
        Calculate Bode plot data for the transfer function.

        Parameters
        ----------
        w : np.ndarray, optional
            1D array of angular frequencies (rad/s) for the Bode plot. If None,
            a default frequency range is generated from 0.1 to 100 rad/s.

        Returns
        -------
        w : np.ndarray
            Frequency array used for the Bode plot.
        mag : np.ndarray
            Magnitude of the transfer function in dB.
        phase : np.ndarray
            Phase of the transfer function in degrees.
        """
        if w is None:
            w = np.logspace(-1, 2, 100)

        mag = np.zeros(len(w))
        phase = np.zeros(len(w))

        for i, freq in enumerate(w):
            s = 1j * freq  # Substitute s = jw
            # Calculate transfer function value
            H = np.polyval(self.num, s) / np.polyval(self.den, s)

            # Calculate magnitude and phase
            mag[i] = 20 * np.log10(np.abs(H))  # Magnitude in dB
            phase[i] = np.angle(H, deg=True)  # Phase in degrees

        return w, mag, phase

    def _reshape_state(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, (self.order,))

    def _reshape_input(self, u: np.ndarray) -> np.ndarray:
        return np.reshape(u, (1,))

    def _check_state(self, x: np.ndarray) -> None:
        if x.shape != (self.order,):
            raise ValueError(
                f"State must be a 1D array of size {self.order}. Got shape {x.shape}."
            )
