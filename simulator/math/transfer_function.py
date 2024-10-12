"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np

from simulator.math.lti_systems import normalize_tf, tf2ccf
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

        _num, _den = normalize_tf(num, den)
        self.num = np.array(_num)
        self.den = np.array(_den)
        self.order = len(self.den) - 1  # Order of the system

        # Calculate the State Space Model (SS) using the Controllable Canonical Form (CCF)
        self.A, self.B, self.C, self.D = tf2ccf(self.num, self.den)

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
        self, u: np.ndarray, x0: np.ndarray = None, t: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the system's response to a given control input.

        Parameters
        ----------
        u : np.ndarray
            1D array representing the control input over time with shape (n,),
            where n is the number of samples.
        x0 : np.ndarray, optional
            1D array representing the initial state of the system with shape
            (n,). Defaults to a zero vector.
        t : np.ndarray, optional
            1D array representing the time array. If None, a default time array
            from 0 to 1 seconds is generated, with the same length as `u`.

        Returns
        -------
        t : np.ndarray
            1D array representing the time array used in the simulation.
        y : np.ndarray
            1D array representing the output of the system over time.
        """
        if x0 is None:
            x0 = np.zeros((self.order, 1))

        self._check_state(x0)

        if t is None:
            t = np.linspace(0.0, 1.0, len(u))
        elif len(u) != len(t):
            raise ValueError(
                "Length of control input 'u' must match the length of time array 't'."
            )

        n = len(t)
        y = np.zeros(n)
        x = np.zeros((self.order, n))
        x[:, 0] = x0
        u = np.reshape(u, (1, n))

        for k in range(1, n):
            xk = x[:, k - 1]
            uk = u[:, k]
            dt = t[k] - t[k - 1]
            dx = lambda t, x: self.state_equation(x, uk)
            x[:, k] = xk + rk4(dx, t[k], xk, dt)  # Update state using RK4
            y[k] = self.output_equation(xk, uk)

        return t, y

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
            Number of samples for the simulation if `t` is not provided. Default
            is 100.

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

        return self.simulate(np.ones(len(t)), x0, t)

    def _reshape_state(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, (self.order,))

    def _reshape_input(self, u: np.ndarray) -> np.ndarray:
        return np.reshape(u, (1,))

    def _check_state(self, x: np.ndarray) -> None:
        if x.shape != (self.order,):
            raise ValueError(
                f"State must be a 1D array of size {self.order}. Got shape {x.shape}."
            )
