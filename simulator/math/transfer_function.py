"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np

from simulator.math.numeric_integration import rk4
from simulator.math.control_theory import normalize_tf, tf2ccf


class TransferFunction:
    """
    A class to represent a transfer function.

    Attributes
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.
    order : int
        The order of the transfer function, which is the degree of the denominator.
    """

    def __init__(self, num: np.ndarray, den: np.ndarray):
        """
        Initialize the Transfer Function with numerator and denominator coefficients.

        Parameters
        ----------
        num : np.ndarray
            N-by-1 array representing the numerator coefficients of the transfer function.
        den : np.ndarray
            N-by-1 array representing the denominator coefficients of the transfer function.

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

    def simulate(
        self, u: np.ndarray, x0: np.ndarray = None, t: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the transfer function system with a control input.

        Parameters
        ----------
        u : np.ndarray
            Control input array for the simulation. The number of samples must match the time array length.
        x0 : np.ndarray, optional
            Initial state of the system. The shape should match the number of states (order).
            If None, the system starts from the zero state.
        t : np.ndarray, optional
            Time array for simulation. If None, a default time array is generated based on the number of samples in u.

        Returns
        -------
        t : np.ndarray
            Time array used in simulation.
        y : np.ndarray
            Output of the system over the time array.

        Raises
        ------
        ValueError
            If the shape of `x0` does not match the number of states,
            if the lengths of `u` and `t` do not match,
            or if `t` is provided when `u` is None.
        """
        if x0 is None:
            x0 = np.zeros(self.order)  # Start from zero state if x0 is not provided

        if x0.shape[0] != self.order:
            raise ValueError(
                f"x0 shape must match the number of states (order). "
                f"Expected {self.order}, but got {x0.shape[0]}."
            )

        if t is None:
            n = u.shape[0]  # Use the length of u for the number of samples
            t = np.linspace(0.0, 1.0, n)  # Generate time array based on n

        if u.shape[0] != t.shape[0]:
            raise ValueError(
                f"Length of control input 'u' must match the length of time array 't'. "
                f"Expected {u.shape[0]} but got {t.shape[0]}."
            )

        n = t.shape[0]
        u = np.array([u])

        # Simulate using CCF
        A, B, C, D = tf2ccf(self.num, self.den)
        y = np.zeros(n)  # Output initialization
        x = np.zeros((self.order, n))  # State initialization
        x[:, 0] = x0  # Set initial state

        for k in range(1, n):
            xk = x[:, k - 1]
            uk = u[:, k]
            dt = t[k] - t[k - 1]
            dy = lambda t, y: A @ y + B @ uk
            x[:, k] = xk + rk4(dy, t[k], xk, dt)  # Update state using RK4
            y[k] = C @ x[:, k] + D @ uk  # Compute output

        return t, y

    def step(
        self, x0: np.ndarray = None, t: np.ndarray = None, n: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the step response for the transfer function system.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state of the system. The shape should match the number of states (order).
            If None, the system starts from the zero state.
        t : np.ndarray, optional
            Time array for simulation. If provided, `n` is set to the length of `t`.
            - If `t` is None and `n` is provided, a time array is generated from 0 to 1 seconds with `n` samples.
            - If neither is provided, defaults to `n=100` and generates `t` using `np.linspace(0, 1, 100)`.
        n : int, optional
            Number of samples for simulation. If `t` is provided, `n` value is ignored.

        Returns
        -------
        t : np.ndarray
            Time array used in simulation.
        y : np.ndarray
            Output of the system over the time array.

        Raises
        ------
        ValueError
            If the shape of `x0` does not match the number of states.
        """
        if x0 is None:
            x0 = np.zeros((self.order,))  # Start from zero state if x0 is not provided

        if x0.shape[0] != self.order:
            raise ValueError(
                f"x0 shape must match the number of states (order). "
                f"Expected {self.order}, got {x0.shape[0]}."
            )

        if t is not None:
            n = len(t)  # Use length of t
        elif n is not None:
            t = np.linspace(0, 1, n)  # Generate time array based on n
        else:
            n = 100  # Default samples
            t = np.linspace(0, 1, n)  # Default time array

        u = np.ones(n)
        return self.simulate(u, x0, t)
