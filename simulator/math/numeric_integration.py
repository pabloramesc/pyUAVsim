"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

def euler(f: callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a single step of the Euler method.

    Parameters
    ----------
    f : callable
        The function that defines the ODE: dy/dt = f(t, y).
    t : float
        Current time.
    y : float or np.ndarray
        Current value of the dependent variable.
    dt : float
        Time step.

    Returns
    -------
    float or np.ndarray
        Incremental change (dy) after time step dt.
    """
    dy = f(t, y) * dt
    return dy

def rk4(f: callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a single step of the 4th order Runge-Kutta (RK4) method.

    Parameters
    ----------
    f : callable
        The function that defines the ODE: dy/dt = f(t, y).
    t : float
        Current time.
    y : float or np.ndarray
        Current value of the dependent variable.
    dt : float
        Time step.

    Returns
    -------
    float or np.ndarray
        Incremental change (dy) after time step dt.
    """
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt / 2, y + k1 / 2)
    k3 = dt * f(t + dt / 2, y + k2 / 2)
    k4 = dt * f(t + dt, y + k3)
    dy = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return dy


def crank_nicolson(f: callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a single step of the Crank-Nicolson method.

    Parameters
    ----------
    f : callable
        The function that defines the ODE: dy/dt = f(t, y).
    t : float
        Current time.
    y : float or np.ndarray
        Current value of the dependent variable.
    dt : float
        Time step.

    Returns
    -------
    float or np.ndarray
        Incremental change (dy) after time step dt.
    """
    # Predictor step (Euler method)
    yp = y + dt * f(t, y)
    # Corrector step (average of Euler and implicit Euler)
    dy = (dt / 2) * (f(t, y) + f(t + dt, yp))
    return dy
