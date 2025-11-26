"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np


def jacobian(func: callable, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate the Jacobian matrix of the function `func` at the point `x0`.

    Parameters
    ----------
    func : callable
        A function that takes an n-size array and returns and m-size array
    x0 : np.ndarray
        A n-size array with the coordinates of the point at which to compute
        the Jacobian
    eps : float, optional
        The finite difference step size, by default 1e-6

    Returns
    -------
    np.ndarray
        The Jacobian as m-by-n matrix
    """
    f0 = func(x0)
    n = x0.size
    m = f0.size
    Jac = np.zeros((m, n))
    for i in range(n):
        x_eps = np.copy(x0)
        x_eps[i] += eps  # increment the i-th state
        f_eps = func(x_eps)
        Jac[:, i] = (f_eps - f0) / eps
    return Jac
