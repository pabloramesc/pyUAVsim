"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np


def normalize_tf(num: np.ndarray, den: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the transfer function by making the leading coefficient of the denominator equal to 1.

    Parameters
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the normalized numerator and denominator coefficients.
    """
    if den[0] != 1:
        factor = den[0]
        den = [coeff / factor for coeff in den]
        num = [coeff / factor for coeff in num]
    return num, den


def tf2ccf(
    num: np.ndarray, den: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in Controllable Canonical Form (CCF).

    Parameters
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.

    Returns
    -------
    A : np.ndarray
        n-by-n state matrix (system dynamics matrix).
    B : np.ndarray
        n-by-1 input matrix.
    C : np.ndarray
        1-by-n output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.

    Notes
    -----
    In Controllable Canonical Form (CCF), the input matrix directly influencesthe states of the system.
    This representation is useful for controller design and analysis,
    as it highlights the controllability properties of the system.
    """
    _num, _den = normalize_tf(num, den)

    # Order of the system
    n = len(_den) - 1

    # num and den coefs
    a = np.array(_den)
    b = np.zeros(len(_den))
    b[len(_den) - len(_num) :] = np.array(_num)

    # A matrix (System dynamics matrix in CCF)
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -a[1:][::-1]

    # B matrix (Input matrix)
    B = np.zeros((n, 1))
    B[-1, 0] = 1.0

    # C matrix (Output matrix)
    C = np.zeros((1, n))
    C[0, :] = b[1:][::-1] - a[1:][::-1] * b[0]

    # D matrix (Feedthrough matrix)
    D = np.zeros((1, 1))
    D[0, 0] = b[0]

    return A, B, C, D


def tf2ocf(
    num: np.ndarray, den: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in Observable Canonical Form (OCF).

    Parameters
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.

    Returns
    -------
    A : np.ndarray
        n-by-n state matrix (system dynamics matrix).
    B : np.ndarray
        n-by-1 input matrix.
    C : np.ndarray
        1-by-n output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.

    Notes
    -----
    In Observable Canonical Form (OCF), the system's states are observable from the output.
    This means that the output can provide information about the internal states of the system.
    The OCF representation is particularly useful for observer design and state estimation.
    """
    _num, _den = normalize_tf(num, den)

    # Order of the system
    n = len(_den) - 1

    # num and den coefs
    a = np.array(_den)
    b = np.zeros(len(_den))
    b[len(_den) - len(_num) :] = np.array(_num)

    # A matrix (System dynamics matrix in OCF)
    A = np.zeros((n, n))
    A[1:, :-1] = np.eye(n - 1)
    A[:, -1] = -a[1:][::-1]

    # B matrix (Input matrix)
    B = np.zeros((n, 1))
    B[:, 0] = b[1:][::-1] - a[1:][::-1] * b[0]

    # C matrix (Output matrix)
    C = np.zeros((1, n))
    C[0, -1] = 1.0

    # D matrix (Feedthrough matrix)
    D = np.zeros((1, 1))
    D[0, 0] = b[0]

    return A, B, C, D


def tf2ss(
    num: np.ndarray, den: np.ndarray, form: str = "ccf"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in either Controllable Canonical Form (CCF)
    or Observable Canonical Form (OCF).

    Parameters
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.
    form : str, optional
        The desired canonical form, either 'ccf' or 'ocf' (default is 'ccf').

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the A, B, C, and D matrices of the state-space representation.

    Raises
    ------
    ValueError
        If the form parameter is not valid (i.e., not 'ccf' or 'ocf').
    """
    if form == "ccf":
        return tf2ccf(num, den)
    elif form == "ocf":
        return tf2ocf(num, den)
    else:
        raise ValueError("Not valid form parameter!")
