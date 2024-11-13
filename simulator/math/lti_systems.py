"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def normalize_tf(num: ArrayLike, den: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the transfer function by making the leading coefficient of the
    denominator equal to 1.

    This function trims leading zeros from both the numerator and denominator
    coefficients and checks if denominator polynomial is empty.

    Parameters
    ----------
    num : array_like
        1D array-like object representing the coefficients of the numerator
        polynomial in descending order of degree.
    den : array_like
        1D array-like object representing the coefficients of the denominator
        polynomial in descending order of degree.

    Returns
    -------
    num : np.ndarray
        The normalized numerator coefficients as a 1-D array.
    den : np.ndarray
        The normalized denominator coefficients as a 1-D array.

    Raises
    ------
    ValueError
        If the denominator polynomial is empty.

    Examples
    --------
    >>> num = [1, 5]
    >>> den = [2, -3, 1]
    >>> normalize_tf(num, den)
    (array([0.5, 2.5]), array([ 1. , -1.5,  0.5]))
    """
    _num = np.array(np.trim_zeros(num, "f"))
    _den = np.array(np.trim_zeros(den, "f"))

    if len(den) == 0:
        raise ValueError("Denominator cannot be empty!")

    factor = _den[0]
    _den = _den / factor
    _num = _num / factor

    return _num, _den


def tf2ccf(
    num: ArrayLike, den: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in Controllable
    Canonical Form (CCF).

    Parameters
    ----------
    num : array_like
        1D array-like object representing the coefficients of the numerator
        polynomial in descending order of degree.
    den : array_like
        1D array-like object representing the coefficients of the denominator
        polynomial in descending order of degree.

    Returns
    -------
    A : np.ndarray
        N-by-N state matrix (system dynamics matrix),
        where N is the order of the system.
    B : np.ndarray
        N-by-1 input matrix.
    C : np.ndarray
        1-by-N output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.

    Notes
    -----
    In Controllable Canonical Form (CCF), the input matrix directly influences
    the states of the system. This representation is useful for controller
    design and analysis, as it highlights the controllability properties of the
    system.

    Examples
    --------
    >>> num = [1, 5]
    >>> den = [2, -3, 1]
    >>> A, B, C, D = tf2ccf(num, den)
    >>> A
    array([[ 0. ,  1. ],
        [-0.5,  1.5]])
    >>> B
    array([[0.],
        [1.]])
    >>> C
    array([[2.5, 0.5]])
    >>> D
    array([[0.]])
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
    num: ArrayLike, den: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in Observable
    Canonical Form (OCF).

    Parameters
    ----------
    num : array_like
        1D array-like object representing the coefficients of the numerator
        polynomial in descending order of degree.
    den : array_like
        1D array-like object representing the coefficients of the denominator
        polynomial in descending order of degree.

    Returns
    -------
    A : np.ndarray
        N-by-N state matrix (system dynamics matrix),
        where N is the order of the system.
    B : np.ndarray
        N-by-1 input matrix.
    C : np.ndarray
        1-by-N output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.

    Notes
    -----
    In Observable Canonical Form (OCF), the system's states are observable
    from the output. This means that the output can provide information about
    the internal states of the system. The OCF representation is particularly
    useful for observer design and state estimation.

    Examples
    --------
    >>> num = [1, 5]
    >>> den = [2, -3, 1]
    >>> A, B, C, D = tf2ocf(num, den)
    >>> A
    array([[ 0. , -0.5],
        [ 1. ,  1.5]])
    >>> B
    array([[2.5],
        [0.5]])
    >>> C
    array([[0., 1.]])
    >>> D
    array([[0.]])
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
    num: ArrayLike, den: ArrayLike, form: str = "ccf"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a transfer function to state-space representation in either
    Controllable Canonical Form (CCF) or Observable Canonical Form (OCF).

    Parameters
    ----------
    num : array_like
        1D array-like object representing the coefficients of the numerator
        polynomial in descending order of degree.
    den : array_like
        1D array-like object representing the coefficients of the denominator
        polynomial in descending order of degree.
    form : str, optional
        The desired canonical form, either 'ccf' or 'ocf' (default is 'ccf').


    Returns
    -------
    A : np.ndarray
        N-by-N state matrix (system dynamics matrix),
        where N is the order of the system.
    B : np.ndarray
        N-by-1 input matrix.
    C : np.ndarray
        1-by-N output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.

    Raises
    ------
    ValueError
        If the form parameter is not valid (i.e., not 'ccf' or 'ocf').

    Examples
    --------
    >>> num = [1, 5]
    >>> den = [2, -3, 1]
    >>> A, B, C, D = tf2ss(num, den)
    >>> A
    array([[ 0. ,  1. ],
        [-0.5,  1.5]])
    >>> B
    array([[0.],
        [1.]])
    >>> C
    array([[2.5, 0.5]])
    >>> D
    array([[0.]])
    """
    if form == "ccf":
        return tf2ccf(num, den)
    elif form == "ocf":
        return tf2ocf(num, den)
    else:
        raise ValueError("Not valid form parameter!")


def tf2zpk(num: ArrayLike, den: ArrayLike) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a transfer function to zero-pole-gain (ZPK) representation.

    Parameters
    ----------
    num : array_like
        1D array-like object representing the coefficients of the numerator
        polynomial in descending order of degree.
    den : array_like
        1D array-like object representing the coefficients of the denominator
        polynomial in descending order of degree.

    Returns
    -------
    zeros : np.ndarray
        1D array containing the zeros of the transfer function.
    poles : np.ndarray
        1D array containing the poles of the transfer function.
    gain : float
        The gain of the transfer function.

    Notes
    -----
    The zeros are the roots of the numerator polynomial, and the poles are the
    roots of the denominator polynomial. The gain is the constant factor that
    relates the output to the input at steady state.

    Examples
    --------
    >>> num = np.array([1, 5])
    >>> den = np.array([2, -3, 1])
    >>> tf2zpk(num, den)
    (array([-5.]), array([1. , 0.5]), np.float64(0.5))
    """
    _num, _den = normalize_tf(num, den)

    zeros = np.roots(_num)
    poles = np.roots(_den)
    gain = _num[0] / _den[0]

    return zeros, poles, gain


def zpk2tf(
    zeros: ArrayLike, poles: ArrayLike, gain: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert zero-pole-gain (ZPK) representation to a transfer function.

    Parameters
    ----------
    zeros : array_like
        1D array-like object representing the zeros of the transfer function.
    poles : array_like
        1D array-like object representing the poles of the transfer function.
    gain : float
        The gain of the transfer function.

    Returns
    -------
    num : np.ndarray
        1D array containing the coefficients of the numerator polynomial in
        descending order of degree.
    den : np.ndarray
        1D array containing the coefficients of the denominator polynomial in
        descending order of degree.

    Notes
    -----
    The transfer function is constructed by multiplying the factors
    corresponding to the zeros and poles, scaled by the gain. This can be
    useful for control design and analysis.

    Examples
    --------
    >>> zeros = [-5]
    >>> poles = [1, 0.5]
    >>> gain = 0.5
    >>> zpk2tf(zeros, poles, gain)
    (array([0.5, 2.5]), array([ 1. , -1.5,  0.5]))
    """
    num = gain * np.poly(zeros)
    den = np.poly(poles)
    return np.atleast_1d(num), np.atleast_1d(den)
