"""
Utility functions for angle manipulations in radians.

These functions handle both single float values and NumPy arrays.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, overload

FloatOrArray = Union[float, NDArray[np.floating]]


@overload
def wrap_angle_pi(angle: float) -> float: ...
@overload
def wrap_angle_pi(angle: NDArray[np.floating]) -> NDArray[np.floating]: ...


def wrap_angle_pi(angle: FloatOrArray) -> FloatOrArray:
    """
    Wrap angle(s) to the range [-pi, pi).

    Args:
        angle (float or array): Angle(s) in radians.

    Returns:
        Wrapped angle(s) in radians [-pi, pi).
    """
    a = np.atleast_1d(angle).astype(np.float64)
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a if isinstance(angle, np.ndarray) else a.item()


@overload
def wrap_angle_2pi(angle: float) -> float: ...
@overload
def wrap_angle_2pi(angle: NDArray[np.floating]) -> NDArray[np.floating]: ...


def wrap_angle_2pi(angle: FloatOrArray) -> FloatOrArray:
    """
    Wrap angle(s) to the range (0, 2*pi].

    Args:
        angle (float or array): Angle(s) in radians.

    Returns:
        Wrapped angle(s) in radians (0, 2*pi].
    """
    a = np.atleast_1d(angle).astype(np.float64)
    a = a % (2 * np.pi)
    a[a == 0.0] = 2 * np.pi
    return a if isinstance(angle, np.ndarray) else a.item()


@overload
def diff_angle_pi(angle1: float, angle2: float) -> float: ...
@overload
def diff_angle_pi(
    angle1: NDArray[np.floating], angle2: NDArray[np.floating]
) -> NDArray[np.floating]: ...


def diff_angle_pi(angle1: FloatOrArray, angle2: FloatOrArray) -> FloatOrArray:
    """
    Calculate the difference between two angles (or arrays of angles)
    ensuring the result is in the range [-pi, pi).

    Args:
        angle1 (float or array): First angle(s) in radians.
        angle2 (float or array): Second angle(s) in radians.

    Returns:
        The difference between angle1 and angle2 in radians [-pi, pi).
    """
    diff = np.asarray(angle1) - np.asarray(angle2)
    wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
    return (
        wrapped
        if isinstance(angle1, np.ndarray) or isinstance(angle2, np.ndarray)
        else wrapped.item()
    )


@overload
def wrap_angle(angle1: float, angle2: float) -> float: ...
@overload
def wrap_angle(
    angle1: NDArray[np.floating], angle2: NDArray[np.floating]
) -> NDArray[np.floating]: ...


def wrap_angle(angle1: FloatOrArray, angle2: FloatOrArray) -> FloatOrArray:
    """
    Wrap angle1 so that difference between angle1 and angle2
    is in the range [-pi, pi].

    Args:
        angle1 (float or array): Angle(s) in radians to be wrapped.
        angle2 (float or array): Reference angle(s) in radians.

    Returns:
        Wrapped angle1 in radians so that difference with angle2 is in [-pi, pi].
    """
    a1 = np.asarray(angle1)
    a2 = np.asarray(angle2)
    wrapped = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi + a2
    return wrapped if isinstance(angle1, np.ndarray) else wrapped.item()
