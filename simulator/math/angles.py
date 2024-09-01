"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

def clip_angle_pi(angle: float) -> float:
    """
    Modify angle value to fit inside the range (-PI, PI].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Angle in radians clipped to the range (-PI, PI].
    """
    angle %= 2 * np.pi
    if angle > +np.pi:
        return angle % -np.pi
    return angle


def clip_angle_2pi(angle: float) -> float:
    """
    Modify angle value to fit inside the range (0, 2*PI].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Angle in radians clipped to the range (0, 2*PI].
    """
    angle %= 2 * np.pi
    if angle == 0.0:
        return 2 * np.pi
    return angle


def clip_angle_180(angle: float) -> float:
    """
    Modify angle value to fit inside the range (-180, 180].

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    float
        Angle in degrees clipped to the range (-180, 180].
    """
    angle %= 360
    if angle > +180.0:
        return angle % -180.0
    return angle


def clip_angle_360(angle: float) -> float:
    """
    Modify angle value to fit inside the range (0, 360].

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    float
        Angle in degrees clipped to the range (0, 360].
    """
    angle %= 360
    if angle == 0.0:
        return 360.0
    return angle


def diff_angle_pi(angle1: float, angle2: float) -> float:
    """
    Compute the difference between two angles in radians,
    returning a result in the range (-PI, PI].

    Parameters
    ----------
    angle1 : float
        First angle in radians.
    angle2 : float
        Second angle in radians.

    Returns
    -------
    float
        Difference between the two angles in radians.
    """
    while angle1 - angle2 > +np.pi:
        angle1 -= 2 * np.pi
    while angle1 - angle2 < -np.pi:
        angle1 += 2 * np.pi
    return angle1 - angle2


def diff_angle_180(angle1: float, angle2: float) -> float:
    """
    Compute the difference between two angles in degrees,
    returning a result in the range (-180, 180].

    Parameters
    ----------
    angle1 : float
        First angle in degrees.
    angle2 : float
        Second angle in degrees.

    Returns
    -------
    float
        Difference between the two angles in degrees.
    """
    while angle1 - angle2 > +180.0:
        angle1 = angle1 - 360.0
    while angle1 - angle2 < -180.0:
        angle1 = angle1 + 360.0
    return angle1 - angle2

