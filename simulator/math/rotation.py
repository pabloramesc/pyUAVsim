"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy import sin, cos

def rot_matrix_axis(axis: np.ndarray = np.zeros(3), t: float = 0.0) -> np.ndarray:
    """
    Compute transformation matrix from vehicle frame to body frame (R^b_v)
    with the aircraft attitude expressed as an axis and a rotation for that axis.

    Parameters
    ----------
    axis : np.ndarray, optional
        3-size array with rotation axis components (ux, uy, uz), by default np.zeros(3)
    t : float, optional
        rotation in rad, by default 0.0

    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    R = np.array(
        [
            [
                cos(t) + ux**2 * (1 - cos(t)),
                ux * uy * (1 - cos(t)) - uz * sin(t),
                ux * uz * (1 - cos(t)) + uy * sin(t),
            ],
            [
                uy * ux * (1 - cos(t)) + uz * sin(t),
                cos(t) + uy**2 * (1 - cos(t)),
                uy * uz * (1 - cos(t)) - ux * sin(t),
            ],
            [
                uz * ux * (1 - cos(t)) - uy * sin(t),
                uz * uy * (1 - cos(t)) + ux * sin(t),
                cos(t) + uz**2 * (1 - cos(t)),
            ],
        ]
    )
    return R


def rot_matrix_zyx(euler: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Compute transformation matrix from vehicle frame to body frame (R^b_v)
    with the aircraft attitude expressed as euler angles in ZYX sequence.

    Parameters
    ----------
    euler : np.ndarray, optional
        3-size array with euler angles (roll, pitch, yaw) in rad, by default np.zeros(3)

    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    r = euler[0]
    p = euler[1]
    y = euler[2]
    sr = sin(r)
    cr = cos(r)
    sp = sin(p)
    cp = cos(p)
    sy = sin(y)
    cy = cos(y)
    R = np.zeros((3, 3))
    R[0, 0] = cp * cy
    R[0, 1] = cp * sy
    R[0, 2] = -sp
    R[1, 0] = sr * sp * cy - cr * sy
    R[1, 1] = sr * sp * sy + cr * cy
    R[1, 2] = sr * cp
    R[2, 0] = cr * sp * cy + sr * sy
    R[2, 1] = cr * sp * sy - sr * cy
    R[2, 2] = cr * cp
    return R


def rotate(
    R: np.ndarray = np.eye(3),
    origin: np.ndarray = np.zeros(3),
    point: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """Rotate a point around an origin using a given rotation matrix.

    Parameters
    ----------
    R : np.ndarray, optional
        3x3 rotation matrix, by default np.eye(3)
    origin : np.ndarray, optional
        3-element array representing the origin, by default np.zeros(3)
    point : np.ndarray, optional
        3-element array representing the point to be rotated, by default np.zeros(3)

    Returns
    -------
    np.ndarray
        3-element array representing the rotated point.
    """
    return origin + np.dot(R.T, (point - origin))


def rotate_points(
    R: np.ndarray = np.eye(3),
    origin: np.ndarray = np.zeros(3),
    points: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """Rotate multiple points around an origin using a given rotation matrix.

    Parameters
    ----------
    R : np.ndarray, optional
        3x3 rotation matrix, by default np.eye(3)
    origin : np.ndarray, optional
        3-element array representing the origin, by default np.zeros(3)
    points : np.ndarray, optional
        N-by-3 size array of N points to be rotated, by default np.zeros(3)

    Returns
    -------
    np.ndarray
        N-by-3 size array of N rotated points
    """
    N = points.shape[0]
    rot_points = np.zeros((N, 3))
    for k in range(N):
        rot_points[k, :] = rotate(R, origin, points[k, :])
    return rot_points


def multi_rotation(
    angles: np.ndarray = np.zeros((100, 3)), points: np.ndarray = np.zeros((100, 3))
) -> np.ndarray:
    """Rotate multiple points using corresponding Euler angles in ZYX sequence.

    Parameters
    ----------
    angles : np.ndarray, optional
        N-by-3 size array of N Euler angles (roll, pitch, yaw) in radians, by default np.zeros((100, 3))
    points : np.ndarray, optional
        N-by-3 size array of N points to be rotated, by default np.zeros((100, 3))

    Returns
    -------
    np.ndarray
        N-by-3 size array of N rotated points.

    Raises
    ------
    ValueError
         If angles and points do not have the same shape.
    """
    if angles.shape != points.shape:
        raise ValueError("angles and points must have same shape")
    N = angles.shape[0]
    rot_points = np.zeros((N, 3))
    for k in range(N):
        R = rot_matrix_zyx(angles[k, :])
        rot_points[k, :] = np.dot(R.T, points[k, :])
    return rot_points


def ned2xyz(ned_coords: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Change coordinates from the NED (North-East-Down) frame to XYZ frame.
    
    In the XYZ frame:
    - X corresponds to East.
    - Y corresponds to North.
    - Z corresponds to the negative of Down.

    Parameters
    ----------
    ned_coords : np.ndarray, optional
        Coordinates in the NED frame, by default np.zeros(3).
        This can be a 1D array of length 3 or a 2D array of shape (N, 3).

    Returns
    -------
    np.ndarray
        Coordinates in the XYZ frame.
        This will have the same shape as the input.
    """
    if ned_coords.ndim > 1:
        N = ned_coords.shape[0]
        xyz_coords = np.zeros((N, 3))
        xyz_coords[:, 0] = ned_coords[:, 1]
        xyz_coords[:, 1] = ned_coords[:, 0]
        xyz_coords[:, 2] = -ned_coords[:, 2]
    else:
        xyz_coords = np.zeros(3)
        xyz_coords[0] = ned_coords[1]
        xyz_coords[1] = ned_coords[0]
        xyz_coords[2] = -ned_coords[2]
    return xyz_coords


def euler2quat(att):
    """Convert Euler angles to quaternion.

    Parameters
    ----------
    att : np.ndarray
        Array of Euler angles (roll, pitch, yaw) in radians.
        This can be a 1D array of length 3 or a 2D array of shape (N, 3).

    Returns
    -------
    np.ndarray
        Quaternion (q0, q1, q2, q3]).
        This can be a 1D array of length 4 or a 2D array of shape (N, 4),
        depending on the input shape.
    """
    if att.ndim == 1:
        r = att[0]
        sr = sin(0.5 * r)
        cr = cos(0.5 * r)
        p = att[1]
        sp = sin(0.5 * p)
        cp = cos(0.5 * p)
        y = att[2]
        sy = sin(0.5 * y)
        cy = cos(0.5 * y)
        q = np.zeros(4)
        q[0] = cr * cp * cy + sr * sp * sy
        q[1] = sr * cp * cy - cr * sp * sy
        q[2] = cr * sp * cy + sr * cp * sy
        q[3] = cr * cp * sy - sr * sp * cy
        return q
    else:
        r = att[:, 0]
        sr = sin(0.5 * r)
        cr = cos(0.5 * r)
        p = att[:, 1]
        sp = sin(0.5 * p)
        cp = cos(0.5 * p)
        y = att[:, 2]
        sy = sin(0.5 * y)
        cy = cos(0.5 * y)
        N = att.shape[0]
        q = np.zeros((N, 4))
        q[:, 0] = cr * cp * cy + sr * sp * sy
        q[:, 1] = sr * cp * cy - cr * sp * sy
        q[:, 2] = cr * sp * cy + sr * cp * sy
        q[:, 3] = cr * cp * sy - sr * sp * cy
        return q


def quat2euler(q):
    """Convert quaternion to Euler angles.


    Parameters
    ----------
    q : np.ndarray
        Quaternion [q0, q1, q2, q3].
        This can be a 1D array of length 4 or a 2D array of shape (N, 4).
    Returns
    -------
    np.ndarray
        Array of Euler angles (roll, pitch, yaw) in radians.
        This can be a 1D array of length 3 or a 2D array of shape (N, 3),
        depending on the input shape.
    """
    if q.ndim == 1:
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        att = np.zeros(3)
        att[0] = np.arctan2(
            2.0 * (q0 * q1 + q2 * q3), (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        )
        att[1] = np.arcsin(2.0 * (q0 * q2 - q1 * q3))
        att[2] = np.arctan2(
            2.0 * (q0 * q1 + q2 * q3), (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        )
        return att
    else:
        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]
        N = q.shape[0]
        att = np.zeros((N, 3))
        att[:, 0] = np.arctan2(
            2.0 * (q0 * q1 + q2 * q3), (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        )
        att[:, 1] = np.arcsin(2.0 * (q0 * q2 - q1 * q3))
        att[:, 2] = np.arctan2(
            2.0 * (q0 * q1 + q2 * q3), (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        )
        return att


def attitude_dt(omega: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """
    Calculate the derivative of the attitude (roll, pitch, yaw) using angular rates in body frame
    and previously estimated roll and pitch angles.

    Parameters
    ----------
    omega : np.ndarray
        3-size array with angular rates in rad/s.
    roll : float
        Roll angle in radians.
    pitch : float
        Pitch angle in radians.

    Returns
    -------
    np.ndarray
        Time derivative of the attitude d(roll, pitch, yaw)/dt in rad/s.
    """
    sr = np.sin(roll)
    cr = np.cos(roll)
    xp = 1.0 / np.cos(pitch)  # secant
    tp = np.tan(pitch)
    mat = np.array(
        [
            [1.0, sr * tp, cr * tp],
            [0.0, cr, -sr],
            [0.0, sr * xp, cr * xp],
        ]
    )
    return mat.dot(omega)
