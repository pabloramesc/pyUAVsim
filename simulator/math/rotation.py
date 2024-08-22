"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy import sin, cos


def rot_matrix_axis(axis: np.ndarray = np.zeros(3), t: float = 0.0) -> np.ndarray:
    """
    Compute transformation matrix for rotation around an axis.

    Parameters
    ----------
    axis : np.ndarray, optional
        3-size array with rotation axis components (ux, uy, uz), by default np.zeros(3)
    t : float, optional
        rotation angle in rad, by default 0.0

    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    ux, uy, uz = axis / np.linalg.norm(axis)
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
    with the aircraft attitude expressed as euler angles using the Z-Y-X rotation sequence.

    Parameters
    ----------
    euler : np.ndarray, optional
        3-size array with euler angles [roll, pitch, yaw] in rad, by default np.zeros(3)

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
    R_vb = np.zeros((3, 3))
    R_vb[0, 0] = cp * cy
    R_vb[0, 1] = cp * sy
    R_vb[0, 2] = -sp
    R_vb[1, 0] = sr * sp * cy - cr * sy
    R_vb[1, 1] = sr * sp * sy + cr * cy
    R_vb[1, 2] = sr * cp
    R_vb[2, 0] = cr * sp * cy + sr * sy
    R_vb[2, 1] = cr * sp * sy - sr * cy
    R_vb[2, 2] = cr * cp
    return R_vb


def rot_matrix_quat(quat: np.ndarray = np.array([1, 0, 0, 0])) -> np.ndarray:
    """
    Compute transformation matrix from vehicle frame to body frame (R^b_v)
    with the aircraft attitude expressed as quaternions.

    Parameters
    ----------
    quat : np.ndarray, optional
        4-size array with aircraft's orientation quaternions [q0, q1, q2, q3], by default np.array([1, 0, 0, 0])

    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    R_vb = np.zeros((3, 3))
    R_vb[0, 0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    R_vb[0, 1] = 2.0 * (q1 * q2 + q0 * q3)
    R_vb[0, 2] = 2.0 * (q1 * q3 - q0 * q2)
    R_vb[1, 0] = 2.0 * (q1 * q2 - q0 * q3)
    R_vb[1, 1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    R_vb[1, 2] = 2.0 * (q2 * q3 + q0 * q1)
    R_vb[2, 0] = 2.0 * (q1 * q3 + q0 * q2)
    R_vb[2, 1] = 2.0 * (q2 * q3 - q0 * q1)
    R_vb[2, 2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    return R_vb


def rot_matrix_wind(alpha: float, beta: float) -> np.ndarray:
    """Compute transformation matrix from wind frame to body frame (R^b_w).

    Parameters
    ----------
    alpha : float
        angle of attack in rads
    beta : float
        side-slip angle in rads

    Returns
    -------
    np.ndarray
        3x3 transformation matrix
    """
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.sin(beta)
    R_wb = np.zeros((3, 3))
    R_wb[0, 0] = cb * ca
    R_wb[0, 1] = -sb * ca
    R_wb[0, 2] = -sa
    R_wb[1, 0] = sb
    R_wb[1, 1] = cb
    R_wb[1, 2] = 0.0
    R_wb[2, 0] = cb * sa
    R_wb[2, 1] = -sb * sa
    R_wb[2, 2] = ca
    return R_wb


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


def ned2xyz(ned: np.ndarray = np.zeros(3)) -> np.ndarray:
    """
    Convert coordinates from the NED (North-East-Down) frame to the XYZ frame.

    Parameters
    ----------
    ned_coords : np.ndarray, optional
        Coordinates in the NED frame. This can be a 1D array of shape (3,) or a 2D array of shape (N, 3).
        Default is np.zeros(3).

    Returns
    -------
    np.ndarray
        Coordinates in the XYZ frame, with the same shape as the input.
    """
    ned = np.atleast_2d(ned)  # Ensure 2D for uniform handling

    xyz = np.zeros((ned.shape[0], 3))
    xyz[:, 0] = ned[:, 1]
    xyz[:, 1] = ned[:, 0]
    xyz[:, 2] = -ned[:, 2]

    return np.squeeze(xyz)  # Maintain shape consistency with input


def euler2quat(att: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion representation.

    Parameters
    ----------
    att : np.ndarray
        Array of Euler angles (roll, pitch, yaw) in radians. Input can be a 1D array of shape (3,)
        or a 2D array of shape (N, 3).

    Returns
    -------
    np.ndarray
        Quaternion(s) as a 1D array of shape (4,) for a single set of Euler angles, or a 2D array
        of shape (N, 4) for multiple sets.
    """
    att = np.atleast_2d(att)  # Ensure 2D for consistent processing

    r = att[:, 0]
    sr = sin(0.5 * r)
    cr = cos(0.5 * r)
    p = att[:, 1]
    sp = sin(0.5 * p)
    cp = cos(0.5 * p)
    y = att[:, 2]
    sy = sin(0.5 * y)
    cy = cos(0.5 * y)

    q = np.zeros((att.shape[0], 4))
    q[:, 0] = cr * cp * cy + sr * sp * sy
    q[:, 1] = sr * cp * cy - cr * sp * sy
    q[:, 2] = cr * sp * cy + sr * cp * sy
    q[:, 3] = cr * cp * sy - sr * sp * cy

    return np.squeeze(q)  # Adjust shape based on input


def quat2euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to Euler angles.

    Parameters
    ----------
    q : np.ndarray
        Input quaternion(s) as a 1D array of shape (4,) or a 2D array of shape (N, 4).

    Returns
    -------
    np.ndarray
        Euler angles (roll, pitch, yaw) in radians. Output shape is (3,) for a single
        quaternion or (N, 3) for multiple.

    Notes
    -----
    - Angles are computed using the ZYX sequence: yaw (z-axis), pitch (y-axis), roll (x-axis).
    """
    q = np.atleast_2d(q)  # Ensure q is at least 2D (N, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    att = np.zeros((q.shape[0], 3))
    att[:, 0] = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3), (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    )
    att[:, 1] = np.arcsin(2.0 * (q0 * q2 - q1 * q3))
    att[:, 2] = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2), (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    )

    return np.squeeze(att)  # Remove axes of length 1 (for single quaternion input)


def euler_kinematics(euler: np.ndarray) -> np.ndarray:
    """
    Calculate the transformation matrix to convert angular rates [p, q, r]
    to Euler angle derivatives d(roll, pitch, yaw)/dt.

    Parameters
    ----------
    euler : np.ndarray
        3-size array with Euler angles [roll, pitch, yaw] in radians.

    Returns
    -------
    np.ndarray
        3x3 transformation matrix for Euler angle kinematics.

    Notes
    -----
    Formula: `euler_dot = R_euler * omega` where
    - `euler_dot` is the time derivative of Euler angles [roll, pitch, yaw].
    - `R_euler` is the transformation matrix that maps angular rates [p, q, r] to
      the derivatives of Euler angles.
    - `omega` is a 3-size array with angular rates [p, q, r] in rad/s.

    The transformation matrix `R_euler` is given by:
    ```
    R_euler = [[1.0, sin(roll)*tan(pitch), cos(roll)*tan(pitch)],
               [0.0, cos(roll), -sin(roll)],
               [0.0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]]
    ```
    where `roll` and `pitch` are the current Euler angles.

    This matrix is used to transform the angular rate vector [p, q, r] into the
    rate of change of Euler angles.
    """
    sr = np.sin(euler[0])
    cr = np.cos(euler[0])
    xp = 1.0 / np.cos(euler[1])  # secant
    tp = np.tan(euler[1])
    R_euler = np.array(
        [
            [1.0, sr * tp, cr * tp],
            [0.0, cr, -sr],
            [0.0, sr * xp, cr * xp],
        ]
    )
    return R_euler


def quaternion_kinematics(omega: np.ndarray) -> np.ndarray:
    """
    Calculate the transformation matrix to convert angular rates [p, q, r]
    to quaternion derivative d(q0, q1, q2, q3)/dt.

    Parameters
    ----------
    omega : np.ndarray
        3-size array with angular rates [p, q, r] in rad/s.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix for quaternion kinematics.

    Notes
    -----
    Formula: `q_dot = 1/2 * Omega * q` where
    - `q_dot` is the time derivative of the quaternion [q0, q1, q2, q3].
    - `Omega` is a 4x4 skew-symmetric matrix constructed from the angular rates [p, q, r].
    - `q` is the current quaternion [q0, q1, q2, q3].

    The transformation matrix `Omega` is given by:
    ```
    Omega = [[  0,  -p,  -q,  -r],
             [  p,   0,   r,  -q],
             [  q,  -r,   0,   p],
             [  r,   q,  -p,   0]]
    ```
    where `p`, `q`, and `r` are the angular rates.

    This matrix is used to transform the quaternion vector into its time derivative,
    considering the current angular rates.
    """
    wx = omega[0]
    wy = omega[1]
    wz = omega[2]
    Omega = np.array(
        [
            [0.0, -wx, -wy, -wz],
            [+wx, 0.0, +wz, -wy],
            [+wy, -wz, 0.0, +wx],
            [+wz, +wy, -wx, 0.0],
        ]
    )
    return Omega
