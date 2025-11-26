import numpy as np


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
    Formula: `euler_dot = R_euler @ omega` where
    - `euler_dot` is the time derivative of Euler angles [roll, pitch, yaw].
    - `R_euler` is the transformation matrix that maps angular rates [p, q, r]
    to the derivatives of Euler angles.
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
    Formula: `q_dot = 1/2 * Omega @ q` where
    - `q_dot` is the time derivative of the quaternion [q0, q1, q2, q3].
    - `Omega` is a 4x4 skew-symmetric matrix constructed from the angular rates
    [p, q, r].
    - `q` is the current quaternion [q0, q1, q2, q3].

    The transformation matrix `Omega` is given by:
    ```
    Omega = [[  0,  -p,  -q,  -r],
             [  p,   0,   r,  -q],
             [  q,  -r,   0,   p],
             [  r,   q,  -p,   0]]
    ```
    where `p`, `q`, and `r` are the angular rates.

    This matrix is used to transform the quaternion vector into its time
    derivative, considering the current angular rates.
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
