import numpy as np

from ..environment.constants import EARTH_GRAVITY_CONSTANT as g
from .ekf import ExtendedKalmanFilter


class AttitudeEKF:
    """
    x = [roll, pitch]
    u = [p, q, r, Va]
    z = [ax, ay, az]
    f(x, u) -> d(roll, pitch)/dt
    h(x, u) -> [accel_x, accel_y, accel_z]
    F_jacobian = ∂f/∂x
    H_jacobian = ∂h/∂x
    """

    def __init__(
        self, dt: float, accel_noise_std: float = 0.1, gyro_noise_std: float = 0.01
    ):
        self.ekf = ExtendedKalmanFilter(
            dt=dt,
            f=self.f,
            h=self.h,
            Q=np.diag([gyro_noise_std**2, gyro_noise_std**2]),
            R=np.diag([accel_noise_std**2, accel_noise_std**2, accel_noise_std**2]),
            P0=np.eye(2) * 0.1,
            x0=np.zeros((2, 1)),
            F_jac=self.F_jacobian,
            H_jac=self.H_jacobian,
        )

    def f(self, x, u):
        roll, pitch = x
        p, q, r, Va = u
        sr, cr, tp = np.sin(roll), np.cos(roll), np.tan(pitch)
        return np.array(
            [
                [p + q * sr * tp + r * cr * tp],
                [q * cr - r * sr],
            ]
        )

    def h(self, x, u):
        roll, pitch = x
        p, q, r, Va = u
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        return np.array(
            [
                [q * Va * sp + g * sp],
                [r * Va * cr - p * Va * sr - g * cp * sr],
                [-q * Va * cp - g * cp * cr],
            ]
        )

    def F_jacobian(self, x, u):
        roll, pitch = x
        p, q, r, Va = u
        sr, cr, tp = np.sin(roll), np.cos(roll), np.tan(pitch)
        sec2p = 1 / (np.cos(pitch) ** 2)
        return np.array(
            [
                [q * cr * tp - r * sr * tp, (q * sr - r * cr) * sec2p],
                [-q * sr - r * cr, 0],
            ]
        )

    def H_jacobian(self, x, u):
        roll, pitch = x
        p, q, r, Va = u
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        return np.array(
            [
                [0, q * Va * cp + g * cp],
                [-g * cp * cr, -r * Va * sp - p * Va * cp + g * sp * sr],
                [0, (q * Va + g * cr) * sp],
            ]
        )

    def predict(self, u):
        return self.ekf.predict(u)

    def update(self, z):
        return self.ekf.update(z, u)
