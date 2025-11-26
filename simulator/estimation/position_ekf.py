import numpy as np

from .ekf import ExtendedKalmanFilter
from ..environment.constants import EARTH_GRAVITY_CONSTANT as g


class PositionEKF:
    """
    x = [pn, pe, Vg, course, wn, we, yaw]
    u = [Va, q, r, roll, pitch]
    f(x, u) -> dx/dt
    h(x, u) -> [pn, pe, Vg, course, wn, we]
    """

    def __init__(self, dt: float):
        self.ekf = ExtendedKalmanFilter(
            dt=dt,
            f=self.f,
            h=self.h,
            Q=np.diag([0.1, 0.1, 1.0, 0.5, 0.1, 0.1, 0.1]),
            R=np.diag([5.0, 5.0, 1.0, 0.5, 0.5, 0.5]),
            P0=np.eye(7) * 10.0,
            x0=np.zeros((7, 1)),
            F_jac=self.F_jacobian,
            H_jac=self.H_jacobian,
        )
        pass

    def f(self, x, u):
        pn, pe, Vg, course, wn, we, yaw = x.flatten()
        Va, q, r, roll, pitch = u.flatten()

        sx, cx = np.sin(course), np.cos(course)
        sr, cr = np.sin(roll), np.cos(roll)
        cp = np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)

        yaw_dot = q * sr / cp + r * cr / cp
        Vg_dot = (
            (Va * cy + wn) * (-Va * yaw_dot * sy)
            + (Va * sy + we) * (+Va * yaw_dot * cx)
        ) / Vg

        return np.array(
            [
                [Vg * cx + wn],  # pn_dot
                [Vg * sx + we],  # pe_dot
                [Vg_dot],  # Vg_dot
                [g / Vg * np.tan(roll) * np.cos(course - yaw)],  # course_dot
                [0],  # wn_dot
                [0],  # we_dot
                [yaw_dot],  # yaw_dot
            ]
        )

    def h(self, x, u):
        pn, pe, Vg, course, wn, we, yaw = x.flatten()
        Va, q, r, roll, pitch = u.flatten()

        return np.array(
            [
                [pn],  # y_gps_n
                [pe],  # y_gps_e
                [Vg],  # y_gps_Vg
                [course],  # y_gps_course
                [Va * np.cos(yaw) + wn - Vg * np.cos(course)],  # y_wind_n
                [Va * np.sin(yaw) + we - Vg * np.sin(course)],  # y_wind_e
            ]
        )

    def F_jacobian(self, x, u):
        pn, pe, Vg, course, wn, we, yaw = x.flatten()
        Va, q, r, roll, pitch = u.flatten()

        sx, cx = np.sin(course), np.cos(course)
        sy, cy = np.sin(yaw), np.cos(yaw)
        sr, cr, tr = np.sin(roll), np.cos(roll), np.tan(roll)
        cp = np.cos(pitch)

        yaw_dot = q * sr / cp + r * cr / cp
        Vg_dot = (
            (Va * cy + wn) * (-Va * yaw_dot * sy)
            + (Va * sy + we) * (+Va * yaw_dot * cx)
        ) / Vg
        dVg_dot_dyaw = -yaw_dot * Va * (wn * cy + we * sy) / Vg
        dcourse_dot_dVg = -g / Vg**2 * tr * np.sin(course - yaw)
        dcourse_dot_dcourse = -g / Vg * tr * np.sin(yaw - course)
        dcourse_dot_dyaw = g / Vg * tr * np.sin(course - yaw)

        return np.array(
            [
                [0, 0, cx, -Vg * sx, 0, 0, 0],
                [0, 0, sx, +Vg * cx, 0, 0, 0],
                [
                    0,
                    0,
                    -Vg_dot / Vg,
                    0,
                    -yaw_dot * Va * sy,
                    +yaw_dot * Va * cx,
                    dVg_dot_dyaw,
                ],
                [0, 0, dcourse_dot_dVg, dcourse_dot_dcourse, 0, 0, dcourse_dot_dyaw],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )

    def H_jacobian(self, x, u):
        pn, pe, Vg, course, wn, we, yaw = x.flatten()
        Va, q, r, roll, pitch = u.flatten()

        sx, cx = np.sin(course), np.cos(course)
        sy, cy = np.sin(yaw), np.cos(yaw)

        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, -cx, +Vg * sx, 1, 0, -Va * sy],
                [0, 0, -sx, -Vg * cx, 0, 1, +Va * cy],
            ]
        )
