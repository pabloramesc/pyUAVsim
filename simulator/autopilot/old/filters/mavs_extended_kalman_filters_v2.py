"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.common.constants import EARTH_GRAVITY_CONSTANT as g
from scipy import stats

########################################################################################################################
##### ATTITUDE EKF V2 ##################################################################################################
########################################################################################################################
class AttitudeEKF_v2:
    def __init__(self, acc_noise: np.ndarray, input_noise: np.ndarray, att_error: np.ndarray, iter: int = 1, accel_threshold: float = 10.0):
        """
        Attitude Extended Kalman Filters from Small Unmanned Aircraft Book
        Estimates the roll and pitch angles using the readings from accelerometer, gyroscope and airspeed sensor

        Parameters
        ----------
        acc_noise : np.ndarray
            3-size array with acceleroemter standar deviation erros

        att_error : np.ndarray
            2-size array with roll and pitch estimated propagation errors

        iter : int, optional
            number of iterations to perform in the propagation step, by default 1

        gate_th : float, optional
            gate threshold to discard noisy measures, by default 7.8
        """
        self.x = np.zeros(2)  # state array (roll, pitch)
        self.u = np.zeros(4)  # control input array (p, q, r, Va)
        self.z = np.zeros(3)  # measurements array (ax, ay, az)

        self.F = np.zeros((2, 2))  # state transition matrix (2x2)
        self.G = np.zeros((2, 4))  # control matrix (2x4)
        self.H = np.zeros((3, 2))  # measurements matrix (3x2)

        self.Q = np.diag(att_error**2)  # process noise covariance matrix (estimation errors)
        self.Qu = np.diag(input_noise**2)  # p, q, r, Va input noise
        self.R = np.diag(acc_noise**2)  # measurements noise covariance matrix (accel noises)

        self.P = np.diag(att_error**2)  # error matrix (2x2)
        self.K = np.zeros((2, 3))  # kalman gain (2x3)
        self.I = np.eye(2)  # P sized identity matrix (2x2)

        self.N = iter  # times to run prediction
        self.gate_th =  accel_threshold # stats.chi2.isf(q=0.01, df=3)  # gate threshold

    def f_func(self, x: np.ndarray, u: np.ndarray):
        roll, pitch = x
        sr = np.sin(roll)
        cr = np.cos(roll)
        tp = np.tan(pitch)
        p, q, r, Va = u
        return np.array(
            [
                p + (q * sr + r * cr) * tp,
                q * cr - r * sr,
            ]
        )

    def h_func(self, x: np.ndarray, u: np.ndarray):
        roll, pitch = x
        sr = np.sin(roll)
        cr = np.cos(roll)
        sp = np.sin(pitch)
        cp = np.cos(pitch)
        p, q, r, Va = u
        return np.array(
            [
                (+q * Va + g) * sp,
                (+r * Va - g * sr) * cp - p * Va * sp,
                (-q * Va - g * cr) * cp,
            ]
        )

    def F_matrix(self, x: np.ndarray, u: np.ndarray):
        roll, pitch = x
        sr = np.sin(roll)
        cr = np.cos(roll)
        tp = np.tan(pitch)
        p, q, r, Va = u
        return np.array(
            [
                [(q * cr - r * sr) * tp, (q * sr + r * cr) * (1 + tp**2)],
                [-q * sr - r * cr, 0],
            ]
        )

    def H_matrix(self, x: np.ndarray, u: np.ndarray):
        roll, pitch = x
        sr = np.sin(roll)
        cr = np.cos(roll)
        sp = np.sin(pitch)
        cp = np.cos(pitch)
        p, q, r, Va = u
        return np.array(
            [
                [0, (q * Va + g) * cp],
                [-g * cr * cp, (-r * Va + g * sr) * sp - p * Va * cp],
                [+g * sr * cp, (+q * Va + g * cr) * sp],
            ]
        )

    def G_matrix(self, x: np.ndarray):
        roll, pitch = x
        sr = np.sin(roll)
        cr = np.cos(roll)
        tp = np.tan(pitch)
        return np.array(
            [
                [1, sr * tp, cr * tp, 0],
                [0, cr, -sr, 0],
            ]
        )

    def initialize(self, x: np.ndarray):
        """
        Initialize the state of the filter

        Parameters
        ----------
        x : np.ndarray
            2-size array with roll and pitch angle in rad
        """
        self.x = x  # roll, pitch

    def prediction(self, u: np.ndarray, dt: float):
        """
        Prediction step

        Parameters
        ----------
        u : np.ndarray
            4-size array with input values from gyroscope and airspeed sensor (p, q, r, Va) in rad/s and m/s

        dt : float
            update period

        Returns
        -------
        _type_
            _description_
        """
        self.u = u
        Tp = dt / self.N
        for kk in range(self.N):
            self.F = self.F_matrix(self.x, self.u)
            self.G = self.G_matrix(self.x)
            ### propagate estimated state
            self.x = self.x + self.f_func(self.x, self.u) * Tp
            ### propagate estimation error matrix
            Ad = self.I + self.F * Tp + self.F @ self.F * Tp**2
            Qt = self.Q + self.G @ self.Qu @ self.G.T
            self.P = Ad @ self.P @ Ad.T + Qt * Tp**2
        return self.x

    def correction(self, z: np.ndarray, dt: float):
        """
        Correction step

        Parameters
        ----------
        z : np.ndarray
            3-size array with accelerometer measures in m/s^2

        dt : float
            update period

        Returns
        -------
        _type_
            _description_
        """
        self.z = z
        try:
            ### compute kalman gain
            self.H = self.H_matrix(self.x, self.u)
            S_inv = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            ### discard measures out of gate
            h = self.h_func(self.x, self.u)
            resid = z - h
            if resid.T @ S_inv @ resid < self.gate_th:
                ### compute kalman gain
                self.K = self.P @ self.H.T @ S_inv
                ### correct estimated state
                self.x = self.x + self.K @ resid
                ### correct estimation error matrix
                I_KH = self.I - self.K @ self.H
                # self.P = I_KH @ self.P
                self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        except np.linalg.LinAlgError:
            print("Position-EKF: S_inv matrix goes singular!")
        return self.x


########################################################################################################################
##### POSITION EKF #####################################################################################################
########################################################################################################################
class PositionEKF_v2:
    def __init__(self, gps_noise: np.ndarray, wind_noise: np.ndarray, pos_error: np.ndarray, iter: int = 1):
        """
        Position Extended Kalman Filter from Small Unmanned Aircraft Book
        Estimates the horizontal position, course, groundspeed, yaw angle and wind using

        Parameters
        ----------
        gps_error : np.ndarray
            _description_
        pos_error : np.ndarray
            _description_
        iter : int, optional
            _description_, by default 10
        gate_th : float, optional
            _description_, by default 200.0
        """
        self.x = np.zeros(7)  # state array (pn, pe, Vg, course, wn, we, yaw)
        self.u = np.zeros(5)  # control input array (Va, q, r, roll, pitch)
        self.z = np.zeros(
            6
        )  # measurements array (pn, pe, Vg, course, wn, we) <-- wind (wn, we) are pseudo measurements

        # self.F = np.zeros((7, 7))  # state transition matrix (7x7)
        # self.G = np.zeros((7, 5))  # control matrix (7x5)
        # self.H = np.zeros((6, 7))  # measurements matrix (6x7)

        self.Q = np.diag(pos_error**2)  # process noise covariance matrix (estimation errors)
        self.R_gps = np.diag(gps_noise**2)  # measurements noise covariance matrix (accel noises)
        self.R_wind = np.diag(wind_noise**2)  # measurements noise covariance matrix (accel noises)

        self.P = np.diag(pos_error**2)  # error matrix (7x7)
        # self.K = np.zeros((7, 6))  # kalman gain (7x6)
        self.I = np.eye(7)  # P sized identity matrix (7x7)

        self.N = iter  # times to run prediction
        self.gps_threshold = 1e6  # stats.chi2.isf(q=0.01, df=4)
        self.wind_threshold = stats.chi2.isf(q=0.01, df=2)

    def f_func(self, x: np.ndarray, u: np.ndarray):
        _, _, Vg, course, wn, we, yaw = x
        sy = np.sin(yaw)
        cy = np.cos(yaw)
        Va, q, r, roll, pitch = u
        yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch)
        Vg_inv = 1.0 / Vg
        return np.array(
            [
                Vg * np.cos(course),  # vn
                Vg * np.sin(course),  # ve
                ((Va * cy + wn) * (-Va * yaw_dot * sy) + (Va * sy + we) * (Va * yaw_dot * cy)) * Vg_inv,  # acc_g
                g * Vg_inv * np.tan(roll),  # course_dt
                0,  # wn_dt
                0,  # we_dt
                yaw_dot,  # d(yaw)/dt
            ]
        )

    def F_matrix(self, x: np.ndarray, u: np.ndarray):
        _, _, Vg, course, wn, we, yaw = x
        sx = np.sin(course)
        cx = np.cos(course)
        sy = np.sin(yaw)
        cy = np.cos(yaw)
        Va, q, r, roll, pitch = u
        yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch)
        Vg_inv = 1.0 / Vg
        Vg_dot = Va * yaw_dot * (we * cy - wn * sy) * Vg_inv
        tr = np.tan(roll)
        dVg_dyaw = -yaw_dot * Va * (wn * cy + we * sy) * Vg_inv
        return np.array(
            [
                [0, 0, cx, -Vg * sx, 0, 0, 0],
                [0, 0, sx, +Vg * cx, 0, 0, 0],
                [0, 0, -Vg_dot * Vg_inv, 0, -yaw_dot * Va * sy * Vg_inv, +yaw_dot * Va * cy * Vg_inv, dVg_dyaw],
                [0, 0, -g * Vg_inv**2 * tr, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )

    def h_gps(self, x: np.ndarray, u: np.ndarray):
        pn, pe, Vg, course, _, _, _ = x
        # Va, _, _, _, _ = u
        return np.array(
            [
                pn,
                pe,
                Vg,
                course,
            ]
        )
        
    def h_wind(self, x: np.ndarray, u: np.ndarray):
        _, _, Vg, course, wn, we, yaw = x
        Va, _, _, _, _ = u
        return np.array(
            [
                Va * np.cos(yaw) + wn - Vg * np.cos(course),
                Va * np.sin(yaw) + we - Vg * np.sin(course),
            ]
        )

    def H_matrix_gps(self, x: np.ndarray, u: np.ndarray):
        # _, _, Vg, course, _, _, yaw = x
        # sx = np.sin(course)
        # cx = np.cos(course)
        # Va, _, _, _, _ = u
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        
    def H_matrix_wind(self, x: np.ndarray, u: np.ndarray):
        _, _, Vg, course, _, _, yaw = x
        sx = np.sin(course)
        cx = np.cos(course)
        Va, _, _, _, _ = u
        return np.array(
            [
                [0, 0, -cx, +Vg * sx, 1, 0, -Va * np.sin(yaw)],
                [0, 0, -sx, -Vg * cx, 0, 1, +Va * np.cos(yaw)],
            ]
        )

    def initialize(self, x: np.ndarray):
        self.x = x  # pn, pe, Vg, course, wn, we, yaw

    def prediction(self, u: np.ndarray, dt: float):
        self.u = u
        Tp = dt / self.N
        for kk in range(self.N):
            self.F = self.F_matrix(self.x, self.u)
            ### propagate estimated state
            self.x = self.x + self.f_func(self.x, self.u) * Tp
            ### propagate estimation error matrix
            Ad = self.I + self.F * Tp + self.F @ self.F * Tp**2
            self.P = Ad @ self.P @ Ad.T + self.Q * Tp**2
        return self.x
    
    def correction_gps(self, z_gps: np.ndarray, dt: float):
        try:
            ### compute kalman gain
            H_gps = self.H_matrix_gps(self.x, self.u)
            S_inv = np.linalg.inv(H_gps @ self.P @ H_gps.T + self.R_gps)
            ### correct estimated state
            h_gps = self.h_gps(self.x, self.u)
            resid = z_gps - h_gps # pn, pe, Vg, course
            if resid.T @ S_inv @ resid < self.gps_threshold:
                ### compute kalman gain
                K_gps = self.P @ H_gps.T @ S_inv
                ### correct estimated state
                self.x = self.x + K_gps @ resid
                ### correct estimation error matrix
                I_KH = self.I - K_gps @ H_gps
                # self.P = I_KH @ self.P
                self.P = I_KH @ self.P @ I_KH.T + K_gps @ self.R_gps @ K_gps.T
        except np.linalg.LinAlgError:
            print("Position-EKF: S_inv matrix goes singular!")
        return self.x


    def correction_wind(self, z_wind: np.ndarray, dt: float):
        try:
            ### compute kalman gain
            H_wind = self.H_matrix_wind(self.x, self.u)
            S_inv = np.linalg.inv(H_wind @ self.P @ H_wind.T + self.R_wind)
            ### correct estimated state
            h_wind = self.h_wind(self.x, self.u)
            resid = z_wind - h_wind  # wn, we
            if resid.T @ S_inv @ resid < self.wind_threshold:
                ### compute kalman gain
                K_wind = self.P @ H_wind.T @ S_inv
                ### correct estimated state
                self.x = self.x + K_wind @ resid
                ### correct estimation error matrix
                I_KH = self.I - K_wind @ H_wind
                # self.P = I_KH @ self.P
                self.P = I_KH @ self.P @ I_KH.T + K_wind @ self.R_wind @ K_wind.T
        except np.linalg.LinAlgError:
            print("Position-EKF: S_inv matrix goes singular!")
        return self.x
