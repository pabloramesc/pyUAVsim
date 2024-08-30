"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np
from simulator.autopilot.filters.filter import Filter, FilterConfig, FilterEstimations
from simulator.autopilot.filters.simple_kalman_filters import Kalman_1D_P_P_V
from simulator.autopilot.filters.low_pass_filter import LowPassFilter
from simulator.autopilot.filters.mavs_extended_kalman_filters import AttitudeEKF, PositionEKF
from simulator.common.constants import (
    DEFAULT_HOME_COORDS,
    DEG2RAD,
    EARTH_GRAVITY_CONSTANT,
    EARTH_GRAVITY_VECTOR,
    GEO_DEG2M,
)
from simulator.sensors.sensors_manager import SensorsManager
from simulator.utils.rotation import attitude_dt, clip_angle_pi, rot_matrix_zyx


@dataclass
class StateEstimationFilterConfig(FilterConfig):
    gyroscope_lpf_gain = 0.5
    accelerometer_lpf_gain = 0.5

    altitude_lpf_gain = 0.98
    airspeed_lpf_gain = 0.95

    aoa_filter_tau: float = 100.0
    aoa_filter_alpha0: float = 0.02

    ekf_iter: int = 1

    acc_noise: np.ndarray = np.array([0.004, 0.004, 0.004])  # (ax, ay, az)
    att_error: np.ndarray = np.array([0.002, 0.002])  # (roll, pitch)
    att_gate_th: float = 65.0

    gps_noise: np.ndarray = np.array([0.4, 0.4, 0.05, 0.0025, 0.05, 0.05])  # (pn, pe, Vg, course, wn, we)
    #[ 0.08324526 -0.20886542 -1.11616987 -1.28943098 -1.01859397] lllllll
    # pos_error: np.ndarray = np.array([2.7584, 2.7584, 0.3235, 0.056, 0.0752, 0.0752, 0.0991])
    pos_error: np.ndarray = np.array([2.0, 2.0, 0.30, 0.0025, 0.05, 0.05, 0.0025])  # (pn, pe, Vg, course, wn, we, yaw)
    # pos_error: np.ndarray = np.array([0.25, 0.25, 0.03, 0.002, 0.3, 0.3, 0.076])
    # pos_error: np.ndarray = np.array([0.488, 0.488, 0.021, 0.00675, 0.10, 0.10, 0.012])
    # pos_error: np.ndarray = np.array([0.488, 0.488, 0.083, 0.00675, 0.31, 0.31, 0.012])  # (pn, pe, Vg, course, wn, we, yaw)
    # pos_error: np.ndarray = np.array([0.52, 0.52, 0.021, 0.0086, 0.10, 0.10, 0.011])  # (pn, pe, Vg, course, wn, we, yaw)
    pos_gate_th: float = 2000

    baro_noise: np.ndarray = np.array([0.8342])  # baro-altimeter measures error (m)
    alt_error: np.ndarray = np.array([0.2])  # state propagation error (m)
    alt_gate_th: float = 1.0

    diff_noise: np.ndarray = np.array([1.8070])  # airspeed sensor measures error (m/s)
    air_error: np.ndarray = np.array([0.43])  # state propagation error (m/s)
    air_gate_th: float = 0.74


class StateEstimationFilter(Filter):
    def __init__(
        self, config: StateEstimationFilterConfig, sensors_manager: SensorsManager, home_coords=DEFAULT_HOME_COORDS
    ) -> None:

        ##### SENSORS FILTERS #####
        self.gyroscope_lpf = LowPassFilter(config.gyroscope_lpf_gain, np.zeros(3))
        self.accelerometer_lpf = LowPassFilter(config.accelerometer_lpf_gain, np.array([0.0, 0.0, 1.0]))
        self.altitude_lpf = LowPassFilter(config.altitude_lpf_gain, 0.0)
        self.airspeed_lpf = LowPassFilter(config.airspeed_lpf_gain, 0.0)
        self.hspeed_lpf = LowPassFilter(0.8, np.zeros(2))

        ##### ATTITUDE EKF #####
        self.attitude_EKF = AttitudeEKF(config.acc_noise, config.att_error, config.ekf_iter, config.att_gate_th)
        self.last_accel_read_t = 0.0

        ##### POSITION EKF #####
        self.position_EKF = PositionEKF(config.gps_noise, config.pos_error, config.ekf_iter, config.pos_gate_th)
        self.last_gps_read_t = 0.0

        ##### ALTITUDE KF #####
        self.altitude_KF = Kalman_1D_P_P_V(0.01, config.baro_noise, config.alt_error, config.alt_gate_th)
        self.last_baro_read_t = 0.0

        ##### AIRSPEED KF #####
        self.airspeed_KF = Kalman_1D_P_P_V(0.01, config.diff_noise, config.air_error, config.air_gate_th)
        self.last_diff_read_t = 0.0

        ##### AOA ESTIMATION FILTER #####
        self.alpha = 0.0
        self.aoa_tau = config.aoa_filter_tau  # m/s * rad
        self.aoa_alpha0 = config.aoa_filter_alpha0  # rad

        super().__init__(config, sensors_manager, home_coords)

    def set_baro_reference(self, home_pressure: float) -> None:
        return super().set_baro_reference(home_pressure)

    def set_home_coords(self, home_coords: tuple) -> None:
        return super().set_home_coords(home_coords)

    def initialize_state(self, state0: np.ndarray = np.zeros(6)) -> None:
        pn, pe, pd, vn, ve, vd = state0
        Vg = np.sqrt(vn**2 + ve**2)
        course = np.arctan2(ve, vn)
        self.airspeed_lpf.reset(vn)
        self.altitude_lpf.reset(-pd)
        self.altitude_KF.initialize(np.array([-pd]))
        self.airspeed_KF.initialize(np.array([vn]))
        self.attitude_EKF.initialize(np.zeros(2))
        self.position_EKF.initialize(np.array([pn, pe, Vg, course, 0, 0, course]))
        self.hspeed_lpf.reset(np.array([vn, ve]))

    def estimate(self, t: float, dt: float = 0.01) -> dict:
        ### read accelerometer
        acc_raw = self.accelerometer.read(t) * EARTH_GRAVITY_CONSTANT  # Gs to m/s^2
        acc_lpf = self.accelerometer_lpf.update(acc_raw)
        ax, ay, az = acc_raw

        ### read gyroscope
        gyr_raw = self.gyroscope.read(t) * DEG2RAD  # dps to rad/s
        gyr_lpf = self.gyroscope_lpf.update(gyr_raw)
        wx, wy, wz = gyr_raw

        ### read airspeed sensor
        dynamic_pressure = self.airspeed.read(t)
        Va_raw = np.sqrt((2.0 / self.home_density) * dynamic_pressure)[0]
        Va_lpf = self.airspeed_lpf.update(Va_raw)

        ### read baro-altimeter
        static_pressure = self.barometer.read(t)
        alt_raw = ((self.home_pressure - static_pressure) / (self.home_density * EARTH_GRAVITY_CONSTANT))[0]
        alt_lpf = self.altitude_lpf.update(alt_raw)

        ##### ATTITUDE EKF (roll, pitch) #####
        # propagation
        u = np.array([wx, wy, wz, Va_lpf])
        roll, pitch = self.attitude_EKF.prediction(u, dt)
        if self.accelerometer.last_update_time + self.accelerometer.reading_delay > self.last_accel_read_t:
            self.last_accel_read_t = t
            # correction
            z = np.array([ax, ay, az])
            roll, pitch = self.attitude_EKF.correction(z, dt)

        ##### HORIZONTAL POSITION EKF (pn, pe, Vg, course, wn, we, yaw) #####
        # propagation
        u = np.array([Va_lpf, wy, wz, roll, pitch])
        pn, pe, Vg, course, wn, we, yaw = self.position_EKF.prediction(u, dt)
        if self.gps.last_update_time + self.gps.reading_delay > self.last_gps_read_t:
            self.last_gps_read_t = t
            # read gps
            lat, long, _, Vg, course_deg = self.gps.read(t)
            course = clip_angle_pi(course_deg * DEG2RAD)
            pn = (lat - self.home_coords[0]) * GEO_DEG2M
            pe = (long - self.home_coords[1]) * GEO_DEG2M
            wn = 0.0  # vn - Va * np.cos(yaw)
            we = 0.0  # ve - Va * np.sin(yaw)
            # correction
            z = np.array([pn, pe, Vg, course, wn, we])
            pn, pe, Vg, course, wn, we, yaw = self.position_EKF.correction(z, dt)

        ### rotation matrices calculation
        R_vb = rot_matrix_zyx(np.array([roll, pitch, yaw]))  # from vehicle frame to body frame
        R_bv = R_vb.T  # from body frame to vehicle frame

        ##### ANGLE OF ATTACK ESTIMATION FILTER #####
        att_dt = attitude_dt(gyr_raw, roll, pitch)  # attitude time derivatives from gyro
        pitch_dt = att_dt[1]
        alpha_dt = -self.aoa_tau / Va_lpf * self.alpha + pitch_dt + self.aoa_alpha0
        self.alpha = self.alpha + alpha_dt * dt

        ### body frame to wind frame rotation matrix
        sa = np.sin(self.alpha)
        ca = np.cos(self.alpha)
        R_bw = np.array(
            [
                [ca, 0, sa],
                [0, 1, 0],
                [-sa, 0, ca],
            ]
        )

        ### NED velocity estimation
        ur = Va_lpf * np.cos(self.alpha)
        vr = 0
        wr = Va_lpf * np.sin(self.alpha)
        airspeed_xyz = np.array([ur, vr, wr])
        airspeed_ned = R_bv.dot(airspeed_xyz)
        wind_ned = np.zeros(3)  # np.array([wn, we, 0])
        vel_ned = airspeed_ned + wind_ned
        _, _, vd = vel_ned
        # vn = Vg * np.cos(course)
        # ve = Vg * np.sin(course)
        vn, ve = self.hspeed_lpf.update(Vg * np.array([np.cos(course), np.sin(course)]))
        # Vg = np.sqrt(vn**2 + ve**2)
        # course = np.arctan2(ve, vn)

        ### non-gravity accelerations estimation
        acc_ned = R_bv.dot(np.array([ax, ay, az])) + EARTH_GRAVITY_VECTOR
        acc_xyz = acc_raw + R_vb.dot(EARTH_GRAVITY_VECTOR)
        acc_wind = R_bw.dot(acc_xyz)

        ### airspeed change rate
        Va_dot = acc_wind[0]

        ##### AIRSPEED SIMPLE KALMAN FILTER #####
        (Va_skf,) = self.airspeed_KF.prediction(np.array([Va_dot]), dt)
        if self.airspeed.last_update_time + self.airspeed.reading_delay > self.last_diff_read_t:
            self.last_diff_read_t = t
            # correction
            (Va_skf,) = self.airspeed_KF.correction(np.array([Va_raw]))

        ##### ALTITUDE KALMAN FILTER #####
        # propagation
        (alt_skf,) = self.altitude_KF.prediction(np.array([-vd]), dt)
        if self.barometer.last_update_time + self.barometer.reading_delay > self.last_baro_read_t:
            self.last_baro_read_t = t
            # correction
            (alt_skf,) = self.altitude_KF.correction(np.array([alt_raw]))

        estimation = FilterEstimations(
            position_ned=np.array([pn, pe, -alt_skf]),
            attitude=np.array([roll, pitch, yaw]),
            angular_rate=gyr_raw,
            groundspeed=np.sqrt(vn**2 + ve**2),
            course=np.arctan2(ve, vn),
            airspeed=Va_skf,
            velocity_ned=np.array([vn, ve, vd]),
            accel_ned=acc_ned,
            wind=np.array([wn, we, 0.0]),
            angle_of_attack=self.alpha,
            side_slip_angle=0.0,
        )
        self.estimations_history.update(estimation)
        return estimation
