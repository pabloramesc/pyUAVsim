"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.math.rotation import (
    rot_matrix_zyx,
    rot_matrix_wind,
    rot_matrix_quat,
    euler2quat,
    quat2euler,
)


class AircraftState:
    def __init__(
        self,
        x0: np.ndarray = None,
        wind0: np.ndarray = np.zeros(3),
        use_quat: bool = False,
    ) -> None:
        """Initialize the AircraftState class.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state vector.
            The structure of this array depends on the orientation representation selected by `use_quat`:
            If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
            If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
            By default None
        wind0 : np.ndarray, optional
            Initial wind vector (3-size array: wx, wy, wz in m/s),
            by default np.zeros(3)
        use_quat : bool, optional
            Flag to indicate whether to use quaternions for representing orientation.
            If True, quaternions will be used; otherwise, Euler angles will be used.
            By default False

        Notes
        -----
        The aircraft's state array `x` elements:
        - pn: Position North (meters)
        - pe: Position East (meters)
        - pd: Position Down (meters)
        - u: Velocity in body frame x-direction (m/s)
        - v: Velocity in body frame y-direction (m/s)
        - w: Velocity in body frame z-direction (m/s)
        - roll: Roll angle (radians)
        - pitch: Pitch angle (radians)
        - yaw: Yaw angle (radians)
        - p: Roll rate (radians/s)
        - q: Pitch rate (radians/s)
        - r: Yaw rate (radians/s)
        - q0, q1, q2, q3: Quaternions representing the aircraft's orientation
        """
        self.use_quat = use_quat

        if self.use_quat:
            self._x = np.zeros(13)
            self._x[6] = 1.0 # neutral quaternion is [1, 0, 0, 0]
            self._x_dot = np.zeros(13)
        else:
            self._x = np.zeros(12)
            self._x_dot = np.zeros(12)

        if not x0 is None:
            self._x = x0

        self._wind = wind0

        if self.use_quat:
            self._R_vb = rot_matrix_quat(self.quaternions)
        else:
            self._R_vb = rot_matrix_zyx(self.attitude_angles)  # vehicle to body frame

        self._R_wb = rot_matrix_wind(self.alpha, self.beta)  # wind to body frame
        self._R_sb = rot_matrix_wind(self.alpha, 0.0)  # stability to body frame

    @property
    def x(self) -> np.ndarray:
        """Aircrfat's state array.
        The structure of this array depends on the orientation representation selected by `use_quat`:
        If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
        """
        return self._x

    @property
    def x_dot(self) -> np.ndarray:
        """Aircraft's state time derivative (dx/dt).
        The structure of this array depends on the orientation representation selected by `use_quat`:
        If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
        """
        return self._x_dot

    @property
    def wind(self) -> np.ndarray:
        """3-size array with NED frame wind velocity [wn, we, wd] in m/s"""
        return self._wind

    @property
    def R_vb(self) -> np.ndarray:
        """Transformation matrix from vehicle frame to body frame (R^b_v)"""
        return self._R_vb

    @property
    def R_wb(self) -> np.ndarray:
        """Transformation matrix from wind frame to body frame (R^b_w)"""
        return self._R_wb

    @property
    def R_sb(self) -> np.ndarray:
        """Transformation matrix from stability frame to body frame (R^b_s)"""
        return self._R_sb

    @property
    def ned_position(self) -> np.ndarray:
        """3-size array with NED frame position [pn, pe, pd] in meters"""
        return self.x[0:3]

    @property
    def body_velocity(self) -> np.ndarray:
        """3-size array with body frame velocity [u, v, w] in m/s"""
        return self.x[3:6]

    @property
    def quaternions(self) -> np.ndarray:
        """3-size array with orientation quaternions [q0, q1, q2, q3]"""
        if self.use_quat:
            return self.x[6:10]
        else:
            return euler2quat(self.x[6:9])

    @property
    def attitude_angles(self) -> np.ndarray:
        """3-size array with attitude angles [roll, pitch, yaw] in radians"""
        if self.use_quat:
            return quat2euler(self.x[6:10])
        else:
            return self.x[6:9]

    @property
    def angular_rates(self) -> np.ndarray:
        """3-size array with angular rates [p, q, r] in radians/s"""
        if self.use_quat:
            return self.x[10:13]
        else:
            return self.x[9:12]

    @property
    def pn(self) -> float:
        """North position (meters)"""
        return self.x[0]

    @property
    def pe(self) -> float:
        """East position (meters)"""
        return self.x[1]

    @property
    def pd(self) -> float:
        """Down position (meters)"""
        return self.x[2]

    @property
    def u(self) -> float:
        """Velocity in body frame x-direction (m/s)"""
        return self.x[3]

    @property
    def v(self) -> float:
        """Velocity in body frame y-direction (m/s)"""
        return self.x[4]

    @property
    def w(self) -> float:
        """Velocity in body frame z-direction (m/s)"""
        return self.x[5]

    @property
    def roll(self) -> float:
        """Roll angle (radians)"""
        return self.attitude_angles[0]

    @property
    def pitch(self) -> float:
        """Pitch angle (radians)"""
        return self.attitude_angles[1]

    @property
    def yaw(self) -> float:
        """Yaw angle (radians)"""
        return self.attitude_angles[2]

    @property
    def p(self) -> float:
        """Roll rate (radians/s)"""
        return self.angular_rates[0]

    @property
    def q(self) -> float:
        """Pitch rate (radians/s)"""
        return self.angular_rates[1]

    @property
    def r(self) -> float:
        """Yaw rate (radians/s)"""
        return self.angular_rates[2]

    @property
    def altitude(self) -> float:
        """Vertical distance to inertial frame (NED) in meters"""
        return -self.pd

    @property
    def body_wind(self) -> np.ndarray:
        """3-size array with body frame wind vector [wx, wy, wz] in m/s"""
        return self.R_vb @ self.wind

    @property
    def body_airspeed(self) -> np.ndarray:
        """3-size array with body frame airspeed vector [ur, vr, wr] in m/s"""
        return self.body_velocity - self.body_wind

    @property
    def airspeed(self) -> float:
        """Airspeed value (m/s)"""
        return np.linalg.norm(self.body_airspeed)

    @property
    def alpha(self) -> float:
        """Angle of attack (rad)"""
        return np.arctan2(self.body_airspeed[2], self.body_airspeed[0])

    @property
    def beta(self) -> float:
        """Side-slip angle (rad)"""
        return np.arcsin(self.body_airspeed[1] / self.airspeed)

    @property
    def groundspeed(self) -> float:
        """Groundspeed value in m/s"""
        return np.linalg.norm(self.body_velocity)

    @property
    def ned_velocity(self) -> float:
        """3-size array with NED frame velocity [vn, ve, vd] in m/s"""
        return self.R_vb.T @ self.body_velocity

    @property
    def vspeed(self) -> float:
        """Vertical speed value in m/s"""
        return -self.ned_velocity[2]

    @property
    def course_angle(self) -> float:
        """Course angle (horizontal groundspeed direction relative to North) value in rads"""
        return np.arctan2(self.ned_velocity[1], self.ned_velocity[0])

    @property
    def path_angle(self) -> float:
        """Path angle (vertical groundspeed angle relative to horizontal plane) value in rads"""
        return -np.arcsin(self.ned_velocity[2] / self.groundspeed)

    @property
    def crab_angle(self) -> float:
        """Crab angle (difference between the course angle and the heading or yaw angle) value in rads"""
        return self.course_angle - self.yaw

    @property
    def air_path_angle(self) -> float:
        """Air-mass-referenced flight path angle (difference between pitch angle and angle of attack) value in rads"""
        return self.pitch - self.beta

    @property
    def body_acceleration(self) -> np.ndarray:
        """3-size array with body frame accelerations [ax, ay, az] in m/s^2"""
        return self.x_dot[3:6]

    def update(
        self,
        x: np.ndarray,
        x_dot: np.ndarray = None,
        wind: np.ndarray = None,
    ) -> None:
        """Update the aircraft's state `x`,
        and optionally the state derivative `x_dot` or the wind vector `wind`.

        Parameters
        ----------
        x : np.ndarray
            Aircraft's state vector.
            The structure of this array depends on the orientation representation selected by `use_quat`:
            If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
            If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
            By default None
        x_dot : np.ndarray, optional
            Aircraft's state derivative. The size corresponds to `x` state vector depending on `use_quat` value.
            By defaut None
        wind : np.ndarray, optional
            3-size array with wind velocity in NED frame: [wn, we, wd], by defaut None

        Notes
        -----
        The aircraft's state array `x` elements:
        - pn: Position North (meters)
        - pe: Position East (meters)
        - pd: Position Down (meters)
        - u: Velocity in body frame x-direction (m/s)
        - v: Velocity in body frame y-direction (m/s)
        - w: Velocity in body frame z-direction (m/s)
        - roll: Roll angle (radians)
        - pitch: Pitch angle (radians)
        - yaw: Yaw angle (radians)
        - p: Roll rate (radians/s)
        - q: Pitch rate (radians/s)
        - r: Yaw rate (radians/s)
        - q0, q1, q2, q3: Quaternions representing the aircraft's orientation
        """
        self._x = x

        if not x_dot is None:
            self._x_dot = x_dot

        if not wind is None:
            self._wind = wind

        if self.use_quat:
            self._R_vb = rot_matrix_quat(self.quaternions)
        else:
            self._R_vb = rot_matrix_zyx(self.attitude_angles)  # vehicle to body frame

        self._R_wb = rot_matrix_wind(self.alpha, self.beta)  # wind to body frame
        self._R_sb = rot_matrix_wind(self.alpha, 0.0)  # stability to body frame


    def __str__(self) -> str:
        """Return a string representation of the aircraft's state.

        Returns
        -------
        str
            A string representation of the aircraft's state
        """
        state_str = (
            f"Aircraft State:\n"
            f"- North Position (pn)  : {self.pn:.3f} m\n"
            f"- East Position (pe)   : {self.pe:.3f} m\n"
            f"- Down Position (pd)   : {self.pd:.3f} m\n"
            f"- Body Velocity (u)    : {self.u:.3f} m/s\n"
            f"- Body Velocity (v)    : {self.v:.3f} m/s\n"
            f"- Body Velocity (w)    : {self.w:.3f} m/s\n"
        )
        if self.use_quat:
            state_str += (
                f"- Quaternion (q0)      : {self.quaternions[0]:.5f}\n"
                f"- Quaternion (q1)      : {self.quaternions[1]:.5f}\n"
                f"- Quaternion (q2)      : {self.quaternions[2]:.5f}\n"
                f"- Quaternion (q3)      : {self.quaternions[3]:.5f}\n"
            )
        else:
            state_str += (
                f"- Roll Angle (roll)    : {self.roll:.3f} rad\n"
                f"- Pitch Angle (pitch)  : {self.pitch:.3f} rad\n"
                f"- Yaw Angle (yaw)      : {self.yaw:.3f} rad\n"
            )
        state_str += (
            f"- Roll Rate (p)        : {self.p:.3f} rad/s\n"
            f"- Pitch Rate (q)       : {self.q:.3f} rad/s\n"
            f"- Yaw Rate (r)         : {self.r:.3f} rad/s"
        )
        return state_str
