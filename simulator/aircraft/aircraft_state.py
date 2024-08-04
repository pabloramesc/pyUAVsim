"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.math.rotation import rot_matrix_zyx, rot_matrix_wind


class AircraftState:
    def __init__(
        self, x0: np.ndarray = np.zeros(12), wind0: np.ndarray = np.zeros(3)
    ) -> None:
        """Initialize the State class.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state array (12 variables: pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r),
            by default np.zeros(12)
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
        wind0 : np.ndarray, optional
            Initial wind vector (3-size array: wx, wy, wz in m/s), by default np.zeros(3)
        """
        self.x = x0
        self.x_dot = np.zeros(12)

        self.wind = wind0

        self.R_vb = rot_matrix_zyx(self.attitude_angles)  # vehicle to body frame
        self.R_wb = rot_matrix_wind(self.alpha, self.beta)  # wind to body frame
        self.R_sb = rot_matrix_wind(self.alpha, 0.0)  # stability to body frame

    @property
    def ned_position(self) -> np.ndarray:
        """3-size array with NED frame position [pn, pe, pd] in meters"""
        return self.x[0:3]

    @property
    def body_velocity(self) -> np.ndarray:
        """3-size array with body frame velocity [u, v, w] in m/s"""
        return self.x[3:6]

    @property
    def attitude_angles(self) -> np.ndarray:
        """3-size array with attitude angles [roll, pitch, yaw] in radians"""
        return self.x[6:9]

    @property
    def angular_rates(self) -> np.ndarray:
        """3-size array with angular rates [p, q, r] in radians/s"""
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
        return self.x[6]

    @property
    def pitch(self) -> float:
        """Pitch angle (radians)"""
        return self.x[7]

    @property
    def yaw(self) -> float:
        """Yaw angle (radians)"""
        return self.x[8]

    @property
    def p(self) -> float:
        """Roll rate (radians/s)"""
        return self.x[9]

    @property
    def q(self) -> float:
        """Pitch rate (radians/s)"""
        return self.x[10]

    @property
    def r(self) -> float:
        """Yaw rate (radians/s)"""
        return self.x[11]

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
            12-size array with aircraft's state: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        x_dot : np.ndarray, optional
            12-size array with aircraft's state derivative, by defaut None
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
        """
        self.x = x
        if not x_dot is None:
            self.x_dot = x_dot
        if not wind is None:
            self.wind = wind

        self.R_vb = rot_matrix_zyx(self.attitude_angles)  # vehicle to body frame
        self.R_wb = rot_matrix_wind(self.alpha, self.beta)  # wind to body frame
        self.R_sb = rot_matrix_wind(self.alpha, 0.0)  # stability to body frame

    def __str__(self):
        """
        Return a string representation of the aircraft state.

        Returns:
        -------
        str
            A string representation of the aircraft state.
        """
        state_names = [
            "pn (North position)",
            "pe (East position)",
            "pd (Down position)",
            "u (body frame velocity along x-axis)",
            "v (body frame velocity along y-axis)",
            "w (body frame velocity along z-axis)",
            "phi (roll angle)",
            "theta (pitch angle)",
            "psi (yaw angle)",
            "p (roll rate)",
            "q (pitch rate)",
            "r (yaw rate)",
        ]

        state_str = "\n".join(
            f"{name}: {value:.3f}" for name, value in zip(state_names, self.x)
        )
        return f"Aircraft State:\n{state_str}"
