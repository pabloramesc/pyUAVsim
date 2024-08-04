"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.airframe_parameters import AirframeParameters

from simulator.math.rotation import rot_matrix_zyx, attitude_dt
from simulator.math.numeric_integration import rk4


class KinematicsDynamics:
    def __init__(self, dt: float, params: AirframeParameters) -> None:
        """Initialize the Dynamics class.

        Parameters
        ----------
        dt : float
            Time step for integration in seconds
        params : AirframeParameters
            Parameters of the airframe
        """
        self.params = params

        self.t = 0.0
        self.dt = dt

    def update(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Integrate the kinematics and dynamics equations to calculate the new state.

        Parameters
        ----------
        x : np.ndarray
            12-size array with aircraft last state: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        u : np.ndarray
            6-size array with external forces and moments in body frame [fx, fy, fz, l, m, n]

        Returns
        -------
        np.ndarray
            12-size array with updated state after integration: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]

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

        The external forces and moments array `u` elements:
        - fx: External force in body frame x-direction (N)
        - fy: External force in body frame y-direction (N)
        - fz: External force in body frame z-direction (N)
        - l: External moment around body x-axis (Nm)
        - m: External moment around body y-axis (Nm)
        - n: External moment around body z-axis (Nm)
        """
        func = lambda t, y: self.derivatives(y, u)
        dx = rk4(func, self.t, x, self.dt)
        return x + dx

    def derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Dynamics function for numeric integration: dx/dt = f(x, u)

        Parameters
        ----------
        x : np.ndarray
            12-size array with aircraft last state: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]

        u : np.ndarray
            6-size array with external forces and moments: [fx, fy, fz, l, m, n]

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt

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

        The external forces and moments array `u` elements:
        - fx: External force in body frame x-direction (N)
        - fy: External force in body frame y-direction (N)
        - fz: External force in body frame z-direction (N)
        - l: External moment around body x-axis (Nm)
        - m: External moment around body y-axis (Nm)
        - n: External moment around body z-axis (Nm)
        """

        x_dot = np.zeros(12)

        # Calculate position kinematics:
        # d[pn pe pd]/dt = R_bv * [u v w]
        R_bv = rot_matrix_zyx(x[6:9]).T  # transformation matrix from body frame to vehicle frame
        x_dot[0:3] = R_bv @ x[3:6]

        # Calculate position dynamics:
        # d[u v w]/dt = -[p q r] x [u v w] + 1/m * [fx fy fz]
        # u_dot = (r * v - q * w) + fx / m
        # v_dot = (p * w - r * u) + fy / m
        # w_dot = (q * u - p * v) + fz / m
        x_dot[3:6] = -np.cross(x[9:12], x[3:6]) + u[0:3] / self.params.m

        # Calculate attitude kinematics:
        # d[roll yaw pitch]/dt = [roll 0 0] + R(roll, 0, 0) * [0 pitch 0] + R(roll, pitch, 0) * [0 0 yaw]
        R_dt = attitude_dt(x[9:12], x[6], x[7])  # derivative of the attitude matrix
        x_dot[6:9] = R_dt @ x[9:12]

        # Calculate attitude dynamics:
        # d[p q r]/dt = J^-1 * (-[p q r] x (J * [p q r]) + [l m n])
        # p_dot = (Gamma1 * p * q - Gamma2 * q * r + Gamma3 * l + Gamma4 * n)
        # q_dot = (Gamma5 * p * r - Gamma6 * (p**2 - r**2) + m / Jy)
        # r_dot = (Gamma7 * p * q - Gamma1 * q * r + Gamma4 * l+ Gamma8 * n)
        x_dot[9:12] = self.params.Jinv @ (-np.cross(x[9:12], (self.params.J @ x[9:12])) + u[3:6])

        return x_dot
