"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.math.angles import rot_matrix_zyx, attitude_dt
from simulator.math.numeric_integration import rk4


class Aircraft:
    def __init__(
        self,
        dt: float,
        params: AirframeParameters,
        state0: np.ndarray = np.zeros(12),
        wind: np.ndarray = np.zeros(3),
    ) -> None:
        # Aircraft current state (12 vars):
        # [pn pe pd u v w roll pitch yaw p q r]
        #  0  1  2  3 4 5 6    7     8   9 1011
        self.params = params
        self.state = state0
        self.wind = wind
        self.t = 0.0
        self.dt = dt

    @property
    def ned_position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def body_velocity(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def attitude_angles(self) -> np.ndarray:
        return self.state[6:9]

    @property
    def angular_rates(self) -> np.ndarray:
        return self.state[9:12]

    @property
    def pn(self) -> float:
        return self.state[0]

    @property
    def pd(self) -> float:
        return self.state[1]

    @property
    def pe(self) -> float:
        return self.state[2]

    @property
    def u(self) -> float:
        return self.state[3]

    @property
    def v(self) -> float:
        return self.state[4]

    @property
    def w(self) -> float:
        return self.state[5]

    @property
    def roll(self) -> float:
        return self.state[6]

    @property
    def pitch(self) -> float:
        return self.state[7]

    @property
    def yaw(self) -> float:
        return self.state[8]

    @property
    def p(self) -> float:
        return self.state[9]

    @property
    def q(self) -> float:
        return self.state[10]

    @property
    def r(self) -> float:
        return self.state[11]

    def dynamics(self, f: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Integrate the kinematics and dynamics equations to calculate the current state.

        Parameters
        ----------
        f : np.ndarray
            3-size array with external forces in body frame (fx, fy, fz)
        m : np.ndarray
            3-size array with external moments applied to body axis (l, m, n)

        Returns
        -------
        np.ndarray
            12-size array with updated state after integration
            (pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r)
        """
        u = np.append(f, m)
        func = lambda t, y: self._f(y, u)
        dy = rk4(func, self.t, self.state, self.dt)
        return self.state + dy

    def _f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Dynamics function for numeric integration: dx/dt = f(x, u)

        Parameters
        ----------
        x : np.ndarray
            12-size array with aircraft last state
            (pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r)
        u : np.ndarray
            6-size array with external forces and moments
            (fx, fy, fz, l, m, n)

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt
        """
        # Extract forces and moments
        f = u[0:3]
        m = u[3:6]
        fx, fy, fz = f
        ml, mm, mn = m

        # Calculate position kinematics:
        # d[pn pe pd]/dt = R_bv * [u v w]
        R_bv = rot_matrix_zyx(
            self.attitude_angles
        ).T  # transformation matrix from body frame to vehicle frame
        dpos = R_bv @ self.body_velocity

        # Calculate position dynamics:
        # d[u v w]/dt = -[p q r] x [u v w] + 1/m * [fx fy fz]
        duvw = -np.cross(self.angular_rates, self.body_velocity) + f / self.params.m
        # du = (self.r * self.v - self.q * self.w) + fx / m
        # dv = (self.p * self.w - self.r * self.u) + fy / m
        # dw = (self.q * self.u - self.p * self.v) + fz / m
        # duvw = np.array([du, dv, dw])

        # Calculate attitude kinematics:
        # d[roll yaw pitch]/dt = [roll 0 0] + R(roll, 0, 0) * [0 pitch 0] + R(roll, pitch, 0) * [0 0 yaw]
        R_dt = attitude_dt(
            self.angular_rates, self.roll, self.pitch
        )  # derivative of the attitude matrix
        datt = R_dt @ self.angular_rates

        # Calculate attitude dynamics:
        # d[p q r]/dt = J^-1 * (-[p q r] x (J * [p q r]) + [l m n])
        dpqr = self.params.Jinv @ (
            -np.cross(self.angular_rates, (self.params.J @ self.angular_rates)) + m
        )
        # dp = (
        #     self.params.Gamma1 * self.p * self.q
        #     - self.params.Gamma2 * self.q * self.r
        #     + self.params.Gamma3 * ml
        #     + self.params.Gamma4 * mn
        # )
        # dq = (
        #     self.params.Gamma5 * self.p * self.r
        #     - self.params.Gamma6 * (self.p**2 - self.r**2)
        #     + mm / self.params.Jy
        # )
        # dr = (
        #     self.params.Gamma7 * self.p * self.q
        #     - self.params.Gamma1 * self.q * self.r
        #     + self.params.Gamma4 * ml
        #     + self.params.Gamma8 * mn
        # )
        # dpqr = np.array([dp, dq, dr])

        dxdt = np.zeros(12)
        dxdt[0:3] = dpos
        dxdt[3:6] = duvw
        dxdt[6:9] = datt
        dxdt[9:12] = dpqr
        return dxdt
