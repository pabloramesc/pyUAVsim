import numpy as np

from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.aircraft_state import AircraftState

from simulator.math.rotation import rot_matrix_zyx, attitude_dt
from simulator.math.numeric_integration import rk4


class Dynamics:
    def __init__(
        self,
        dt: float,
        params: AirframeParameters,
        state: AircraftState,
        wind: np.ndarray = np.zeros(3),
    ) -> None:
        """Initialize the Dynamics class.

        Parameters
        ----------
        dt : float
            Time step for integration (seconds)
        params : AirframeParameters
            Parameters of the airframe
        state : AircraftState
            State object representing the aircraft state
        wind : np.ndarray, optional
            Wind vector (3-size array: wx, wy, wz in m/s), by default np.zeros(3)
        """
        self.params = params
        self.state = state
        self.wind = wind
        self.t = 0.0
        self.dt = dt

    def dynamics(self, f: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Integrate the kinematics and dynamics equations to calculate the current state.

        Parameters
        ----------
        f : np.ndarray
            3-size array with external forces in body frame [fx, fy, fz] (N)
        m : np.ndarray
            3-size array with external moments applied to body axis [l, m, n] (Nm)

        Returns
        -------
        np.ndarray
            12-size array with updated state after integration
            [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
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
        u = np.append(f, m)
        func = lambda t, y: self._func(y, u)
        dy = rk4(func, self.t, self.state.state, self.dt)
        return self.state.state + dy

    def _func(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Dynamics function for numeric integration: dx/dt = f(x, u)

        Parameters
        ----------
        x : np.ndarray
            12-size array with aircraft last state
            [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
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
        u : np.ndarray
            6-size array with external forces and moments
            [fx, fy, fz, l, m, n]
            - fx: External force in body frame x-direction (N)
            - fy: External force in body frame y-direction (N)
            - fz: External force in body frame z-direction (N)
            - l: External moment around body x-axis (Nm)
            - m: External moment around body y-axis (Nm)
            - n: External moment around body z-axis (Nm)

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt
            [d(pn), d(pe), d(pd), d(u), d(v), d(w), d(roll), d(pitch), d(yaw), d(p), d(q), d(r)]
        """
        # Extract forces and moments
        f = u[0:3]
        m = u[3:6]
        fx, fy, fz = f
        ml, mm, mn = m

        # Calculate position kinematics:
        # d[pn pe pd]/dt = R_bv * [u v w]
        R_bv = rot_matrix_zyx(
            self.state.attitude_angles
        ).T  # transformation matrix from body frame to vehicle frame
        dpos = R_bv @ self.state.body_velocity

        # Calculate position dynamics:
        # d[u v w]/dt = -[p q r] x [u v w] + 1/m * [fx fy fz]
        duvw = (
            -np.cross(self.state.angular_rates, self.state.body_velocity)
            + f / self.params.m
        )
        # du = (self.r * self.v - self.q * self.w) + fx / self.params.m
        # dv = (self.p * self.w - self.r * self.u) + fy / self.params.m
        # dw = (self.q * self.u - self.p * self.v) + fz / self.params.m
        # duvw = np.array([du, dv, dw])

        # Calculate attitude kinematics:
        # d[roll yaw pitch]/dt = [roll 0 0] + R(roll, 0, 0) * [0 pitch 0] + R(roll, pitch, 0) * [0 0 yaw]
        R_dt = attitude_dt(
            self.state.angular_rates, self.state.roll, self.state.pitch
        )  # derivative of the attitude matrix
        datt = R_dt @ self.state.angular_rates

        # Calculate attitude dynamics:
        # d[p q r]/dt = J^-1 * (-[p q r] x (J * [p q r]) + [l m n])
        dpqr = self.params.Jinv @ (
            -np.cross(
                self.state.angular_rates, (self.params.J @ self.state.angular_rates)
            )
            + m
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
