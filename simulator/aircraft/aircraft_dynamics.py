"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from scipy.optimize import minimize

from simulator.aircraft.aerodynamic_model import AerodynamicModel
from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.aircraft.propulsion_model import PropulsionModel
from simulator.common.constants import EARTH_GRAVITY_VECTOR
from simulator.math.numeric_integration import rk4
from simulator.math.rotation import (
    euler_kinematics,
    rot_matrix_zyx,
    quaternion_kinematics,
    rot_matrix_quat,
)


class AircraftDynamics:
    def __init__(
        self,
        dt: float,
        params: AirframeParameters,
        use_quat: bool = False,
        wind0: np.ndarray = np.zeros(3),
        x0: np.ndarray = np.zeros(12),
        delta0: np.ndarray = np.zeros(4),
    ) -> None:
        """Initialize the AircraftDynamics class.

        Parameters
        ----------
        dt : float
            Time step for integration (seconds)
        params : AirframeParameters
            Parameters of the airframe
        use_quat : bool, optional
            Flag to indicate whether to use quaternions for representing orientation.
            If True, quaternions will be used; otherwise, Euler angles will be used.
            By default False
        wind0 : np.ndarray, optional
            Initial wind vector in NED frame (3-size array: wn, we, wd in m/s), by default np.zeros(3)
        x0 : np.ndarray, optional
            Initial state vector.
            The structure of this array depends on the orientation representation selected by `use_quat`:
            If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
            If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
            By default None
        delta0 : np.ndarray, optional
            Initial delta array (4 variables: delta_a, delta_e, delta_r, delta_t), by default np.zeros(12)
        """
        self.t = 0.0
        self.dt = dt

        self.use_quat = use_quat

        self.u = np.zeros(6)

        self.params = params
        self.state = AircraftState(x0, wind0)
        self.deltas = ControlDeltas(delta0)
        self.aerodynamics = AerodynamicModel(params)
        self.propulsion = PropulsionModel(params)

    def set_state(self, x: np.ndarray = np.zeros(12)) -> None:
        self.state.update(x)

    def set_deltas(self, delta: np.ndarray = np.zeros(4)) -> None:
        self.deltas.update(delta)

    def update(self, deltas: ControlDeltas = None) -> AircraftState:
        """Update aircraft's state simulating the flight dynamics.
        Firstly, forces and moments are calculated using gravity, aerodynamics and propulsion models.
        Then, numeric integration of the kinematic and dynamic equations provide the new updated state.

        If `deltas` parameter is provided,
        internal control deltas atribute will be updated before new state calculation.

        Parameters
        ----------
        deltas : ControlDeltas, optional
            Current control deltas, by default None

        Returns
        -------
        AircraftState
            New updated aricraft's state
        """
        if not deltas is None:
            self.deltas = deltas
        u = self.forces_moments(self.state, self.deltas)
        x = self.kinematics_dynamics(self.state.x, u)
        x_dot = self.state_derivatives(x, u)
        self.t += self.dt
        self.u = u
        self.state.update(x, x_dot)

    def forces_moments(self, state: AircraftState, deltas: ControlDeltas) -> np.ndarray:
        """Calcuate external forces and moments acting on the aircraft due to gravity, aerodynamics and propulsion.

        Parameters
        ----------
        state : AircraftState
            Current aircraft's state
        deltas : ControlDeltas
            Current control deltas

        Returns
        -------
        np.ndarray
            Total external forces and moments array in body frame: [fx, fy, fx, l, m, n]
        """
        # gravity force in body frame
        fg = state.R_vb @ EARTH_GRAVITY_VECTOR
        ug = np.concatenate([fg, np.zeros(3)])  # gravity doesn't generate any moment

        # aerodynamic forces and moments
        ua = self.aerodynamics.calculate_forces_moments(state, deltas)

        # propulsion forces and moments
        up = self.propulsion.calculate_forces_moments(state, deltas)

        # total forces and moments
        return ug + ua + up

    def kinematics_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Integrate the kinematics and dynamics equations to calculate the new state.

        Parameters
        ----------
        x : np.ndarray
            Aircraft's state vector.
            The structure of this array depends on the orientation representation selected by `use_quat`:
            If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
            If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
            By default None
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
        - q0, q1, q2, q3: Quaternions representing the aircraft's orientation

        The external forces and moments array `u` elements:
        - fx: External force in body frame x-direction (N)
        - fy: External force in body frame y-direction (N)
        - fz: External force in body frame z-direction (N)
        - l: External moment around body x-axis (Nm)
        - m: External moment around body y-axis (Nm)
        - n: External moment around body z-axis (Nm)
        """
        func = lambda t, y: self.state_derivatives(y, u)
        dx = rk4(func, self.t, x, self.dt)
        x2 = x + dx
        if self.use_quat:
            x2[6:10] = x2[6:10] / np.linalg.norm(
                x[6:10]
            )  # normalize quaternion after integration step
        return x2

    def state_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """State transition or dynamics function for numeric integration: dx/dt = f(x, u)

        Parameters
        ----------
        x : np.ndarray
            Aircraft's state vector.
            The structure of this array depends on the orientation representation selected by `use_quat`:
            If euler angles are used, the array contains 12 elements: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
            If quaternions are used, the array contains 13 elements: [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
            By default None
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
        - q0, q1, q2, q3: Quaternions representing the aircraft's orientation

        The external forces and moments array `u` elements:
        - fx: External force in body frame x-direction (N)
        - fy: External force in body frame y-direction (N)
        - fz: External force in body frame z-direction (N)
        - l: External moment around body x-axis (Nm)
        - m: External moment around body y-axis (Nm)
        - n: External moment around body z-axis (Nm)
        """
        x_dot = None

        if self.use_quat:
            x_dot = np.zeros(13)

            # Calculate position kinematics:
            # d[pn pe pd]/dt = R_bv * [u v w]
            R_bv = rot_matrix_quat(
                x[6:10]
            ).T  # transformation matrix from body frame to vehicle frame
            x_dot[0:3] = R_bv @ x[3:6]

            # Calculate position dynamics:
            # d[u v w]/dt = -[p q r] x [u v w] + 1/m * [fx fy fz]
            # u_dot = (r * v - q * w) + fx / m
            # v_dot = (p * w - r * u) + fy / m
            # w_dot = (q * u - p * v) + fz / m
            x_dot[3:6] = -np.cross(x[10:13], x[3:6]) + u[0:3] / self.params.m

            # Calculate attitude kinematics:
            # q_dot = 1/2 * Omega([p q r]) * q
            x_dot[6:10] = quaternion_kinematics(
                x[10:13], x[6:10]
            )  # derivative of the orientation quaternion

            # Calculate attitude dynamics:
            # d[p q r]/dt = J^-1 * (-[p q r] x (J * [p q r]) + [l m n])
            # p_dot = (Gamma1 * p * q - Gamma2 * q * r + Gamma3 * l + Gamma4 * n)
            # q_dot = (Gamma5 * p * r - Gamma6 * (p**2 - r**2) + m / Jy)
            # r_dot = (Gamma7 * p * q - Gamma1 * q * r + Gamma4 * l+ Gamma8 * n)
            x_dot[10:13] = self.params.Jinv @ (
                -np.cross(x[10:13], (self.params.J @ x[10:13])) + u[3:6]
            )

        else:
            x_dot = np.zeros(12)

            # Calculate position kinematics:
            # d[pn pe pd]/dt = R_bv * [u v w]
            R_bv = rot_matrix_zyx(
                x[6:9]
            ).T  # transformation matrix from body frame to vehicle frame
            x_dot[0:3] = R_bv @ x[3:6]

            # Calculate position dynamics:
            # d[u v w]/dt = -[p q r] x [u v w] + 1/m * [fx fy fz]
            # u_dot = (r * v - q * w) + fx / m
            # v_dot = (p * w - r * u) + fy / m
            # w_dot = (q * u - p * v) + fz / m
            x_dot[3:6] = -np.cross(x[9:12], x[3:6]) + u[0:3] / self.params.m

            # Calculate attitude kinematics:
            # d[roll yaw pitch]/dt = [roll 0 0] + R(roll, 0, 0) * [0 pitch 0] + R(roll, pitch, 0) * [0 0 yaw]
            x_dot[6:9] = euler_kinematics(
                x[9:12], x[6], x[7]
            )  # derivative of the attitude in euler angles

            # Calculate attitude dynamics:
            # d[p q r]/dt = J^-1 * (-[p q r] x (J * [p q r]) + [l m n])
            # p_dot = (Gamma1 * p * q - Gamma2 * q * r + Gamma3 * l + Gamma4 * n)
            # q_dot = (Gamma5 * p * r - Gamma6 * (p**2 - r**2) + m / Jy)
            # r_dot = (Gamma7 * p * q - Gamma1 * q * r + Gamma4 * l+ Gamma8 * n)
            x_dot[9:12] = self.params.Jinv @ (
                -np.cross(x[9:12], (self.params.J @ x[9:12])) + u[3:6]
            )

        return x_dot

    def trim(
        self,
        Va: float,
        gamma: float,
        R_orb: float,
        update: bool = True,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the trimmed states and deltas for the trim conditions,
        such that the aircraft maintains a steady flight.

        If `update` is True, the internal `state` and `deltas` of the aircraft are updated with the computed trim values.

        Parameters
        ----------
        Va : float
            The trim airspeed value in m/s
        gamma : float
            The trim path angle in radians
        R_orb : float
            The trim orbit radius in meters
        update : bool, optional
            To update internal state and deltas with calculated trim,
            by default True

        Returns
        -------
        tuple[ndarray, ndarray]
            The trimmed aircraft's state `x_trim` and trimmed control deltas `delta_trim`
        """
        if verbose:
            print("Calculating trim states and deltas...")

        # desired derivative state for the trim conditions
        if self.use_quat:
            x_dot_trim = np.zeros(13)
            # TODO: turn rate constrain for quaternions
        else:
            x_dot_trim = np.zeros(12)
            x_dot_trim[8] = Va / R_orb * np.cos(gamma)  # turn rate
        x_dot_trim[0] = +Va * np.cos(gamma)  # horizontal speed
        x_dot_trim[2] = -Va * np.sin(gamma)  # climb rate

        # cuadratic error between desired x_dot_trim and dynamics function
        def objective(v: np.ndarray) -> float:
            if self.use_quat:
                state = AircraftState(v[0:13], use_quat=True)
                deltas = ControlDeltas(v[13:17])
            else:
                state = AircraftState(v[0:12], use_quat=False)
                deltas = ControlDeltas(v[12:16])
            u = self.forces_moments(state, deltas)
            x_dot = self.state_derivatives(state.x, u)
            err = np.linalg.norm(x_dot_trim - x_dot)
            return err**2

        # initial guesses for x_trim and delta_trim
        if self.use_quat:
            x0_trim = np.zeros(13)
        else:
            x0_trim = np.zeros(12)
        x0_trim[3] = x_dot_trim[0]  # u = d(pn)/dt
        x0_trim[5] = x_dot_trim[2]  # w = d(pd)/dt
        delta0_trim = np.zeros(4)
        delta0_trim[3] = 0.5

        # equality constrains
        def cons_eq_x(v: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    np.linalg.norm(v[3:6])
                    - Va,  # velocity magnitude equals to airspeed
                    v[4] - 0.0,  # zero side velocity
                ]
            )

        # inequality constrains
        def cons_ineq_u(v: np.ndarray) -> np.ndarray:
            if self.use_quat:
                return np.array(
                    [
                        v[16] - 0.1,  # delta_t > 0.1
                        1.0 - v[16],  # delta_t < 1.0
                    ]
                )
            else:
                return np.array(
                    [
                        v[15] - 0.1,  # delta_t > 0.1
                        1.0 - v[15],  # delta_t < 1.0
                    ]
                )

        # minimize calculate trim variables
        result = minimize(
            objective,
            x0=np.concatenate([x0_trim, delta0_trim]),
            method="SLSQP",
            tol=1e-9,
            constraints=[
                {"type": "eq", "fun": cons_eq_x},
                {"type": "ineq", "fun": cons_ineq_u},
            ],
            options={"maxiter": 1000, "disp": True},
        )
        if self.use_quat:
            x_trim = result.x[0:13]
            delta_trim = result.x[13:17]
        else:
            x_trim = result.x[0:12]
            delta_trim = result.x[12:16]
        state_trim = AircraftState(x_trim, use_quat=self.use_quat)
        deltas_trim = ControlDeltas(delta_trim)

        if verbose:
            if result.success:
                print("Trim calculation success! Plotting results...")
                print(state_trim)
                print(deltas_trim)
            else:
                print("Trim calculation failed!")

        # compute trimmed states
        alpha = state_trim.alpha
        beta = state_trim.beta
        roll = state_trim.roll
        pitch = alpha + gamma
        if self.use_quat:
            x_trim[0] = 0.0  # pn
            x_trim[1] = 0.0  # pe
            x_trim[2] = 0.0  # pd
            x_trim[3] = Va * np.cos(alpha) * np.cos(beta)  # u
            x_trim[4] = Va * np.sin(beta)  # v
            x_trim[5] = Va * np.sin(alpha) * np.cos(beta)  # w
            x_trim[6] = 0.0  # q0
            x_trim[7] = 0.0  # q1
            x_trim[8] = 0.0  # q2
            x_trim[9] = 0.0  # q3
            x_trim[10] = -Va / R_orb * np.sin(pitch)  # p
            x_trim[11] = Va / R_orb * np.sin(roll) * np.cos(pitch)  # q
            x_trim[12] = Va / R_orb * np.cos(roll) * np.cos(pitch)  # r
        else:
            x_trim[0] = 0.0  # pn
            x_trim[1] = 0.0  # pe
            x_trim[2] = 0.0  # pd
            x_trim[3] = Va * np.cos(alpha) * np.cos(beta)  # u
            x_trim[4] = Va * np.sin(beta)  # v
            x_trim[5] = Va * np.sin(alpha) * np.cos(beta)  # w
            x_trim[6] = roll  # roll
            x_trim[7] = pitch  # pitch
            x_trim[8] = 0.0  # yaw
            x_trim[9] = -Va / R_orb * np.sin(pitch)  # p
            x_trim[10] = Va / R_orb * np.sin(roll) * np.cos(pitch)  # q
            x_trim[11] = Va / R_orb * np.cos(roll) * np.cos(pitch)  # r

        # update internal atributes if needed
        if update:
            self.set_state(x_trim)
            self.set_deltas(delta_trim)
            if verbose:
                print("Internal states and deltas updated with calculated trim.")

        return x_trim, delta_trim
