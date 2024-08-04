"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from scipy.optimize import minimize, fsolve

from simulator.math.numeric_differentiation import jacobian

from simulator.aircraft import (
    AirframeParameters,
    AircraftState,
    ControlDeltas,
    ForcesMoments,
    KinematicsDynamics,
)


class Trim:

    def __init__(self, params: AirframeParameters) -> None:
        self.params = params
        self.forces_moments = ForcesMoments(self.params)
        self.kinematics_dynamics = KinematicsDynamics(0.0, self.params)

    def calculate_trim(
        self, Va: float, gamma: float, R_orb: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        Va : float
            The trim airspeed value in m/s
        gamma : float
            The trim path angle in radians
        R_orb : float
            The trim orbit radius in meters

        Returns
        -------
        tuple[ndarray, ndarray]
            The trimmed aircraft's state (x_trim) and trimmed control deltas (delta_trim)
        """

        x_dot_trim = np.zeros(12)
        x_dot_trim[0] = +Va * np.cos(gamma)  # horizontal speed
        x_dot_trim[2] = -Va * np.sin(gamma)  # climb rate
        x_dot_trim[8] = Va / R_orb * np.cos(gamma)  # turn rate

        def objective(x: np.ndarray) -> float:
            x_trim = x[0:12]
            u_trim = x[12:16]
            err = np.linalg.norm(x_dot_trim - self.derivatives(x_trim, u_trim))
            return err**2

        x0_trim = np.zeros(12)
        x0_trim[3] = x_dot_trim[0]  # u = d(pn)/dt
        x0_trim[5] = x_dot_trim[2]  # w = d(pd)/dt
        u0_trim = np.zeros(4)
        u0_trim[3] = 0.5

        def cons_eq_x(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    np.linalg.norm(x[3:6]) - Va, # velocity magnitude equals to airspeed
                    x[4] - 0.0,  # zero side velocity
                ]
            )

        def cons_ineq_u(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    x[15] - 0.1,  # delta_t > 0.1
                    1.0 - x[15],  # delta_t < 1.0
                ]
            )

        cons = [{"type": "eq", "fun": cons_eq_x}, {"type": "ineq", "fun": cons_ineq_u}]

        result = minimize(
            objective,
            x0=np.append(x0_trim, u0_trim),
            method="SLSQP",
            tol=1e-9,
            constraints=cons,
            options={"maxiter": 1000, "disp": True},
        )

        x_trim = result.x[0:12]
        u_trim = result.x[12:16]

        state_trim = AircraftState(x_trim)
        dela_trim = ControlDeltas(u_trim)

        alpha = state_trim.alpha
        beta = state_trim.beta
        roll = state_trim.roll
        pitch = alpha + gamma

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

        return x_trim, u_trim

    def derivatives(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Dynamycs function as dx/dt = f(x, delta), where x is the current state and delta the control array.

        Parameters
        ----------
        x : np.ndarray
            12-size array with the aircrfat's state: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        delta : np.ndarray
            4-size array with the control deltas: [delta_a, delta_e, delta_r, delta_t]

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt
        """
        state = AircraftState(x)
        deltas = ControlDeltas(delta)
        u = self.forces_moments.update(state, deltas)
        x_dot = self.kinematics_dynamics.derivatives(x, u)
        return x_dot
