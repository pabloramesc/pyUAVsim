"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from scipy.optimize import minimize

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
        x_dot_trim[0] = Va * np.cos(gamma)  # horizontal speed (d(pn)/dt)
        x_dot_trim[2] = -Va * np.sin(gamma)  # climb rate (d(pd)/dt)
        x_dot_trim[8] = Va / R_orb * np.cos(gamma)  # turn rate (d(yaw)/dt)

        def fun(x: np.ndarray) -> float:
            x_trim = x[0:12]
            u_trim = x[12:16]
            err = np.linalg.norm(x_dot_trim - self.x_dot(x_trim, u_trim))
            return err**2

        x0_trim = np.zeros(12)
        x0_trim[3] = x_dot_trim[0]  # u = d(pn)/dt
        x0_trim[5] = x_dot_trim[2]  # w = d(pd)/dt
        x0_trim[11] = x_dot_trim[8]  # r = d(yaw)/dt
        u0_trim = np.zeros(4)

        def cons_eq_x(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    x[0] - 0.0, # pn = 0.0
                    x[1] - 0.0, # pe = 0.0
                    x[2] - 0.0, # pd = 0.0
                    x[3] - x0_trim[3],  # u = Va * cos(gamma)
                    x[4] - 0.0,  # v = 0.0
                    x[5] - x0_trim[5],  # w = -Va * sin(gamma)
                    x[8] - 0.0, # yaw = 0.0
                    x[9] - 0.0, # p = 0.0
                    x[10] - 0.0, # q = 0.0
                    x[11] - x0_trim[11], # r = Va / Rorb * cos(gamma)
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
            fun,
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

        return x_trim, u_trim

    def x_dot(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        x : np.ndarray
            12-size array with the aircrfat's state: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        u : np.ndarray
            4-size array with the control deltas: [delta_a, delta_e, delta_r, delta_t]

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt
        """
        state = AircraftState(state0=x)
        deltas = ControlDeltas(deltas0=u)

        forces_moments = ForcesMoments(self.params)
        f, m = forces_moments.update(state, deltas)

        kinematics_dynamics = KinematicsDynamics(0.0, self.params, state)
        dxdt = kinematics_dynamics.x_dot(x, u=np.append(f, m))

        return dxdt
