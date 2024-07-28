"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.forces_moments import ForcesMoments
from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.aircraft.kinematics_dynamics import KinematicsDynamics


class Aircraft:
    def __init__(
        self,
        dt: float,
        params: AirframeParameters,
        wind: np.ndarray = np.zeros(3),
        state0: np.ndarray = np.zeros(12),
        deltas0: np.ndarray = np.zeros(4),
    ) -> None:
        """Initialize the Aircraft class.

        Parameters
        ----------
        dt : float
            Time step for integration (seconds)
        params : AirframeParameters
            Parameters of the airframe
        wind : np.ndarray, optional
            Wind vector in NED frame (3-size array: wn, we, wd in m/s), by default np.zeros(3)
        state0 : np.ndarray, optional
            Initial state array (12 variables: pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r),
            by default np.zeros(12)
        deltas0 : np.ndarray, optional
            Initial deltas array (4 variables: da, de, dr, dt),
            by default np.zeros(12)
        """
        self.t = 0.0
        self.dt = dt

        self.wind = wind

        self.params = params
        self.state = AircraftState(state0, wind)
        self.deltas = ControlDeltas(deltas0)
        self.kinematics_dynamics = KinematicsDynamics(dt, params, self.state, wind)
        self.forces_moments = ForcesMoments(params)

    def update_deltas(self, deltas: np.ndarray = np.zeros(4)) -> None:
        self.deltas.update(deltas)

    def update_state(self) -> None:
        """Update the aircraft state using external forces and moments.
        It integrates the kinematics and dynamics equations to calculate the current state.

        ### State array (12 variables):
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
        u = self.forces_moments.update(self.state, self.deltas)
        f = u[0:3]
        m = u[3:6]
        new_state = self.kinematics_dynamics.update(f, m)
        self.state.update(new_state)