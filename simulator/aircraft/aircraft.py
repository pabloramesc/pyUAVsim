"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.dynamics import Dynamics

class Aircraft:
    def __init__(self, dt: float, params: AirframeParameters, state0: np.ndarray = np.zeros(12), wind: np.ndarray = np.zeros(3)) -> None:
        """Initialize the Aircraft class.

        Parameters
        ----------
        dt : float
            Time step for integration (seconds)
        params : AirframeParameters
            Parameters of the airframe
        state0 : np.ndarray, optional
            Initial state array (12 variables: pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r),
            by default np.zeros(12)
        wind : np.ndarray, optional
            Wind vector in NED frame (3-size array: wn, we, wd in m/s), by default np.zeros(3)
        """
        self.params = params
        self.state = AircraftState(state0, wind)
        self.dynamics = Dynamics(dt, params, self.state, wind)
        self.t = 0.0
        self.dt = dt

    def update_state(self, f: np.ndarray, m: np.ndarray) -> None:
        """Update the aircraft state using external forces and moments.
        It integrates the kinematics and dynamics equations to calculate the current state. 

        Parameters
        ----------
        f : np.ndarray
            3-size array with external forces in body frame [fx, fy, fz] (N)
        m : np.ndarray
            3-size array with external moments applied to body axis [l, m, n] (Nm)

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
        new_state = self.dynamics.dynamics(f, m)
        self.state.update(new_state)
