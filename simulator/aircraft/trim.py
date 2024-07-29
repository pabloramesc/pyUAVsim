"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""
import numpy as np

from simulator.aircraft import AirframeParameters, AircraftState
class Trim:
    
    def __init__(self, params: AirframeParameters) -> None:
        self.params = params

        self.state = np.zeros(12)
        self.deltas = np.zeros(4)
    
    def set_trim_parameters(Va: float, gamma: float, R_orb: float):
        """_summary_

        Parameters
        ----------
        Va : float
            The trim airspeed value in m/s
        gamma : float
            The trim path angle in radians
        R_orb : float
            The trim orbit radius in meters
        """

        x_dot_trim = np.zeros(12)
        x_dot_trim[2] = Va * np.sin(gamma) # climb rate
        x_dot_trim[8] = Va / R_orb * np.cos(gamma) # turn rate

    def x_dot(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        x : np.ndarray
            12-size array with the state variables: [pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r]
        u : np.ndarray
            4-size array with the control deltas: [delta_a, delta_e, delta_r, delta_t]

        Returns
        -------
        np.ndarray
            12-size array with time derivative of the aircraft state: dx/dt
        """
        state = AircraftState()