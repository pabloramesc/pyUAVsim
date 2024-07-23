"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""
import numpy as np

class ControlSurfaces:

    def __init__(self, deltas0: np.ndarray = np.zeros(3)) -> None:
        """Initialize de ControlSurface class.

        Parameters
        ----------
        deltas0 : np.ndarray, optional
            Control surfaces' deflection angles [da, de, dr] in rads, by default np.zeros(3)
        """
        self.deltas = deltas0

    def update(self, deltas: np.ndarray) -> None:
        """Update the control surfaces' deflection angles.

        Parameters
        ----------
        deltas : np.ndarray
            Control surfaces' deflection angles [da, de, dr] in rads
        """
        self.deltas = deltas

    @property
    def delta_a(self) -> float:
        """Aileron deflection angle in rads"""
        return self.deltas[0]
    
    @property
    def delta_e(self) -> float:
        """Elevator deflection angle in rads"""
        return self.deltas[1]
    
    @property
    def delta_r(self) -> float:
        """Rudder deflection angle in rads"""
        return self.deltas[2]