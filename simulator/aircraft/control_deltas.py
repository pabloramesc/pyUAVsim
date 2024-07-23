"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""
import numpy as np

class ControlDeltas:

    def __init__(self, deltas0: np.ndarray = np.zeros(4)) -> None:
        """Initialize de ControlDeltas class.

        Parameters
        ----------
        deltas0 : np.ndarray, optional
            4-size array with initial control deltas array [da, de, dr, dt], by default np.zeros(4)

        ### Deltas array (4 variables)
        - da: aileron deflection angle (rads)
        - de: elevator deflection angle (rads)
        - dr: rudder deflection angle (rads)
        - dt: throttle setting between 0.0 and 1.0 (adimensional)
        """
        self.deltas = deltas0

    def update(self, deltas: np.ndarray) -> None:
        """Update the control deltas.

        Parameters
        ----------
        deltas : np.ndarray
            4-size array with initial control deltas array [da, de, dr, dt]
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
    
    @property
    def delta_t(self) -> float:
        """Throttle setting between 0.0 and 1.0"""
        return self.deltas[3]