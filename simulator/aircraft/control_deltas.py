"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.math.angles import wrap_angle_pi


class ControlDeltas:

    def __init__(self, delta0: np.ndarray = None, max_angle: float = np.pi / 2) -> None:
        """
        Initialize de ControlDeltas class.

        Parameters
        ----------
        delta0 : np.ndarray, optional
            4-size array with initial control deltas array [da, de, dr, dt], by default None
        max_angle : float, optional
            Maximum deflection angle (absolute value) for control surfaces, by default np.pi/2

        ### Deltas array (4 variables)
        - da: aileron deflection angle (rads)
        - de: elevator deflection angle (rads)
        - dr: rudder deflection angle (rads)
        - dt: throttle setting between 0.0 and 1.0 (adimensional)
        """
        if delta0 is None:
            self.delta = np.zeros(4)
        else:
            self._check_delta(delta0)
            self.delta = np.copy(delta0)
        self.max_angle = max_angle

    def update(self, delta: np.ndarray) -> None:
        """
        Update the control deltas.

        Parameters
        ----------
        delta : np.ndarray
            4-size array with initial control deltas array [da, de, dr, dt]
        """
        self._check_delta(delta)
        self.delta = np.copy(delta)

    @property
    def delta_a(self) -> float:
        """Aileron deflection angle in rads"""
        return np.clip(self.delta[0], -self.max_angle, +self.max_angle)

    @property
    def delta_e(self) -> float:
        """Elevator deflection angle in rads"""
        return np.clip(self.delta[1], -self.max_angle, +self.max_angle)

    @property
    def delta_r(self) -> float:
        """Rudder deflection angle in rads"""
        return np.clip(self.delta[2], -self.max_angle, +self.max_angle)

    @property
    def delta_t(self) -> float:
        """Throttle setting between 0.0 and 1.0"""
        return np.clip(self.delta[3], 0.0, 1.0)

    @delta_a.setter
    def delta_a(self, value: float) -> None:
        self.delta[0] = np.clip(value, -self.max_angle, +self.max_angle)

    @delta_e.setter
    def delta_e(self, value: float) -> None:
        self.delta[1] = np.clip(value, -self.max_angle, +self.max_angle)

    @delta_r.setter
    def delta_r(self, value: float) -> None:
        self.delta[2] = np.clip(value, -self.max_angle, +self.max_angle)

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        self.delta[3] = np.clip(value, 0.0, 1.0)

    def _check_delta(self, delta: np.ndarray) -> None:
        if not isinstance(delta, np.ndarray):
            raise ValueError("type must be a numpy array!")
        if delta.shape != (4,):
            raise ValueError("shape must be (4,)")

    def __str__(self) -> str:
        """
        Return a string representation of the control deltas.

        Returns
        -------
        str
            A string representation of the control deltas
        """
        return (
            f"Control Deltas:\n"
            f"- Aileron (delta_a)    : {self.delta_a:.2f} rad\n"
            f"- Elevator (delta_e)   : {self.delta_e:.2f} rad\n"
            f"- Rudder (delta_r)     : {self.delta_r:.2f} rad\n"
            f"- Throttle (delta_t)   : {self.delta_t:.2f} (0.0 to 1.0)"
        )
