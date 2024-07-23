"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.aerodynamics import Aerodynamics
from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.aircraft.dynamics import Dynamics
from simulator.common.constants import EARTH_GRAVITY_VECTOR


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
        state0 : np.ndarray, optional
            Initial state array (12 variables: pn, pe, pd, u, v, w, roll, pitch, yaw, p, q, r),
            by default np.zeros(12)
        wind : np.ndarray, optional
            Wind vector in NED frame (3-size array: wn, we, wd in m/s), by default np.zeros(3)
        """
        self.t = 0.0
        self.dt = dt

        self.wind = wind
        self.grav = EARTH_GRAVITY_VECTOR  # earth's gravity field vector in NED frame

        self.params = params
        self.state = AircraftState(state0, wind)
        self.deltas = ControlDeltas(deltas0)
        self.dynamics = Dynamics(dt, params, self.state, wind)
        self.aerodynamics = Aerodynamics(params)

        self.forces = np.zeros(3)
        self.moments = np.zeros(3)

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
        new_state = self.dynamics.dynamics(self.forces, self.moments)
        self.state.update(new_state)

    def update_forces_and_moments(self) -> None:
        f_grav = self.state.R_vb * self.grav  # gravity force

        f_lift = self.aerodynamics.lift_force(self.state, self.deltas)
        f_drag = self.aerodynamics.drag_force(self.state, self.deltas)
        f_lat = self.aerodynamics.lateral_force(self.state, self.deltas)
        f_aero = self.state.R_wb * np.array([-f_drag, f_lat, -f_lift]) # aerodynamic forces
        
        f_prop = self.aerodynamics.motor_force(self.state, self.deltas) # motor force
        
        self.forces = f_grav + f_aero + f_prop

        # m_pitch = self.aerodynamics.pitch_moment(self.state, self.deltas)
        # m_roll = self.aerodynamics.roll_moment(self.)

        # TODO: complete force and moments calculation