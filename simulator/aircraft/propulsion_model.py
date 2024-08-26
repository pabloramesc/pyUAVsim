"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas


class PropulsionModel:

    def __init__(self, params: AirframeParameters) -> None:
        """Initialize the ForcesMoments class.

        Parameters
        ----------
        params : AirframeParameters
            Instance of AirframeParameters containing the aircraft's properties
        """
        self.params = params

    def calculate_forces_moments(
        self, state: AircraftState, deltas: ControlDeltas
    ) -> np.ndarray:
        """Calcuate propulsion forces and moments acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
            Current aircraft's state
        deltas : ControlDeltas
            Current cotrol deltas

        Returns
        -------
        ndarray
            Calculated external forces and moments array in body frame: [fx, fy, fx, l, m, n]
        """
        Vin = self.params.Vmax * deltas.delta_t
        
        # propulsion forces in body frame
        T_prop = self.propulsion_force(Vin, state.airspeed)

        # propulsion moments
        Q_prop = self.propulsion_moment(Vin, state.airspeed)

        # total forces and moments
        u = np.zeros(6)
        u[0:3] = np.array([T_prop, 0.0, 0.0])
        u[3:6] = np.array([-Q_prop, 0.0, 0.0])
        return u

    def propeller_speed(self, Vin: float, Va: float) -> float:
        """Calculate the propeller speed using the DC motor model.

        Parameters
        ----------
        Vin : float
            Motor voltage in V
        Va : float
            Aicraft's current airspeed in m/s

        Returns
        -------
        float
            The propeller speed in radians per second (rad/s)
        """
        a = (
            self.params.rho
            * self.params.Dprop**5
            / (2.0 * np.pi) ** 2
            * self.params.CQ0
        )  # a = (rho D^5 / (2 pi)^2) CQ0
        b = (
            self.params.rho
            * self.params.Dprop**4
            / (2.0 * np.pi)
            * self.params.CQ1
            * Va
            + self.params.KQ * self.params.KV / self.params.Rmotor
        )  # b = (rho D^4 / (2 pi)) CQ1 Va + (KQ KV / R)
        c = (
            self.params.rho * self.params.Dprop**3 * self.params.CQ2 * Va**2
            - self.params.KQ / self.params.Rmotor * Vin
            + self.params.KQ * self.params.i0
        )  # c = (rho D^3) Cq2 Va^2 - (KQ / R) Vin + KQ i0
        Omega = (-b + np.sqrt(b**2 - 4 * a * c)) / (2.0 * a)
        return Omega

    def propulsion_force(self, Vin: float, Va: float) -> float:
        """Calculate the motor force acting on the aircraft.

        Parameters
        ----------
        Vin : float
            Motor voltage in V
        Va : float
            Aicraft's current airspeed in m/s
            
        Returns
        -------
        float
            The motor force acting on the aircraft in newtons (N)
        """
        Omega = self.propeller_speed(Vin, Va)
        Jprop = (
            2.0 * np.pi * Va / (Omega * self.params.Dprop)
        )  # advance ratio
        CT_vs_J = self.params.CT0 + self.params.CT1 * Jprop + self.params.CT2 * Jprop**2
        Tp = (
            self.params.rho
            * (0.5 * Omega / np.pi) ** 2
            * self.params.Dprop**4
            * CT_vs_J
        )
        return Tp  # motor thrust

    def propulsion_moment(self, Vin: float, Va: float) -> float:
        """Calculate the motor moment acting on the aircraft.

        Parameters
        ----------
        Vin : float
            Motor voltage in V
        Va : float
            Aicraft's current airspeed in m/s

        Returns
        -------
        float
            The motor moment acting on the aircraft in newton-meters (Nm)
        """
        Omega = self.propeller_speed(Vin, Va)
        Jprop = (
            2.0 * np.pi * Va / (Omega * self.params.Dprop)
        )  # advance ratio
        CQ_vs_J = self.params.CQ0 + self.params.CQ1 * Jprop + self.params.CQ2 * Jprop**2
        Qp = (
            self.params.rho
            * (0.5 * Omega / np.pi) ** 2
            * self.params.Dprop**5
            * CQ_vs_J
        )
        return Qp  # motor torque
