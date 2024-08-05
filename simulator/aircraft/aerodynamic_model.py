"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas


class AerodynamicModel:

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
        """Calcuate aerodynamic forces and moments acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
            Current aircraft's state
        deltas : ControlDeltas
            Current cotrol deltas

        Returns
        -------
        ndarray
            External forces and moments array in body frame: [fx, fy, fx, l, m, n]
        """

        # aerodynamic forces (lift, drag and lateral force) in body frame
        F_lift = self.lift_force(
            state, deltas
        )  # lift force is expressed in the stability frame
        F_drag = self.drag_force(
            state, deltas
        )  # drag force is expressed in the stability frame
        F_lat = self.lateral_force(
            state, deltas
        )  # lateral force is expressed in body frame

        # aerodynamic moments (pitch, roll and yaw moments)
        M_pitch = self.pitch_moment(state, deltas)
        M_roll = self.roll_moment(state, deltas)
        M_yaw = self.yaw_moment(state, deltas)

        # total forces and moments in body frame
        u = np.zeros(6)
        u[0:3] = state.R_sb @ np.array([-F_drag, 0.0, -F_lift]) + np.array(
            [0.0, F_lat, 0.0]
        )
        u[3:6] = np.array([M_roll, M_pitch, M_yaw])
        return u

    def lift_coefficient_vs_alpha(self, alpha: float, model: str = "accurate") -> float:
        """Calculate the lift coefficient as a function of angle of attack.

        Parameters
        ----------
        alpha : float
            Angle of attack in rads
        model : str, optional
            Model to use for calculation ("accurate" or "linear"), by default "accurate"

        Returns
        -------
        float
            Lift coefficient at the given angle of attack (adimensional)

        Raises
        ------
        ValueError
            If the model parameter is not "accurate" or "linear"
        """
        CL_vs_alpha = 0.0
        if model == "accurate":
            exp_neg = np.exp(-self.params.M * (alpha - self.params.alpha0))
            exp_pos = np.exp(+self.params.M * (alpha + self.params.alpha0))
            sigma = (1.0 + exp_neg + exp_pos) / ((1.0 + exp_neg) * (1.0 + exp_pos))
            CL_linear = self.params.CL0 + self.params.CL_alpha * alpha
            CL_stall = 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
            CL_vs_alpha = (1.0 - sigma) * CL_linear + sigma * CL_stall
        elif model == "linear":
            # CL_alpha = (np.pi * self.params.AR) / (
            #     1.0 + np.sqrt(1.0 + (0.5 * self.params.AR) ** 2)
            # )  # CL_alpha aproximation for small aircrafts
            # CL_vs_alpha = self.params.CL0 + CL_alpha * alpha
            CL_vs_alpha = self.params.CL0 + self.params.CL_alpha * alpha
        else:
            raise ValueError("Model parameter must be 'accurate' or 'linear'!")
        return CL_vs_alpha

    def lift_force(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the lift force acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The lift force acting on the aircraft in newtons (N)
        """
        CL_vs_alpha = self.lift_coefficient_vs_alpha(state.alpha)
        CL_vs_q = self.params.CL_q * (0.5 * self.params.c / state.airspeed) * state.q
        CL_vs_sigma_e = self.params.CL_delta_e * deltas.delta_e
        F_lift = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * (CL_vs_alpha + CL_vs_q + CL_vs_sigma_e)
        )  # F_lift = 1/2 rho Va^2 S CL(alpha, q, delta_e)
        return F_lift

    def drag_coefficient_vs_alpha(
        self, alpha: float, model: str = "quadratic"
    ) -> float:
        """Calculate the drag coefficient as a function of angle of attack

        Parameters
        ----------
        alpha : float
            Angle of attack in rads
        model : str, optional
            Model to use for calculation ("quadratic" or "linear"), by default "quadratic"

        Returns
        -------
        float
            Drag coefficient at the given angle of attack (adimensional)

        Raises
        ------
        ValueError
            If the model parameter is not "quadratic" or "linear"
        """
        CD_vs_alpha = 0.0
        if model == "quadratic":
            CL_vs_alpha = self.lift_coefficient_vs_alpha(alpha, "linear")
            CD_vs_alpha = self.params.CD_p + CL_vs_alpha**2 / (
                np.pi * self.params.e * self.params.AR
            )
        elif model == "linear":
            CD_vs_alpha = self.params.CD0 + self.params.CD_alpha * alpha
        else:
            raise ValueError("Model parameter must be 'quadratic' or 'linear'!")
        return CD_vs_alpha

    def drag_force(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the drag force acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The drag force acting on the aircraft in newtons (N)
        """
        CD_vs_alpha = self.drag_coefficient_vs_alpha(state.alpha)
        CD_vs_q = self.params.CD_q * (0.5 * self.params.c / state.airspeed) * state.q
        CD_vs_sigma_e = self.params.CD_delta_e * deltas.delta_e
        F_drag = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * (CD_vs_alpha + CD_vs_q + CD_vs_sigma_e)
        )  # F_drag = 1/2 rho Va^2 S CD(alpha, q, delta_e)
        return F_drag

    def pitch_moment(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the pitching moment acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
             The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The pitching moment acting on the aircraft in newton-meters (Nm)
        """
        Cm_vs_alpha = self.params.Cm0 + self.params.Cm_alpha * state.alpha
        Cm_vs_q = self.params.Cm_q * (0.5 * self.params.c / state.airspeed) * state.q
        Cm_vs_sigma_e = self.params.Cm_delta_e * deltas.delta_e
        m = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * self.params.c
            * (Cm_vs_alpha + Cm_vs_q + Cm_vs_sigma_e)
        )  # m = 1/2 rho Va^2 S c Cm(alpha, q, delta_e)
        return m

    def lateral_force(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the lateral force acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
             The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The lateral force acting on the aircraft in newtons (N)
        """
        # CY_vs_beta = (
        #     self.params.CY0 + self.params.CY_beta * state.beta
        # )  # CY0 = 0 for symmetrical aircrafts in XZ plane
        CY_vs_beta = self.params.CY_beta * state.beta
        CY_vs_p = self.params.CY_p * (0.5 * self.params.b / state.airspeed) * state.p
        CY_vs_r = self.params.CY_r * (0.5 * self.params.b / state.airspeed) * state.r
        CY_vs_delta_a = self.params.CY_delta_a * deltas.delta_a
        # CY_vs_delta_r = self.params.CY_delta_r * deltas.delta_r
        fy = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * (CY_vs_beta + CY_vs_p + CY_vs_r + CY_vs_delta_a)
        )  # fy = 1/2 rho Va**2 S CY(beta, p, r, delta_a, delta_r)
        return fy

    def roll_moment(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the roll moment acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
             The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The roll moment acting on the aircraft in newton-meters (Nm)
        """
        # Cl_vs_beta = (
        #     self.params.Cl0 + self.params.Cl_beta * state.beta
        # )  # Cl0 = 0 for symmetrical aircrafts in XZ plane
        Cl_vs_beta = self.params.Cl_beta * state.beta
        Cl_vs_p = self.params.Cl_p * (0.5 * self.params.b / state.airspeed) * state.p
        Cl_vs_r = self.params.Cl_r * (0.5 * self.params.b / state.airspeed) * state.r
        Cl_vs_delta_a = self.params.Cl_delta_a * deltas.delta_a
        Cl_vs_delta_r = self.params.Cl_delta_r * deltas.delta_r
        fy = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * self.params.b
            * (Cl_vs_beta + Cl_vs_p + Cl_vs_r + Cl_vs_delta_a + Cl_vs_delta_r)
        )  # fy = 1/2 * rho * Va**2 * S * b * Cl(beta, p, r, delta_a, delta_r)
        return fy

    def yaw_moment(self, state: AircraftState, deltas: ControlDeltas) -> float:
        """Calculate the yaw moment acting on the aircraft.

        Parameters
        ----------
        state : AircraftState
             The current state of the aircraft
        deltas : ControlSurfaces
            The current deflections of the aircraft's control surfaces

        Returns
        -------
        float
            The yaw moment acting on the aircraft in newton-meters (Nm)
        """
        # Cn_vs_beta = (
        #     self.params.Cn0 + self.params.Cn_beta * state.beta
        # )  # Cn0 = 0 for symmetrical aircrafts in XZ plane
        Cn_vs_beta = self.params.Cn_beta * state.beta
        Cn_vs_p = self.params.Cn_p * (0.5 * self.params.b / state.airspeed) * state.p
        Cn_vs_r = self.params.Cn_r * (0.5 * self.params.b / state.airspeed) * state.r
        Cn_vs_delta_a = self.params.Cn_delta_a * deltas.delta_a
        Cn_vs_delta_r = self.params.Cn_delta_r * deltas.delta_r
        fy = (
            0.5
            * self.params.rho
            * state.airspeed**2
            * self.params.S
            * self.params.b
            * (Cn_vs_beta + Cn_vs_p + Cn_vs_r + Cn_vs_delta_a + Cn_vs_delta_r)
        )  # fy = 1/2 rho Va**2 S b Cn(beta, p, r, delta_a, delta_r)
        return fy
