"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.control_surfaces import ControlSurfaces
from simulator.aircraft.airframe_parameters import AirframeParameters

class Aerodynamics:

    def __init__(self, params: AirframeParameters) -> None:
        self.params = params

        self.lift = 0.0
        self.drag = 0.0
        self.moment = 0.0

    def calculate_lift(self, state: AircraftState, deltas:ControlSurfaces) -> float:
        CL_at_alpha = self.lift_coefficient_at_alpha(state.alpha)
        CL_at_q = self.params.CL_q * (0.5 * self.params.c / state.airspeed) * state.q
        CL_at_sigma_e = self.params.CL_delta_e * deltas.delta_e
        f_lift = 0.5 * self.params.rho * state.airspeed**2 * self.params.S * (CL_at_alpha + CL_at_q + CL_at_sigma_e)
        return f_lift

    def calculate_drag(self, state: AircraftState, deltas:ControlSurfaces) -> float:
        CD_at_alpha = self.drag_coefficient_at_alpha(state.alpha)
        CD_at_q = self.params.CD_q * (0.5 * self.params.c / state.airspeed) * state.q
        CD_at_sigma_e = self.params.CD_delta_e * deltas.delta_e
        f_drag = 0.5 * self.params.rho * state.airspeed**2 * self.params.S * (CD_at_alpha + CD_at_q + CD_at_sigma_e)
        return f_drag

    def calculate_moment(self) -> float:
        pass

    def lift_coefficient_at_alpha(self, alpha: float, model: str = "accurate") -> float:
        CL_at_alpha = 0.0
        if model == "accurate":
            pass
        elif model == "linear":
            ar = self.params.b**2 / self.params.S # wing aspect ratio
            CL_alpha = (np.pi * ar) / (1.0 + np.sqrt(1.0 + (0.5 * ar)**2))
            CL_at_alpha = self.params.CL0 + CL_alpha * alpha
        else:
            raise ValueError("Model parameter must be 'accurate' or 'linear'!")
        return CL_at_alpha
    
    def drag_coefficient_at_alpha(self, alpha: float) -> float:
        CD_at_alpha = 0.0
        return CD_at_alpha