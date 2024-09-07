"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.aircraft.propulsion_model import PropulsionModel
from simulator.common.constants import EARTH_GRAVITY_CONSTANT as g
from simulator.math.numeric_differentiation import jacobian


@dataclass
class AutopilotConfig:
    ##############################
    ##### CONTROL PARAMETERS #####
    ##############################

    ##### CONTROL LIMITS #####
    # aileron deflection limits (deflection angle in rads)
    max_aileron = np.deg2rad(+45.0)
    min_aileron = np.deg2rad(-45.0)
    # elevator deflection limits (deflection angle in rads)
    max_elevator = np.deg2rad(+45.0)
    min_elevator = np.deg2rad(-45.0)
    # rudder deflection limits (deflection angle in rads)
    max_rudder = np.deg2rad(+45.0)
    min_rudder = np.deg2rad(-45.0)
    # throttle thrust limits (percentage %)
    max_throttle = 1.0
    min_throttle = 0.0
    # aircraft control roll limit
    max_roll = np.deg2rad(+45.0)
    min_roll = np.deg2rad(-45.0)
    # aircraft control pitch limit
    max_pitch = np.deg2rad(+30.0)
    min_pitch = np.deg2rad(-30.0)

    ##### ROLL CONTROL WITH AILERON #####
    # design parameters
    wn_roll = 20.0
    zeta_roll = 1.0
    # computed gains
    kp_roll_aileron = 0.0
    kd_roll_aileron = 0.0

    ##### COURSE CONTROL WITH ROLL #####
    # design parameters
    BW_course = 20.0  # usually between 10.0 and 20.0
    zeta_course = 2.0
    # computed gains
    kp_course_roll = 0.0
    ki_course_roll = 0.0

    ##### SIDESLIP CONTROL WITH RUDDER #####
    # design parameters
    wn_sideslip = 20.0
    zeta_sideslip = 0.7
    # computed gains
    kp_sideslip_rudder = 0.0
    ki_sideslip_rudder = 0.0

    ##### YAW DAMPER WITH RUDDER #####
    kr_damper = 0.2
    p_wo = 0.5
    Ts_damper = None

    ##### PITCH CONTROL WITH ELEVATOR #####
    # design parameters
    wn_pitch = 20.0
    zeta_pitch = 1.0
    # computed gains
    kp_pitch_elevator = 0.0
    kd_pitch_elevator = 0.0

    ##### ALTITUDE CONTROL WITH PITCH #####
    # design parameters
    BW_altitude = 15.0  # usually between 10.0 and 20.0
    zeta_altitude = 1.5
    # computed gains
    kp_altitude_pitch = 0.0
    ki_altitude_pitch = 0.0

    ##### AIRSPEED CONTROL WITH THROTTLE #####
    # design parameters
    wn_airspeed = 1.0
    zeta_airspeed = 1.0
    # computed gains
    kp_airspeed_throttle = 0.0
    ki_airspeed_throttle = 0.0

    ##### AIRSPEED CONTROL WITH PITCH #####
    # design parameters
    BW_airspeed2 = 5.0
    zeta_airspeed2 = 0.2
    # computed gains
    kp_airspeed_pitch = 0.0
    ki_airspeed_pitch = 0.0

    ###############################
    ##### GUIDANCE PARAMETERS #####
    ###############################

    # course in the infinit for path following
    course_inf = np.deg2rad(90.0)

    # min aircraft turn radius (used as gain for path following)
    min_turn_radius = 50.0

    # max slope for paths between waypoints (avoid creating vertical stacked waypoints)
    max_path_slope = np.deg2rad(+60.0)
    min_path_slope = np.deg2rad(-60.0)

    # default wait orbit radius
    wait_orbit_radius = 100.0

    #####################################
    ##### AUTOPILOT MODE PARAMETERS #####
    #####################################

    ##### FLY-BY-WIRE MODE #####
    fbw_max_pitch_rate = 5.0
    fbw_max_roll_rate = 5.0

    ##### TAKE-OFF MODE #####
    take_off_throttle = 1.0
    take_off_pitch = np.deg2rad(5.0)
    take_off_altitude = 10.0

    ##### CLIMB MODE #####
    climb_throttle = 1.0
    climb_airspeed = 20.0
    climb_altitude = 50.0

    wp_default_radius = 10.0

    def calculate_control_gains(
        self, params: AirframeParameters, state_trim: AircraftState, deltas_trim: ControlDeltas
    ) -> None:
        """
        Calculate all control gains for the autopilot system.

        Parameters
        ----------
        params : AirframeParameters
            The aircraft's physical properties.
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        deltas_trim : ControlDeltas
            The trimmed deltas used to compute the control gains.
        """
        self._calculate_roll_gains(params, state_trim)
        self._calculate_pitch_gains(params, state_trim)
        self._calculate_course_gains(state_trim)
        self._calculate_sideslip_gains(params, state_trim)
        self._calculate_altitude_gains(state_trim)
        self._calculate_airspeed_gains(params, state_trim, deltas_trim)

    def _calculate_roll_gains(
        self, params: AirframeParameters, state_trim: AircraftState
    ) -> None:
        """
        Calculate roll control gains using ailerons.

        Parameters
        ----------
        params : AirframeParameters
            The aircraft's physical properties.
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        """
        Va_trim = state_trim.airspeed

        a_phi_adim = 0.5 * params.rho * Va_trim**2 * params.S * params.b
        a_phi1 = -a_phi_adim * params.Cp_p * 0.5 * params.b / Va_trim
        a_phi2 = +a_phi_adim * params.Cp_delta_a

        self.kp_roll_aileron = self.wn_roll**2 / a_phi2
        self.kd_roll_aileron = (2.0 * self.zeta_roll * self.wn_roll - a_phi1) / a_phi2

    def _calculate_pitch_gains(
        self, params: AirframeParameters, state_trim: AircraftState,
    ) -> None:
        """
        Calculate pitch control gains using the elevator.

        Parameters
        ----------
        params : AirframeParameters
            The aircraft's physical properties.
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        """
        Va_trim = state_trim.airspeed

        a_theta_adim = 0.5 * params.rho * Va_trim**2 * params.c * params.S / params.Jy
        a_theta1 = -a_theta_adim * params.Cm_q * 0.5 * params.c / Va_trim
        a_theta2 = -a_theta_adim * params.Cm_alpha
        a_theta3 = +a_theta_adim * params.Cm_delta_e

        self.kp_pitch_elevator = (self.wn_pitch**2 - a_theta2) / a_theta3
        self.kd_pitch_elevator = (
            2.0 * self.zeta_pitch * self.wn_pitch - a_theta1
        ) / a_theta3

        self.K_theta_DC = self.kp_pitch_elevator * a_theta3 / self.wn_pitch**2

    def _calculate_course_gains(self, state_trim: AircraftState) -> None:
        """
        Calculate course control gains using roll.

        Parameters
        ----------
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        """
        Vg_trim = state_trim.groundspeed

        self.wn_course = self.wn_roll / self.BW_course
        self.kp_course_roll = 2.0 * self.zeta_course * self.wn_course * Vg_trim / g
        self.ki_course_roll = self.wn_course**2 * Vg_trim / g

    def _calculate_sideslip_gains(
        self, params: AirframeParameters, state_trim: AircraftState
    ) -> None:
        """
        Calculate sideslip control gains using the rudder.

        Parameters
        ----------
        params : AirframeParameters
            The aircraft's physical properties.
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        """
        Va_trim = state_trim.airspeed

        a_beta_adim = 0.5 * params.rho * Va_trim * params.S / params.m
        a_beta1 = -a_beta_adim * params.CY_beta
        a_beta2 = +a_beta_adim * params.CY_delta_r

        self.kp_sideslip_rudder = (
            2.0 * self.zeta_sideslip * self.wn_sideslip - a_beta1
        ) / a_beta2
        self.ki_sideslip_rudder = self.wn_sideslip**2 / a_beta2

    def _calculate_altitude_gains(self, state_trim: AircraftState) -> None:
        """
        Calculate altitude control gains using pitch.

        Parameters
        ----------
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        """
        Va_trim = state_trim.airspeed

        self.wn_altitude = self.wn_pitch / self.BW_altitude
        
        self.kp_altitude_pitch = (
            2.0 * self.zeta_altitude * self.wn_altitude / (self.K_theta_DC * Va_trim)
        )
        self.ki_altitude_pitch = self.wn_altitude**2 / (self.K_theta_DC * Va_trim)

    def _calculate_airspeed_gains(
        self, params: AirframeParameters, state_trim: AircraftState, deltas_trim: ControlDeltas
    ) -> None:
        """
        Calculate airspeed control gains using throttle or pitch.

        Parameters
        ----------
        params : AirframeParameters
            The aircraft's physical properties.
        state_trim : AircraftState
            The trimmed state of the aircraft used to compute the control gains.
        deltas_trim : ControlDeltas
            The trimmed deltas used to compute the control gains.
        """
        Va_trim = state_trim.airspeed

        # Propulsion Model
        prop_model = PropulsionModel(params)
        prop_func = lambda x: prop_model.propulsion_force(params.Vmax * x[0], x[1])
        dTp_delta_t, dTp_Va = jacobian(
            func=prop_func, x0=np.array([deltas_trim.delta_t, Va_trim])
        )[0]

        a_V1 = (
            params.rho
            * Va_trim
            * params.S
            / params.m
            * (
                params.CD_0
                + params.CD_alpha * state_trim.alpha
                + params.CD_delta_e * deltas_trim.delta_e
            )
            - dTp_Va / params.m
        )
        a_V2 = dTp_delta_t / params.m
        a_V3 = g * np.cos(state_trim.pitch - state_trim.alpha)

        # Airspeed control with throttle
        self.kp_airspeed_throttle = (
            2.0 * self.zeta_airspeed * self.wn_airspeed - a_V1
        ) / a_V2
        self.ki_airspeed_throttle = self.wn_airspeed**2 / a_V2

        # Airspeed control with pitch
        self.wn_airspeed2 = self.wn_pitch / self.BW_airspeed2
        K_pitch_DC = (
            self.kp_pitch_elevator
            * deltas_trim.delta_e
            / self.wn_pitch**2
        )
        self.kp_airspeed_pitch = (
            a_V1 - 2.0 * self.zeta_airspeed2 * self.wn_airspeed2
        ) / (K_pitch_DC * g)
        self.ki_airspeed_pitch = -(self.wn_airspeed2**2) / (K_pitch_DC * g)
