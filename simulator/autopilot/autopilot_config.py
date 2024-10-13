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
from simulator.environment.constants import EARTH_GRAVITY_CONSTANT as g
from simulator.math.numeric_differentiation import jacobian


@dataclass
class AutopilotConfig:

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
    ki_roll_aileron = 0.0
    kd_roll_aileron = 0.0

    ##### COURSE CONTROL WITH ROLL #####
    # design parameters
    BW_course = 20.0  # usually between 10.0 and 20.0
    zeta_course = 2.0
    # computed gains
    kp_course_roll = 0.0
    ki_course_roll = 0.0
    kd_course_roll = 0.0

    ##### SIDESLIP CONTROL WITH RUDDER #####
    # design parameters
    wn_sideslip = 10.0
    zeta_sideslip = 0.7
    # computed gains
    kp_sideslip_rudder = 0.0
    ki_sideslip_rudder = 0.0
    kd_sideslip_rudder = 0.0

    ##### YAW DAMPER WITH RUDDER #####
    kr_damper = 0.2
    pwo_damper = 0.45
    Ts_damper = None

    ##### TURN CORDINATION WITH RUDDER #####
    kp_turn_coord = 0.5
    ki_turn_coord = 0.1
    kd_turn_coord = 0.0

    ##### PITCH CONTROL WITH ELEVATOR #####
    # design parameters
    wn_pitch = 20.0
    zeta_pitch = 1.0
    # computed gains
    kp_pitch_elevator = 0.0
    ki_pitch_elevator = 0.0
    kd_pitch_elevator = 0.0

    ##### ALTITUDE CONTROL WITH PITCH #####
    # design parameters
    BW_altitude = 20.0  # usually between 10.0 and 20.0
    zeta_altitude = 1.5
    # computed gains
    kp_altitude_pitch = 0.0
    ki_altitude_pitch = 0.0
    kd_altitude_pitch = 0.0

    ##### AIRSPEED CONTROL WITH THROTTLE #####
    # design parameters
    wn_airspeed = 1.0
    zeta_airspeed = 1.0
    # computed gains
    kp_airspeed_throttle = 0.0
    ki_airspeed_throttle = 0.0
    kd_airspeed_throttle = 0.0

    ##### AIRSPEED CONTROL WITH PITCH #####
    # design parameters
    BW_airspeed2 = 5.0
    zeta_airspeed2 = 0.2
    # computed gains
    kp_airspeed_pitch = 0.0
    ki_airspeed_pitch = 0.0
    kd_airspeed_pitch = 0.0

    ##### PATH FOLLOWER PARAMETERS #####
    # target course at an infinite distance from the line path
    course_inf = np.deg2rad(90.0)
    # min aircraft turn radius (used as gain for path following)
    min_turn_radius = 100.0
    # max slope for paths between waypoints (avoid creating vertical stacked waypoints)
    max_path_slope = np.deg2rad(+60.0)
    min_path_slope = np.deg2rad(-60.0)
    # default wait orbit radius
    wait_orbit_radius = 200.0

    ##### AUTOPILOT MODE PARAMETERS #####
    # fly-by-wire mode
    fbw_max_pitch_rate = 5.0
    fbw_max_roll_rate = 5.0
    # take off mode
    take_off_throttle = 1.0  # max throttle
    take_off_pitch = np.deg2rad(5.0)  # target pitch
    take_off_altitude = 10.0  # end of take off mode
    # climb mode
    climb_throttle = 1.0  # max throttle
    climb_airspeed = 20.0  # target airspeed
    climb_altitude = 50.0  # end of climb mode
    # auto mode
    wp_default_radius = 50.0  # in meters

    def calculate_control_gains(
        self,
        params: AirframeParameters,
        state_trim: AircraftState,
        deltas_trim: ControlDeltas,
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
        self._calculate_altitude_gains(state_trim)
        self._calculate_airspeed_gains(params, state_trim, deltas_trim)
        self._calculate_sideslip_gains(params, state_trim)
        self._calculate_yaw_damper_gains(params, state_trim, deltas_trim)
        self._update_min_turn_radius(params, state_trim)
        pass

    def _calculate_roll_gains(
        self, params: AirframeParameters, state_trim: AircraftState
    ) -> None:
        Va_trim = state_trim.airspeed
        a_phi_adim = 0.5 * params.rho * Va_trim**2 * params.S * params.b
        a_phi1 = -a_phi_adim * params.Cp_p * 0.5 * params.b / Va_trim
        a_phi2 = +a_phi_adim * params.Cp_delta_a

        self.kp_roll_aileron = self.wn_roll**2 / a_phi2
        self.kd_roll_aileron = (2.0 * self.zeta_roll * self.wn_roll - a_phi1) / a_phi2

    def _calculate_pitch_gains(
        self,
        params: AirframeParameters,
        state_trim: AircraftState,
    ) -> None:
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
        Vg_trim = state_trim.groundspeed
        self.wn_course = self.wn_roll / self.BW_course

        self.kp_course_roll = 2.0 * self.zeta_course * self.wn_course * Vg_trim / g
        self.ki_course_roll = self.wn_course**2 * Vg_trim / g

    def _calculate_sideslip_gains(
        self, params: AirframeParameters, state_trim: AircraftState
    ) -> None:
        Va_trim = state_trim.airspeed
        a_beta_adim = 0.5 * params.rho * Va_trim * params.S / params.m
        a_beta1 = -a_beta_adim * params.CY_beta
        a_beta2 = +a_beta_adim * params.CY_delta_r

        self.kp_sideslip_rudder = (
            2.0 * self.zeta_sideslip * self.wn_sideslip - a_beta1
        ) / a_beta2
        self.ki_sideslip_rudder = self.wn_sideslip**2 / a_beta2

    def _calculate_altitude_gains(self, state_trim: AircraftState) -> None:
        Va_trim = state_trim.airspeed
        self.wn_altitude = self.wn_pitch / self.BW_altitude

        self.kp_altitude_pitch = (
            2.0 * self.zeta_altitude * self.wn_altitude / (self.K_theta_DC * Va_trim)
        )
        self.ki_altitude_pitch = self.wn_altitude**2 / (self.K_theta_DC * Va_trim)

    def _calculate_airspeed_gains(
        self,
        params: AirframeParameters,
        state_trim: AircraftState,
        deltas_trim: ControlDeltas,
    ) -> None:
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
        K_pitch_DC = self.kp_pitch_elevator * deltas_trim.delta_e / self.wn_pitch**2
        self.kp_airspeed_pitch = (
            a_V1 - 2.0 * self.zeta_airspeed2 * self.wn_airspeed2
        ) / (K_pitch_DC * g)
        self.ki_airspeed_pitch = -(self.wn_airspeed2**2) / (K_pitch_DC * g)

    def _update_min_turn_radius(
        self, params: AirframeParameters, state_trim: AircraftState
    ) -> None:
        R_turn = state_trim.groundspeed**2 / (g * np.tan(self.max_roll))

        if self.min_turn_radius < R_turn:
            self.min_turn_radius = R_turn

        if self.wait_orbit_radius < 2 * R_turn:
            self.wait_orbit_radius = 2 * R_turn

        self.max_turn_ratio = g * np.tan(self.max_roll) / state_trim.groundspeed

    def _calculate_yaw_damper_gains(
        self,
        params: AirframeParameters,
        state_trim: AircraftState,
        deltas_trim: ControlDeltas,
    ) -> None:
        rho_S = params.rho * params.S
        Va_trim = state_trim.airspeed
        beta_trim = state_trim.beta
        u_trim, v_trim, w_trim = state_trim.body_velocity
        p_trim, q_trim, r_trim = state_trim.angular_rates
        delta_a_trim, delta_e_trim, delta_r_trim, delta_t_trim = deltas_trim.delta
        Yv = (
            (
                (0.25 * rho_S * params.b * v_trim / (params.m * Va_trim))
                * (params.CY_p * p_trim + params.CY_r * r_trim)
            )
            + (
                (rho_S * v_trim / params.m)
                * (
                    params.CY_0
                    + params.CY_beta * beta_trim
                    + params.CY_delta_a * delta_a_trim
                    + params.CY_delta_r * delta_r_trim
                )
            )
            + (
                (0.5 * rho_S * params.CY_beta / params.m)
                * np.sqrt(u_trim**2 + w_trim**2)
            )
        )
        Yp = +w_trim + 0.25 * rho_S * Va_trim * params.b / params.m * params.CY_p
        Yr = -u_trim + 0.25 * rho_S * Va_trim * params.b / params.m * params.CY_r
        Ydelta_a = 0.5 * rho_S * Va_trim**2 / params.m * params.CY_delta_a
        Ydelta_r = 0.5 * rho_S * Va_trim**2 / params.m * params.CY_delta_r
        Lv = (
            (
                (0.25 * rho_S * params.b**2 * v_trim / Va_trim)
                * (params.Cp_p * p_trim + params.Cp_r * r_trim)
            )
            + (
                (rho_S * params.b * v_trim)
                * (
                    params.Cp_0
                    + params.Cp_beta * beta_trim
                    + params.Cp_delta_a * delta_a_trim
                    + params.Cp_delta_r * delta_r_trim
                )
            )
            + (
                (0.5 * rho_S * params.b * params.Cp_beta)
                * np.sqrt(u_trim**2 + w_trim**2)
            )
        )
        Lp = +params.Gamma1 * q_trim + (
            0.25 * rho_S * params.b**2 * Va_trim * params.Cp_p
        )
        Lr = -params.Gamma2 * q_trim + (
            0.25 * rho_S * params.b**2 * Va_trim * params.Cp_r
        )
        Ldelta_a = 0.5 * rho_S * params.b * Va_trim**2 * params.Cp_delta_a
        Ldelta_r = 0.5 * rho_S * params.b * Va_trim**2 * params.Cp_delta_r
        Nv = (
            (
                (0.25 * rho_S * params.b**2 * v_trim / Va_trim)
                * (params.Cr_p * p_trim + params.Cr_r * r_trim)
            )
            + (
                (rho_S * params.b * v_trim)
                * (
                    params.Cr_0
                    + params.Cr_beta * beta_trim
                    + params.Cr_delta_a * delta_a_trim
                    + params.Cr_delta_r * delta_r_trim
                )
            )
            + (
                (0.5 * rho_S * params.b * params.Cr_beta)
                * np.sqrt(u_trim**2 + w_trim**2)
            )
        )
        Np = +params.Gamma7 * q_trim + (
            0.25 * rho_S * params.b**2 * Va_trim * params.Cr_p
        )
        Nr = -params.Gamma1 * q_trim + (
            0.25 * rho_S * params.b**2 * Va_trim * params.Cr_r
        )
        Ndelta_a = 0.5 * rho_S * params.b * Va_trim**2 * params.Cr_delta_a
        Ndelta_r = 0.5 * rho_S * params.b * Va_trim**2 * params.Cr_delta_r

        self.dutch_roll_freq = np.sqrt(Yv * Nr - Yr * Nv)
        self.pwo_damper = self.dutch_roll_freq / 10.0
        self.kr_damper = -(Nr * Ndelta_r + Ydelta_r * Nv) / Ndelta_r**2 + np.sqrt(
            (Nr * Ndelta_r + Ydelta_r * Nv) ** 2 / Ndelta_r**4
            - (Yv**2 + Nr**2 + 2 * Yr * Nv) / Ndelta_r**2
        )
