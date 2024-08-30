"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass
import numpy as np
from numpy import sin, cos
import toml


from simulator.common.constants import DEG2RAD, EARTH_GRAVITY_CONSTANT, MSL_STANDARD_DENSITY


@dataclass
class AutopilotConfiguration:

    ##############################
    ##### CONTROL PARAMETERS #####
    ##############################

    ##### CONTROL LIMITS #####
    # elevator deflection limits (deflection angle in rads)
    max_elevator = +45.0 * DEG2RAD
    min_elevator = -45.0 * DEG2RAD
    # aileron deflection limits (deflection angle in rads)
    max_aileron = +45.0 * DEG2RAD
    min_aileron = -45.0 * DEG2RAD
    # rudder deflection limits (deflection angle in rads)
    max_rudder = +45.0 * DEG2RAD
    min_rudder = -45.0 * DEG2RAD
    # throttle thrust limits (percentage %)
    max_throttle = 1.0
    min_throttle = 0.0
    # aircraft control roll limit
    max_roll = +30.0 * DEG2RAD
    min_roll = -30.0 * DEG2RAD
    # aircraft control pitch limit
    max_pitch = +20.0 * DEG2RAD
    min_pitch = -10.0 * DEG2RAD

    ##### ROLL CONTROL WITH AILERON #####
    # design parameters
    wn_roll = 10.0
    zeta_roll = 1.0
    # computed gains
    kp_roll_aileron = 0.0
    kd_roll_aileron = 0.0

    ##### COURSE CONTROL WITH ROLL #####
    # design parameters
    BW_course = 20.0  # usually between 10.0 and 20.0
    zeta_course = 0.60
    # computed gains
    kp_course_roll = 0.0
    ki_course_roll = 0.0

    ##### SIDESLIP CONTROL WITH RUDDER #####
    # design parameters
    wn_sideslip = 24.0
    zeta_sideslip = 0.707
    # computed gains
    kp_sideslip_rudder = 0.0
    ki_sideslip_rudder = 0.0

    ##### YAW DAMPER WITH RUDDER #####
    kr_damper = 0.2
    p_wo = 0.5
    Ts_damper = None

    ##### PITCH CONTROL WITH ELEVATOR #####
    # design parameters
    wn_pitch = 10.0
    zeta_pitch = 0.72
    # computed gains
    kp_pitch_elevator = 0.0
    kd_pitch_elevator = 0.0

    ##### ALTITUDE CONTROL WITH PITCH #####
    # design parameters
    BW_altitude = 15.0  # usually between 10.0 and 20.0
    zeta_altitude = 1.75
    # computed gains
    kp_altitude_pitch = 0.0
    ki_altitude_pitch = 0.0

    ##### AIRSPEED CONTROL WITH THROTTLE #####
    # design parameters
    wn_airspeed = 25.0
    zeta_airspeed = 2.10
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

    # course in the infinit for path guidance
    course_inf = 90.0 * DEG2RAD

    # min aircraft turn radius (used as gain for guidance)
    min_turn_radius = 50.0

    # max slope for paths between waypoints (avoid creating vertical stacked waypoints)
    max_path_slope = +60.0 * DEG2RAD
    min_path_slope = -60.0 * DEG2RAD

    # default wait orbit radius
    wait_orbit_radius = 100.0

    ################################
    ##### AUTOPILOT PARAMETERS #####
    ################################

    ##### FLY-BY-WIRE MODE #####
    fbw_max_pitch_rate = 5.0
    fbw_max_roll_rate = 5.0

    ##### TAKE-OFF MODE #####
    take_off_throttle = 1.0
    take_off_pitch = 5.0 * DEG2RAD
    take_off_altitude = 10.0

    ##### CLIMB MODE #####
    climb_throttle = 1.0
    climb_airspeed = 20.0
    climb_altitude = 50.0

    waypoint_default_radius = 10.0

    def compute_control_gains(self, aircraft: object, mission: object, trim_vars: dict) -> None:
        ##### ENVIROMENT VARIABLES #####
        rho = mission.cruise.rho  # MSL_STANDARD_DENSITY
        g = mission.cruise.gravity  # EARTH_GRAVITY_CONSTANT

        #############################################
        ##### AIRCRAFT VARIABLES AND PARAMETERS #####
        #############################################

        ##### AIRFRAME PARAMETERS #####
        m = aircraft.wing.mass.mass_total  # mass of the airframe
        b = aircraft.ref.bref
        c = aircraft.ref.cref  # mean aerodynamic chord of the wing
        S = aircraft.ref.Sref  # surface area of the wing

        Jx = aircraft.wing.mass.Ixx  # inertial matrix (1,1) element
        Jy = aircraft.wing.mass.Iyy  # inertial matrix (2,2) element
        Jz = aircraft.wing.mass.Izz  # inertial matrix (3,3) element
        Jxy = aircraft.wing.mass.Ixy
        Jxz = aircraft.wing.mass.Ixz
        Jyz = aircraft.wing.mass.Iyz

        Gamma = Jx * Jz - Jxz**2
        Gamma_1 = Jxz * (Jx - Jy + Jz) / Gamma
        Gamma_2 = (Jz * (Jz - Jy) + Jxz**2) / Gamma
        Gamma_3 = Jz / Gamma
        Gamma_4 = Jxz / Gamma
        Gamma_5 = (Jz - Jx) / Jy
        Gamma_6 = Jxz / Jy
        Gamma_7 = ((Jx - Jy) * Jx + Jxz**2) / Gamma
        Gamma_8 = Jx / Gamma

        ##### AERODYNAMIC COEFICIENTS #####
        CL_0 = aircraft.general.aero.CL_0
        CL_alpha = aircraft.general.aero.CL_a
        CL_q = aircraft.general.aero.CL_q
        CL_delta_e = aircraft.general.aero.CL_deltaelev

        # TODO [PABLO]: revisar coefs CD
        CD_0 = aircraft.general.aero.CD_0
        CD_alpha = 0.0  # aircraft.general.aero.CD_a
        CD_q = 0.0  # aircraft.general.aero.CD_q
        CD_delta_e = 0.0  # aircraft.general.aero.CD_deltaelev

        Cm_0 = aircraft.general.aero.CM_0
        Cm_alpha = aircraft.general.aero.CM_a
        Cm_q = aircraft.general.aero.CM_q
        Cm_delta_e = aircraft.general.aero.CM_deltaelev
        
        Cn_0 = aircraft.general.aero.CN_0
        Cn_beta = aircraft.general.aero.CN_b
        Cn_p = aircraft.general.aero.CN_p
        Cn_r = aircraft.general.aero.CN_r
        Cn_delta_a = 0.0  # (aircraft.general.aero.CN_deltaailL - aircraft.general.aero.CN_deltaailR)/2
        Cn_delta_r = aircraft.general.aero.CN_deltarudd
        
        Cl_0 = aircraft.general.aero.CP_0
        Cl_beta = aircraft.general.aero.CP_b
        Cl_p = aircraft.general.aero.CP_p
        Cl_r = aircraft.general.aero.CP_r
        Cl_delta_a = (aircraft.general.aero.CP_deltaailL - aircraft.general.aero.CP_deltaailR)/2
        Cl_delta_r = aircraft.general.aero.CP_deltarudd

        # TODO [PABLO]: check coeficients Cp Cl Cn
        Cp_0 = Gamma_3 * Cl_0 + Gamma_4 * Cn_0
        Cp_beta = Gamma_3 * Cl_beta + Gamma_4 * Cn_beta
        Cp_p = Gamma_3 * Cl_p + Gamma_4 * Cn_p
        Cp_r = Gamma_3 * Cl_r + Gamma_4 * Cn_r
        Cp_delta_a = Gamma_3 * Cl_delta_a + Gamma_4 * Cn_delta_a
        Cp_delta_r = Gamma_3 * Cl_delta_r + Gamma_4 * Cn_delta_r

        Cr_0 = Gamma_4 * Cl_0 + Gamma_8 * Cn_0
        Cr_beta = Gamma_4 * Cl_beta + Gamma_8 * Cn_beta
        Cr_p = Gamma_4 * Cl_p + Gamma_8 * Cn_p
        Cr_r = Gamma_4 * Cl_r + Gamma_8 * Cn_r
        Cr_delta_a = Gamma_4 * Cl_delta_a + Gamma_8 * Cn_delta_a
        Cr_delta_r = Gamma_4 * Cl_delta_r + Gamma_8 * Cn_delta_r

        CY_0 = aircraft.general.aero.CY_0
        CY_beta = aircraft.general.aero.CY_b
        CY_p = aircraft.general.aero.CY_p
        CY_r = aircraft.general.aero.CY_r
        CY_delta_a = 0.0  # (aircraft.general.aero.CY_deltaailL - aircraft.general.aero.CY_deltaailR) / 2
        CY_delta_r = aircraft.general.aero.CY_deltarudd

        ##### PROPULSION PARAMETERS #####
        Tstatic = aircraft.propulsion.Tstatic
        RPMmax = aircraft.propulsion.RPMmax
        prop_pitch = aircraft.propulsion.prop_pitch
        Nmotors = aircraft.propulsion.Nmotors
        Tp = lambda delta_t, Va: Tstatic * (1.0 - Va / (prop_pitch * RPMmax / 60.0)) * Nmotors * delta_t
        dTp_delta_t = lambda delta_t, Va: Tstatic * (1.0 - Va / (prop_pitch * RPMmax / 60.0)) * Nmotors
        dTp_Va = lambda delta_t, Va: -(Tstatic * Nmotors * delta_t) / (prop_pitch * RPMmax / 60.0)

        ##### TRIM VARIABLES #####
        Vg_trim = mission.cruise.velocity
        Va_trim = mission.cruise.velocity

        p_trim, q_trim, r_trim = 0.0, 0.0, 0.0
        u_trim, v_trim, w_trim = Va_trim, 0.0, 0.0

        gamma_trim = mission.cruise.gamma
        alpha_trim = trim_vars["alpha"]
        beta_trim = 0.0

        phi_trim = 0.0
        theta_trim = alpha_trim + gamma_trim
        psi_trim = 0.0

        delta_e_trim = trim_vars["elev"]
        delta_a_trim = 0.0
        delta_t_trim = trim_vars["throttle"]
        delta_r_trim = 0.0

        ##################################################
        ##### LATERAL STATE-SPACE MODELS COEFICIENTS #####
        ##################################################
        rho_S = rho * S
        Yv = (
            0.25 * rho_S * b * v_trim / (m * Va_trim) * (CY_p * p_trim + CY_r * r_trim)
            + rho_S * v_trim / m * (CY_0 + CY_beta * beta_trim + CY_delta_a * delta_a_trim + CY_delta_r * delta_r_trim)
            + 0.5 * rho_S * CY_beta / m * np.sqrt(u_trim**2 + w_trim**2)
        )
        Yp = +w_trim + 0.25 * rho_S * Va_trim * b / m * CY_p
        Yr = -u_trim + 0.25 * rho_S * Va_trim * b / m * CY_r
        Ydelta_a = 0.5 * rho_S * Va_trim**2 / m * CY_delta_a
        Ydelta_r = 0.5 * rho_S * Va_trim**2 / m * CY_delta_r
        Lv = (
            0.25 * rho_S * b**2 * v_trim / Va_trim * (Cp_p * p_trim + Cp_r * r_trim)
            + rho_S * b * v_trim * (Cp_0 + Cp_beta * beta_trim + Cp_delta_a * delta_a_trim + Cp_delta_r * delta_r_trim)
            + 0.5 * rho_S * b * Cp_beta * np.sqrt(u_trim**2 + w_trim**2)
        )
        Lp = +Gamma_1 * q_trim + 0.25 * rho_S * b**2 * Va_trim * Cp_p
        Lr = -Gamma_2 * q_trim + 0.25 * rho_S * b**2 * Va_trim * Cp_r
        Ldelta_a = 0.5 * rho_S * b * Va_trim**2 * Cp_delta_a
        Ldelta_r = 0.5 * rho_S * b * Va_trim**2 * Cp_delta_r
        Nv = (
            0.25 * rho_S * b**2 * v_trim / Va_trim * (Cr_p * p_trim + Cr_r * r_trim)
            + rho_S * b * v_trim * (Cr_0 + Cr_beta * beta_trim + Cr_delta_a * delta_a_trim + Cr_delta_r * delta_r_trim)
            + 0.5 * rho_S * b * Cr_beta * np.sqrt(u_trim**2 + w_trim**2)
        )
        Np = +Gamma_7 * q_trim + 0.25 * rho_S * b**2 * Va_trim * Cr_p
        Nr = -Gamma_1 * q_trim + 0.25 * rho_S * b**2 * Va_trim * Cr_r
        Ndelta_a = 0.5 * rho_S * b * Va_trim**2 * Cr_delta_a
        Ndelta_r = 0.5 * rho_S * b * Va_trim**2 * Cr_delta_r

        ########################################################
        ##### LATERAL TRANSFER FUNCTION MODELS COEFICIENTS #####
        ########################################################

        ##### ROLL TRANSFER FUNCTION #####
        a_phi_adim = 0.5 * rho * Va_trim**2 * S * c
        a_phi1 = -a_phi_adim * Cp_p * 0.5 * c / Va_trim
        a_phi2 = a_phi_adim * Cp_delta_a

        ##### SIDESLIP TRANSFER FUNCTION #####
        a_beta_adim = 0.5 * rho * Va_trim * S / m
        a_beta1 = -a_beta_adim * CY_beta
        a_beta2 = a_beta_adim * CY_delta_r

        #############################################################
        ##### LONGITUDINAL TRANSFER FUNCTION MODELS COEFICIENTS #####
        #############################################################

        ##### PITCH TRANSFER FUNCTION #####
        a_theta_adim = 0.5 * rho * Va_trim**2 * c * S / Jy
        a_theta1 = -a_theta_adim * Cm_q * 0.5 * c / Va_trim
        a_theta2 = -a_theta_adim * Cm_alpha
        a_theta3 = a_theta_adim * Cm_delta_e

        ##### AIRSPEED TRANSFER FUNCTION #####
        a_V1 = (
            rho * Va_trim * S / m * (CD_0 + CD_alpha * alpha_trim + CD_delta_e * delta_e_trim)
            - dTp_Va(delta_t_trim, Va_trim) / m
        )
        a_V2 = dTp_delta_t(delta_t_trim, Va_trim) / m
        a_V3 = g * cos(theta_trim - alpha_trim)

        ###############################################
        ##### LATERAL AUTOPILOT GAINS CALCULATION #####
        ###############################################

        ##### ROLL HOLD WITH AILERON #####
        wn_roll = self.wn_roll
        zeta_roll = self.zeta_roll
        kp_roll = wn_roll**2 / a_phi2
        kd_roll = (2.0 * zeta_roll * wn_roll - a_phi1) / a_phi2
        self.kp_roll_aileron = kp_roll
        self.kd_roll_aileron = kd_roll
        if self.max_aileron > +aircraft.wing.controls.ailR.delta_max:
            self.max_aileron = +aircraft.wing.controls.ailR.delta_max
        if self.min_aileron < -aircraft.wing.controls.ailR.delta_max:
            self.min_aileron = -aircraft.wing.controls.ailR.delta_max

        ##### COURSE HOLD WITH ROLL #####
        BW_course = self.BW_course
        wn_course = wn_roll / BW_course
        zeta_course = self.zeta_course
        kp_course = 2.0 * zeta_course * wn_course * Vg_trim / g
        ki_course = wn_course**2 * Vg_trim / g
        self.kp_course_roll = kp_course
        self.ki_course_roll = ki_course

        ##### SIDESLIP HOLD WITH RUDDER #####
        wn_sideslip = self.wn_sideslip
        zeta_sideslip = self.zeta_sideslip
        kp_sideslip = (2.0 * zeta_sideslip * wn_sideslip - a_beta1) / a_beta2
        ki_sideslip = wn_sideslip**2 / a_beta2
        self.kp_sideslip_rudder = kp_sideslip
        self.ki_sideslip_rudder = ki_sideslip

        ##### YAW DAMPER WITH RUDDER #####
        # TODO [PABLO]: implement yaw damper params calculation
        # wn_damper = np.sqrt(Yv * Nr - Yr * Nv)
        # self.p_wo = wn_damper / 10.0
        # self.kr_damper = -(Nr * Ndelta_r + Ydelta_r * Nv) / Ndelta_r**2 + np.sqrt(
        #     ((Nr * Ndelta_r + Ydelta_r * Nv) / Ndelta_r**2) ** 2
        #     - ((Yv**2 + Nr**2 + 2.0 * Yr * Nv) / Ndelta_r**2)
        # )
        if self.max_rudder > +aircraft.VTP.controls.rudd.delta_max:
            self.max_rudder = +aircraft.VTP.controls.rudd.delta_max
        if self.min_rudder < -aircraft.VTP.controls.rudd.delta_max:
            self.min_rudder = -aircraft.VTP.controls.rudd.delta_max

        ####################################################
        ##### LONGITUDINAL AUTOPILOT GAINS CALCULATION #####
        ####################################################

        ##### PITCH HOLD WITH ELEVATOR #####
        wn_pitch = self.wn_pitch
        zeta_pitch = self.zeta_pitch
        kp_pitch = (wn_pitch**2 - a_theta2) / a_theta3
        kd_pitch = (2.0 * zeta_pitch * wn_pitch - a_theta1) / a_theta3
        K_pitch_DC = kp_pitch * a_theta3 / wn_pitch**2
        self.kp_pitch_elevator = kp_pitch
        self.kd_pitch_elevator = kd_pitch
        if self.max_elevator > +aircraft.tail.controls.elev.delta_max:
            self.max_elevator = +aircraft.tail.controls.elev.delta_max
        if self.min_elevator < -aircraft.tail.controls.elev.delta_max:
            self.min_elevator = -aircraft.tail.controls.elev.delta_max

        ##### ALTITUDE HOLD WITH PITCH #####
        BW_altitude = self.BW_altitude  # usually between 10.0 and 20.0
        wn_altitude = wn_pitch / BW_altitude
        zeta_altitude = self.zeta_altitude
        kp_altitude = 2.0 * zeta_altitude * wn_altitude / (K_pitch_DC * Va_trim)
        ki_altitude = wn_altitude**2 / (K_pitch_DC * Va_trim)
        self.kp_altitude_pitch = kp_altitude
        self.ki_altitude_pitch = ki_altitude

        ##### AIRSPEED HOLD WITH THROTTLE #####
        wn_airspeed = self.wn_airspeed
        zeta_airspeed = self.zeta_airspeed
        kp_airspeed = (2.0 * zeta_airspeed * wn_airspeed - a_V1) / a_V2
        ki_airspeed = wn_airspeed**2 / a_V2
        self.kp_airspeed_throttle = kp_airspeed
        self.ki_airspeed_throttle = ki_airspeed

        ##### AIRSPEED HOLD WITH PITCH #####
        BW_airspeed2 = self.BW_airspeed2
        wn_airspeed2 = wn_pitch / BW_airspeed2
        zeta_airspeed2 = self.zeta_airspeed2
        kp_airspeed2 = (a_V1 - 2.0 * zeta_airspeed2 * wn_airspeed2) / (K_pitch_DC * g)
        ki_airspeed2 = -(wn_airspeed2**2) / (K_pitch_DC * g)
        self.kp_airspeed_pitch = kp_airspeed2
        self.ki_airspeed_pitch = ki_airspeed2

    def set_params(self, **kwargs) -> None:
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

    def load_config_from_toml(self, file_name: str) -> None:
        config = toml.load(file_name)

        valid_sections = ("CONTROL", "GUIDANCE", "AUTOPILOT")
        angular_params_keys = (
            "max_elevator",
            "min_elevator",
            "max_aileron",
            "min_aileron",
            "max_rudder",
            "min_rudder",
            "max_roll",
            "min_roll",
            "max_pitch",
            "min_pitch",
            "course_inf",
            "fbw_max_pitch_rate",
            "fbw_max_roll_rate",
            "take_off_pitch",
        )

        for section_name, section_params in config.items():
            if section_name in valid_sections:
                for key, value in section_params.items():
                    if key in angular_params_keys:
                        value *= DEG2RAD
                    setattr(self, key, value)
