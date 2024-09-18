"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import json
import toml
import yaml

import numpy as np

from typing import Dict
from dataclasses import dataclass, asdict


@dataclass
class AirframeParameters:
    """
    Dataclass to store airframe parameters including physical parameters, motor parameters,
    aerodynamic coefficients, and dimensionless coefficients.

    Attributes
    ----------
    m : float
        Mass of the aircraft in kilograms.
    Jx : float
        Moment of inertia about the x-axis in kg-m^2.
    Jy : float
        Moment of inertia about the y-axis in kg-m^2.
    Jz : float
        Moment of inertia about the z-axis in kg-m^2.
    Jxz : float
        Product of inertia in kg-m^2.
    S : float
        Wing area in square meters.
    b : float
        Wingspan in meters.
    c : float
        Mean aerodynamic chord in meters.
    e : float
        Oswald efficiency factor (dimensionless).
    rho : float
        Air density in kg/m^3.
    Vmax : float
        Maximum voltage for the motor in volts.
    Dprop : float
        Propeller diameter in meters.
    KV : float
        Motor velocity constant in V-s/rad.
    KQ : float
        Motor torque constant in N-m.
    Rmotor : float
        Motor resistance in Ohms.
    i0 : float
        No-load current in Amperes.
    CQ2 : float
        Quadratic coefficient for motor torque (dimensionless).
    CQ1 : float
        Linear coefficient for motor torque (dimensionless).
    CQ0 : float
        Constant term for motor torque (dimensionless).
    CT2 : float
        Quadratic coefficient for motor thrust (dimensionless).
    CT1 : float
        Linear coefficient for motor thrust (dimensionless).
    CT0 : float
        Constant term for motor thrust (dimensionless).
    alpha0 : float
        Zero-lift angle of attack in radians.
    M : float
        Lift sigmoid function transition rate.
    CL_0 : float
        Coefficient of lift at zero angle of attack.
    CL_alpha : float
        Slope of the lift coefficient versus angle of attack curve.
    CL_q : float
        Slope of the lift coefficient versus pitch rate curve.
    CL_delta_e : float
        Slope of the lift coefficient versus elevator deflection curve.
    CD_0 : float
        Coefficient of drag at zero lift.
    CD_alpha : float
        Slope of the drag coefficient versus angle of attack curve.
    CD_p : float
        Parasitic drag coefficient due to roll rate.
    CD_q : float
        Slope of the drag coefficient versus pitch rate curve.
    CD_delta_e : float
        Slope of the drag coefficient versus elevator deflection curve.
    Cm_0 : float
        Pitching moment coefficient at zero angle of attack.
    Cm_alpha : float
        Slope of the pitching moment coefficient versus angle of attack curve.
    Cm_q : float
        Slope of the pitching moment coefficient versus pitch rate curve.
    Cm_delta_e : float
        Slope of the pitching moment coefficient versus elevator deflection curve.
    CY_0 : float
        Side force coefficient at zero sideslip angle.
    CY_beta : float
        Side force coefficient due to sideslip angle.
    CY_p : float
        Side force coefficient due to roll rate.
    CY_r : float
        Side force coefficient due to yaw rate.
    CY_delta_a : float
        Side force coefficient due to aileron deflection.
    CY_delta_r : float
        Side force coefficient due to rudder deflection.
    Cl_0 : float
        Rolling moment coefficient at zero sideslip angle.
    Cl_beta : float
        Rolling moment coefficient due to sideslip angle.
    Cl_p : float
        Rolling moment coefficient due to roll rate.
    Cl_r : float
        Rolling moment coefficient due to yaw rate.
    Cl_delta_a : float
        Rolling moment coefficient due to aileron deflection.
    Cl_delta_r : float
        Rolling moment coefficient due to rudder deflection.
    Cn_0 : float
        Yawing moment coefficient at zero sideslip angle.
    Cn_beta : float
        Yawing moment coefficient due to sideslip angle.
    Cn_p : float
        Yawing moment coefficient due to roll rate.
    Cn_r : float
        Yawing moment coefficient due to yaw rate.
    Cn_delta_a : float
        Yawing moment coefficient due to aileron deflection.
    Cn_delta_r : float
        Yawing moment coefficient due to rudder deflection.
    AR : float
        Wing aspect ratio, calculated as b^2 / S.
    J : np.ndarray
        Inertia matrix of the aircraft.
    Jinv : np.ndarray
        Inverse of the inertia matrix.
    Gamma : float
        Gamma parameter for calculating aerodynamic coefficients.
    Gamma1 : float
        Gamma1 parameter for calculating aerodynamic coefficients.
    Gamma2 : float
        Gamma2 parameter for calculating aerodynamic coefficients.
    Gamma3 : float
        Gamma3 parameter for calculating aerodynamic coefficients.
    Gamma4 : float
        Gamma4 parameter for calculating aerodynamic coefficients.
    Gamma5 : float
        Gamma5 parameter for calculating aerodynamic coefficients.
    Gamma6 : float
        Gamma6 parameter for calculating aerodynamic coefficients.
    Gamma7 : float
        Gamma7 parameter for calculating aerodynamic coefficients.
    Gamma8 : float
        Gamma8 parameter for calculating aerodynamic coefficients.
    Cp_0 : float
        Coefficient for roll dynamics.
    Cp_beta : float
        Coefficient for roll dynamics due to sideslip.
    Cp_p : float
        Coefficient for roll dynamics due to roll rate.
    Cp_r : float
        Coefficient for roll dynamics due to yaw rate.
    Cp_delta_a : float
        Coefficient for roll dynamics due to aileron deflection.
    Cp_delta_r : float
        Coefficient for roll dynamics due to rudder deflection.
    Cr_0 : float
        Coefficient for yaw dynamics.
    Cr_beta : float
        Coefficient for yaw dynamics due to sideslip.
    Cr_p : float
        Coefficient for yaw dynamics due to roll rate.
    Cr_r : float
        Coefficient for yaw dynamics due to yaw rate.
    Cr_delta_a : float
        Coefficient for yaw dynamics due to aileron deflection.
    Cr_delta_r : float
        Coefficient for yaw dynamics due to rudder deflection.
    """

    # Physical parameters
    m: float = 0.0  # mass in kg
    Jx: float = 0.0  # moment of inertia about x-axis in kg-m^2
    Jy: float = 0.0  # moment of inertia about y-axis in kg-m^2
    Jz: float = 0.0  # moment of inertia about z-axis in kg-m^2
    Jxz: float = 0.0  # product of inertia in kg-m^2

    S: float = 0.0  # wing area in m^2
    b: float = 0.0  # wingspan in m
    c: float = 0.0  # mean aerodynamic chord in m
    e: float = 0.0  # Oswald efficiency factor (dimensionless)

    rho: float = 0.0  # air density in kg/m^3

    # Motor parameters
    Vmax: float = 0.0  # maximum voltage in V
    Dprop: float = 0.0  # propeller diameter in m
    KV: float = 0.0  # motor velocity constant in V-s/rad
    KQ: float = 0.0  # motor torque constant in N-m
    Rmotor: float = 0.0  # motor resistance in Ohms
    i0: float = 0.0  # no-load current in A

    CQ2: float = 0.0  # quadratic coefficient for motor torque (dimensionless)
    CQ1: float = 0.0  # linear coefficient for motor torque (dimensionless)
    CQ0: float = 0.0  # constant term for motor torque (dimensionless)

    CT2: float = 0.0  # quadratic coefficient for motor thrust (dimensionless)
    CT1: float = 0.0  # linear coefficient for motor thrust (dimensionless)
    CT0: float = 0.0  # constant term for motor thrust (dimensionless)

    # Aerodynamic coefficients
    alpha0: float = 0.0  # zero-lift angle of attack
    M: float = 0.0  # lift sigmoid function transition rate

    CL_0: float = 0.0  # coefficient of lift at zero angle of attack
    CL_alpha: float = 0.0  # slope of CL vs alpha curve
    CL_q: float = 0.0  # slope of CL vs pitch rate curve
    CL_delta_e: float = 0.0  # slope of CL vs elevator deflection curve

    CD_0: float = 0.0  # coefficient of drag at zero lift
    CD_alpha: float = 0.0  # slope of CD vs alpha curve
    CD_p: float = 0.0  # parasitic drag coefficient due to roll rate
    CD_q: float = 0.0  # slope of CD vs pitch rate curve
    CD_delta_e: float = 0.0  # slope of CD vs elevator deflection curve

    Cm_0: float = 0.0  # pitching moment coefficient at zero angle of attack
    Cm_alpha: float = 0.0  # slope of Cm vs alpha curve
    Cm_q: float = 0.0  # slope of Cm vs pitch rate curve
    Cm_delta_e: float = 0.0  # slope of Cm vs elevator deflection curve

    CY_0: float = 0.0
    CY_beta: float = 0.0  # side force coefficient due to sideslip angle
    CY_p: float = 0.0  # side force coefficient due to roll rate
    CY_r: float = 0.0  # side force coefficient due to yaw rate
    CY_delta_a: float = 0.0  # side force coefficient due to aileron deflection
    CY_delta_r: float = 0.0

    Cl_0: float = 0.0
    Cl_beta: float = 0.0  # rolling moment coefficient due to sideslip angle
    Cl_p: float = 0.0  # rolling moment coefficient due to roll rate
    Cl_r: float = 0.0  # rolling moment coefficient due to yaw rate
    Cl_delta_a: float = 0.0  # rolling moment coefficient due to aileron deflection
    Cl_delta_r: float = 0.0  # rolling moment coefficient due to rudder deflection

    Cn_0: float = 0.0
    Cn_beta: float = 0.0  # yawing moment coefficient due to sideslip angle
    Cn_p: float = 0.0  # yawing moment coefficient due to roll rate
    Cn_r: float = 0.0  # yawing moment coefficient due to yaw rate
    Cn_delta_a: float = 0.0  # yawing moment coefficient due to aileron deflection
    Cn_delta_r: float = 0.0  # yawing moment coefficient due to rudder deflection

    def __post_init__(self) -> None:
        # Calculate wing aspect ratio
        self.AR = self.b**2 / self.S
        self._calculate_inertia_matrix()
        self._calculate_gammas()
        self._calculate_Cp_coeficients()
        self._calculate_Cr_coeficients()

    def __str__(self):
        params_dict = asdict(self)
        return self._format_dict(params_dict)

    def _format_dict(self, params_dict: Dict[str, float]) -> str:
        max_key_length = max(len(key) for key in params_dict.keys())
        lines = []
        for key, value in params_dict.items():
            lines.append(f"{key.ljust(max_key_length)}: {value}")
        return "\n".join(lines)

    def _calculate_inertia_matrix(self) -> None:
        self.J = np.array(
            [
                [self.Jx, 0.0, -self.Jxz],
                [0.0, self.Jy, 0.0],
                [-self.Jxz, 0.0, self.Jz],
            ]
        )
        self.Jinv = np.linalg.inv(self.J)  # inverse inertia matrix

    def _calculate_gammas(self) -> None:
        self.Gamma = self.Jx * self.Jz - self.Jxz**2
        self.Gamma1 = (self.Jxz * (self.Jx - self.Jy + self.Jz)) / self.Gamma
        self.Gamma2 = (self.Jz * (self.Jz - self.Jy) + self.Jxz**2) / self.Gamma
        self.Gamma3 = self.Jz / self.Gamma
        self.Gamma4 = self.Jxz / self.Gamma
        self.Gamma5 = (self.Jz - self.Jx) / self.Jy
        self.Gamma6 = self.Jxz / self.Jy
        self.Gamma7 = ((self.Jx - self.Jy) * self.Jx + self.Jxz**2) / self.Gamma
        self.Gamma8 = self.Jx / self.Gamma

    def _calculate_Cp_coeficients(self) -> None:
        self.Cp_0 = self.Gamma3 * self.Cl_0 + self.Gamma4 * self.Cn_0
        self.Cp_beta = self.Gamma3 * self.Cl_beta + self.Gamma4 * self.Cn_beta
        self.Cp_p = self.Gamma3 * self.Cl_p + self.Gamma4 * self.Cn_p
        self.Cp_r = self.Gamma3 * self.Cl_r + self.Gamma4 * self.Cn_r
        self.Cp_delta_a = self.Gamma3 * self.Cl_delta_a + self.Gamma4 * self.Cn_delta_a
        self.Cp_delta_r = self.Gamma3 * self.Cl_delta_r + self.Gamma4 * self.Cn_delta_r

    def _calculate_Cr_coeficients(self) -> None:
        self.Cr_0 = self.Gamma4 * self.Cl_0 + self.Gamma8 * self.Cn_0
        self.Cr_beta = self.Gamma4 * self.Cl_beta + self.Gamma8 * self.Cn_beta
        self.Cr_p = self.Gamma4 * self.Cl_p + self.Gamma8 * self.Cn_p
        self.Cr_r = self.Gamma4 * self.Cl_r + self.Gamma8 * self.Cn_r
        self.Cr_delta_a = self.Gamma4 * self.Cl_delta_a + self.Gamma8 * self.Cn_delta_a
        self.Cr_delta_r = self.Gamma4 * self.Cl_delta_r + self.Gamma8 * self.Cn_delta_r


def load_airframe_parameters_from_json(file_path):
    """
    Load AirframeParameters data from a JSON file.

    Parameters:
    -----------
    file_path : str
        Path to the JSON file containing the parameters.

    Returns:
    --------
    AirframeParameters
        Instance of AirframeParameters populated with data from the JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return AirframeParameters(**data)


def load_airframe_parameters_from_yaml(file_path):
    """
    Load AirframeParameters data from a YAML file.

    Parameters:
    -----------
    file_path : str
        Path to the YAML file containing the parameters.

    Returns:
    --------
    AirframeParameters
        Instance of AirframeParameters populated with data from the YAML file.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return AirframeParameters(**data)


def load_airframe_parameters_from_toml(file_path):
    """
    Load AirframeParameters data from a TOML file.

    Parameters:
    -----------
    file_path : str
        Path to the TOML file containing the parameters.

    Returns:
    --------
    AirframeParameters
        Instance of AirframeParameters populated with data from the TOML file.
    """
    data = toml.load(file_path)
    return AirframeParameters(**data)
