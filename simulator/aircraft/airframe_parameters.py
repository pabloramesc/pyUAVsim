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

    Attributes:
    ----------
    TODO: complete docstring
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
    rho: float = 0.0  # air density in kg/m^3
    e: float = 0.0  # Oswald efficiency factor (dimensionless)

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
    M: float = 0.0  # lift sigmoid function transition rate
    alpha0: float = 0.0  # zero-lift angle of attack
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

        # Calculate inertia matrix
        self.J = np.array(
            [
                [self.Jx, 0.0, -self.Jxz],
                [0.0, self.Jy, 0.0],
                [-self.Jxz, 0.0, self.Jz],
            ]
        )
        self.Jinv = np.linalg.inv(self.J)  # inverse inertia matrix

        # Calculate Gammas
        self.Gamma = self.Jx * self.Jz - self.Jxz**2
        self.Gamma1 = (self.Jxz * (self.Jx - self.Jy + self.Jz)) / self.Gamma
        self.Gamma2 = (self.Jz * (self.Jz - self.Jy) + self.Jxz**2) / self.Gamma
        self.Gamma3 = self.Jz / self.Gamma
        self.Gamma4 = self.Jxz / self.Gamma
        self.Gamma5 = (self.Jz - self.Jx) / self.Jy
        self.Gamma6 = self.Jxz / self.Jy
        self.Gamma7 = ((self.Jx - self.Jy) * self.Jx + self.Jxz**2) / self.Gamma
        self.Gamma8 = self.Jx / self.Gamma

        # calculate Cp coeficients
        self.Cp_0 = self.Gamma3 * self.Cl_0 + self.Gamma4 * self.Cn_0
        self.Cp_beta = self.Gamma3 * self.Cl_beta + self.Gamma4 * self.Cn_beta
        self.Cp_p = self.Gamma3 * self.Cl_p + self.Gamma4 * self.Cn_p
        self.Cp_r = self.Gamma3 * self.Cl_r + self.Gamma4 * self.Cn_r
        self.Cp_delta_a = (
            self.Gamma3 * self.Cl_delta_a + self.Gamma4 * self.Cn_delta_a
        )
        self.Cp_delta_r = (
            self.Gamma3 * self.Cl_delta_r + self.Gamma4 * self.Cn_delta_r
        )

        # calculate Cr coeficeints
        self.Cr_0 = self.Gamma4 * self.Cl_0 + self.Gamma8 * self.Cn_0
        self.Cr_beta = self.Gamma4 * self.Cl_beta + self.Gamma8 * self.Cn_beta
        self.Cr_p = self.Gamma4 * self.Cl_p + self.Gamma8 * self.Cn_p
        self.Cr_r = self.Gamma4 * self.Cl_r + self.Gamma8 * self.Cn_r
        self.Cr_delta_a = (
            self.Gamma4 * self.Cl_delta_a + self.Gamma8 * self.Cn_delta_a
        )
        self.Cr_delta_r = (
            self.Gamma4 * self.Cl_delta_r + self.Gamma8 * self.Cn_delta_r
        )

    def __str__(self):
        params_dict = asdict(self)
        return self._format_dict(params_dict)

    def _format_dict(self, params_dict: Dict[str, float]) -> str:
        max_key_length = max(len(key) for key in params_dict.keys())
        lines = []
        for key, value in params_dict.items():
            lines.append(f"{key.ljust(max_key_length)}: {value}")
        return "\n".join(lines)


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
