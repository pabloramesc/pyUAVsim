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
    ### Physical parameters
    m : float
        Mass of the aircraft in kilograms (kg).
    Jx : float
        Moment of inertia about the x-axis in kilograms-square meters (kg-m^2).
    Jy : float
        Moment of inertia about the y-axis in kilograms-square meters (kg-m^2).
    Jz : float
        Moment of inertia about the z-axis in kilograms-square meters (kg-m^2).
    Jxz : float
        Product of inertia in kilograms-square meters (kg-m^2).
    S : float
        Wing area in square meters (m^2).
    b : float
        Wingspan of the aircraft in meters (m).
    c : float
        Mean aerodynamic chord in meters (m).
    rho : float
        Air density in kilograms per cubic meter (kg/m^3).
    e : float
        Oswald efficiency factor (dimensionless).

    ### Motor parameters
    Vmax : float
        Maximum voltage of the motor in volts (V).
    Dprop : float
        Propeller diameter in meters (m).
    KV : float
        Motor velocity constant in volt-seconds per radian (V-s/rad).
    KQ : float
        Motor torque constant in Newton-meters (N-m).
    Rmotor : float
        Motor resistance in ohms (Ω).
    i0 : float
        No-load current of the motor in amperes (A).
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

    ### Aerodynamic coefficients
    CL0 : float
        Coefficient of lift at zero angle of attack (dimensionless).
    CD0 : float
        Coefficient of drag at zero lift (dimensionless).
    Cm0 : float
        Pitching moment coefficient at zero angle of attack (dimensionless).
    CL_alpha : float
        Slope of lift coefficient vs. angle of attack curve (dimensionless).
    CD_alpha : float
        Slope of drag coefficient vs. angle of attack curve (dimensionless).
    Cm_alpha : float
        Slope of pitching moment coefficient vs. angle of attack curve (dimensionless).
    CL_q : float
        Slope of lift coefficient vs. pitch rate curve (dimensionless).
    CD_q : float
        Slope of drag coefficient vs. pitch rate curve (dimensionless).
    Cm_q : float
        Slope of pitching moment coefficient vs. pitch rate curve (dimensionless).
    CL_delta_e : float
        Slope of lift coefficient vs. elevator deflection curve (dimensionless).
    CD_delta_e : float
        Slope of drag coefficient vs. elevator deflection curve (dimensionless).
    Cm_delta_e : float
        Slope of pitching moment coefficient vs. elevator deflection curve (dimensionless).
    M : float
        Lift sigmoid function transition rate (dimensionless).
    alpha0 : float
        Zero-lift angle of attack in radians (rad).
    CD_p : float
        Parasitic drag coefficient due to roll rate (dimensionless).
    CY_beta : float
        Side force coefficient due to sideslip angle (dimensionless).
    Cl_beta : float
        Rolling moment coefficient due to sideslip angle (dimensionless).
    Cn_beta : float
        Yawing moment coefficient due to sideslip angle (dimensionless).
    CY_p : float
        Side force coefficient due to roll rate (dimensionless).
    Cl_p : float
        Rolling moment coefficient due to roll rate (dimensionless).
    Cn_p : float
        Yawing moment coefficient due to roll rate (dimensionless).
    CY_r : float
        Side force coefficient due to yaw rate (dimensionless).
    Cl_r : float
        Rolling moment coefficient due to yaw rate (dimensionless).
    Cn_r : float
        Yawing moment coefficient due to yaw rate (dimensionless).
    CY_delta_a : float
        Side force coefficient due to aileron deflection (dimensionless).
    Cl_delta_a : float
        Rolling moment coefficient due to aileron deflection (dimensionless).
    Cn_delta_a : float
        Yawing moment coefficient due to aileron deflection (dimensionless).
    Cl_delta_r : float
        Rolling moment coefficient due to rudder deflection (dimensionless).
    Cn_delta_r : float
        Yawing moment coefficient due to rudder deflection (dimensionless).
    """

    # Physical parameters
    m: float  # mass in kg
    Jx: float  # moment of inertia about x-axis in kg-m^2
    Jy: float  # moment of inertia about y-axis in kg-m^2
    Jz: float  # moment of inertia about z-axis in kg-m^2
    Jxz: float  # product of inertia in kg-m^2
    S: float  # wing area in m^2
    b: float  # wingspan in m
    c: float  # mean aerodynamic chord in m
    rho: float  # air density in kg/m^3
    e: float  # Oswald efficiency factor (dimensionless)

    # Motor parameters
    Vmax: float  # maximum voltage in V
    Dprop: float  # propeller diameter in m
    KV: float  # motor velocity constant in V-s/rad
    KQ: float  # motor torque constant in N-m
    Rmotor: float  # motor resistance in Ohms
    i0: float  # no-load current in A
    CQ2: float  # quadratic coefficient for motor torque (dimensionless)
    CQ1: float  # linear coefficient for motor torque (dimensionless)
    CQ0: float  # constant term for motor torque (dimensionless)
    CT2: float  # quadratic coefficient for motor thrust (dimensionless)
    CT1: float  # linear coefficient for motor thrust (dimensionless)
    CT0: float  # constant term for motor thrust (dimensionless)

    # Aerodynamic coefficients
    CL0: float  # coefficient of lift at zero angle of attack
    CL_alpha: float  # slope of CL vs alpha curve
    CL_q: float  # slope of CL vs pitch rate curve
    CL_delta_e: float  # slope of CL vs elevator deflection curve
    CD0: float  # coefficient of drag at zero lift
    CD_alpha: float  # slope of CD vs alpha curve
    CD_p: float  # parasitic drag coefficient due to roll rate
    CD_q: float  # slope of CD vs pitch rate curve
    CD_delta_e: float  # slope of CD vs elevator deflection curve
    Cm0: float  # pitching moment coefficient at zero angle of attack
    Cm_alpha: float  # slope of Cm vs alpha curve
    Cm_q: float  # slope of Cm vs pitch rate curve
    Cm_delta_e: float  # slope of Cm vs elevator deflection curve
    M: float  # lift sigmoid function transition rate
    alpha0: float  # zero-lift angle of attack
    CY_0: float = 0.0
    CY_beta: float  # side force coefficient due to sideslip angle
    CY_p: float  # side force coefficient due to roll rate
    CY_r: float  # side force coefficient due to yaw rate
    CY_delta_a: float  # side force coefficient due to aileron deflection
    CY_delta_r: float = 0.0
    Cl_0: float = 0.0
    Cl_beta: float  # rolling moment coefficient due to sideslip angle
    Cl_p: float  # rolling moment coefficient due to roll rate
    Cl_r: float  # rolling moment coefficient due to yaw rate
    Cl_delta_a: float  # rolling moment coefficient due to aileron deflection
    Cl_delta_r: float  # rolling moment coefficient due to rudder deflection
    Cn_0: float = 0.0
    Cn_beta: float  # yawing moment coefficient due to sideslip angle
    Cn_p: float  # yawing moment coefficient due to roll rate
    Cn_r: float  # yawing moment coefficient due to yaw rate
    Cn_delta_a: float  # yawing moment coefficient due to aileron deflection
    Cn_delta_r: float  # yawing moment coefficient due to rudder deflection

    def __post_init__(self) -> None:
        # Calculate wing aspect ratio
        self.AR = self.b**2 / self.S  

        #Calculate inertia matrix
        self.J = np.array(
            [
                [self.Jx, 0.0, -self.Jxz],
                [0.0, self.Jy, 0.0],
                [-self.Jxz, 0.0, self.Jz],
            ]
        )
        self.Jinv = np.linalg.inv(self.J) # inverse inertia matrix

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
