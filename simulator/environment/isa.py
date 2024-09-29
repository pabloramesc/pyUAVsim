"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.environment.constants import EARTH_GRAVITY_CONSTANT, FT2M, KELVIN0CELSIUS
from simulator.environment.isa_constants import *


def _check_altitude(alt: float):
    """
    Check if the altitude is within the supported ISA range.
    """
    if not (ISA_MIN_ALTITUDE <= alt <= ISA_MAX_ALTITUDE):
        raise ValueError(
            f"Altitude {alt} not supported. "
            f"ISA model supports altitudes between {ISA_MIN_ALTITUDE} and {ISA_MAX_ALTITUDE} meters."
        )


def isa_temperature(alt: float, celsius: bool = False) -> float:
    """
    Compute ISA temperature at a given altitude.

    Parameters
    ----------
    alt : float
        Altitude above MSL (mean sea level) in meters
    celsius : bool, optional
        Return temperature in Celsius, by default False (returns in Kelvin)

    Returns
    -------
    float
        Temperature in K (or °C if `celsius` is True)

    Raises
    ------
    ValueError
        If altitude is outside the supported range (-6 to 86 km)
    """
    _check_altitude(alt)

    # Troposphere
    if ISA_MIN_ALTITUDE <= alt <= TROPOPAUSE_BASE_ALTITUDE:
        t0 = MSL_STANDARD_TEMPERTURE
        lr = TROPOSPHERE_LAPSE_RATE

    # Tropopause
    elif TROPOPAUSE_BASE_ALTITUDE < alt <= STRATOSPHERE1_BASE_ALTITUDE:
        t0 = TROPOPAUSE_CONSTANT_TEMPERATURE
        lr = 0.0

    # Stratosphere region 1
    elif STRATOSPHERE1_BASE_ALTITUDE < alt <= STRATOSPHERE2_BASE_ALTITUDE:
        t0 = STRATOSPHERE1_BASE_TEMPERATURE
        lr = STRATOSPHERE1_LAPSE_RATE

    # Stratosphere region 2
    elif STRATOSPHERE2_BASE_ALTITUDE < alt <= STRATOPAUSE_BASE_ALTITUDE:
        t0 = STRATOSPHERE2_BASE_TEMPERATURE
        lr = STRATOSPHERE2_LAPSE_RATE

    # Stratopause
    elif STRATOPAUSE_BASE_ALTITUDE < alt <= MESOSPHERE1_BASE_ALTITUDE:
        t0 = STRATOPAUSE_CONSTANT_TEMPERATURE
        lr = 0.0

    # Mesosphere region 1
    elif MESOSPHERE1_BASE_ALTITUDE < alt <= MESOSPHERE2_BASE_ALTITUDE:
        t0 = MESOSPHERE1_BASE_TEMPERATURE
        lr = MESOSPHERE1_LAPSE_RATE

    # Mesosphere region 2
    else:
        t0 = MESOSPHERE2_BASE_TEMPERATURE
        lr = MESOSPHERE2_LAPSE_RATE

    t_K = t0 + lr * alt
    return (t_K - KELVIN0CELSIUS) if celsius else t_K


def isa_pressure(alt: float) -> float:
    """
    Compute ISA (International Standard Atmosphere 1976) pressure at a given altitude.

    Parameters
    ----------
    alt : float, optional
        Altitude above MSL (mean sea level) in meters

    Returns
    -------
    float
        Pressure in Pa

    Raises
    ------
    ValueError
        If altitude is outside the supported range (-6 to 86 km)
    """
    _check_altitude(alt)
    g0_R = EARTH_GRAVITY_CONSTANT / AIR_GAS_CONSTANT

    # Troposphere
    if ISA_MIN_ALTITUDE <= alt <= TROPOPAUSE_BASE_ALTITUDE:
        z = alt - TROPOSPHERE_BASE_ALTITUDE
        p0 = MSL_STANDARD_PRESSURE
        t0 = MSL_STANDARD_TEMPERTURE
        lr = TROPOSPHERE_LAPSE_RATE
        p = p0 * (1.0 + lr * z / t0) ** (-g0_R / lr)

    # Tropopause
    elif TROPOPAUSE_BASE_ALTITUDE < alt <= STRATOSPHERE1_BASE_ALTITUDE:
        z = alt - TROPOPAUSE_BASE_ALTITUDE
        p0 = isa_pressure(TROPOPAUSE_BASE_ALTITUDE)
        t0 = TROPOPAUSE_CONSTANT_TEMPERATURE
        p = p0 * np.exp(-g0_R / t0 * z)

    # Stratosphere region 1
    elif STRATOSPHERE1_BASE_ALTITUDE < alt <= STRATOSPHERE2_BASE_ALTITUDE:
        z = alt - STRATOSPHERE1_BASE_ALTITUDE
        p0 = isa_pressure(STRATOSPHERE1_BASE_ALTITUDE)
        t0 = STRATOSPHERE1_BASE_TEMPERATURE
        lr = STRATOSPHERE1_LAPSE_RATE
        p = p0 * (1.0 + lr * z / t0) ** (-g0_R / lr)

    # Stratosphere region 2
    elif STRATOSPHERE2_BASE_ALTITUDE < alt <= STRATOPAUSE_BASE_ALTITUDE:
        z = alt - STRATOSPHERE2_BASE_ALTITUDE
        p0 = isa_pressure(STRATOSPHERE2_BASE_ALTITUDE)
        t0 = STRATOSPHERE2_BASE_TEMPERATURE
        lr = STRATOSPHERE2_LAPSE_RATE
        p = p0 * (1.0 + lr * z / t0) ** (-g0_R / lr)

    # Stratopause
    elif STRATOPAUSE_BASE_ALTITUDE < alt <= MESOSPHERE1_BASE_ALTITUDE:
        z = alt - STRATOPAUSE_BASE_ALTITUDE
        p0 = isa_pressure(STRATOPAUSE_BASE_ALTITUDE)
        t0 = STRATOPAUSE_CONSTANT_TEMPERATURE
        p = p0 * np.exp(-g0_R / t0 * z)

    # Mesosphere region 1
    elif MESOSPHERE1_BASE_ALTITUDE < alt <= MESOSPHERE2_BASE_ALTITUDE:
        z = alt - MESOSPHERE1_BASE_ALTITUDE
        p0 = isa_pressure(MESOSPHERE1_BASE_ALTITUDE)
        t0 = MESOSPHERE1_BASE_TEMPERATURE
        lr = MESOSPHERE1_LAPSE_RATE
        p = p0 * (1.0 + lr * z / t0) ** (-g0_R / lr)

    # Mesosphere region 2
    else:
        z = alt - MESOSPHERE2_BASE_ALTITUDE
        p0 = isa_pressure(MESOSPHERE2_BASE_ALTITUDE)
        t0 = MESOSPHERE2_BASE_TEMPERATURE
        lr = MESOSPHERE2_LAPSE_RATE
        p = p0 * (1.0 + lr * z / t0) ** (-g0_R / lr)

    return p


def isa_density(alt: float) -> float:
    """
    Compute ISA (International Standard Atmosphere 1976) density at a given altitude.

    Parameters
    ----------
    alt : float, optional
        Altitude above MSL (mean sea level) in meters

    Returns
    -------
    float
        Density in kg/m^3

    Raises
    ------
    ValueError
        If altitude is outside the supported range (-6 to 86 km)
    """
    _check_altitude(alt)
    return isa_pressure(alt) / (AIR_GAS_CONSTANT * isa_temperature(alt))


def isa_soundspeed(alt: float) -> float:
    """
    Compute ISA (International Standard Atmosphere 1976) speed of sound at a given altitude.

    Parameters
    ----------
    alt : float, optional
        Altitude above MSL (mean sea level) in meters

    Returns
    -------
    float
        Speed of sound in m/s

    Raises
    ------
    ValueError
        If altitude is outside the supported range (-6 to 86 km)
    """
    _check_altitude(alt)
    return np.sqrt(AIR_SPECIFIC_HEAT * AIR_GAS_CONSTANT * isa_temperature(alt))


if __name__ == "__main__":
    print("International Standard Atmosphere (ISA) 1976 Table")
    print("+--------+---------+--------+---------+--------+--------------+---------+")
    print("| h (ft) |  h (m)  | T (°C) |  T (K)  | P (Pa) | rho (kg/m^3) | a (m/s) |")
    print("+--------+---------+--------+---------+--------+--------------+---------+")
    for h_ft in np.arange(-2e3, 60e3 + 1e3, 1e3):
        h_m = h_ft * FT2M
        t_K = isa_temperature(h_m)
        t_C = t_K - KELVIN0CELSIUS
        p = isa_pressure(h_m)
        rho = isa_density(h_m)
        a = isa_soundspeed(h_m)
        print(
            f"|  {h_ft:5.0f} | {h_m:7.1f} | {t_C:6.2f} | {t_K:7.2f} | {np.round(p):6.0f} "
            f"|       {rho:6.4f} |   {a:5.1f} |"
        )
    print("+--------+---------+--------+---------+--------+--------------+---------+")
