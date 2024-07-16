"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

# International Standard Atmosphere 1976 (ISA)
from simulator.common.constants import (
    AIR_SPECIFIC_HEAT,
    DRY_AIR_GAS_CONSTANT,
    FT2M,
    ISA_MAX_ALTITUDE,
    ISA_MIN_ALTITUDE,
    KELVIN0CELSIUS,
    MESOSPHERE_BASE_ALTITUDE1,
    MESOSPHERE_BASE_ALTITUDE2,
    MESOSPHERE_BASE_TEMPERATURE1,
    MESOSPHERE_BASE_TEMPERATURE2,
    MESOSPHERE_LAPSE_RATE1,
    MESOSPHERE_LAPSE_RATE2,
    MSL_STANDARD_PRESSURE,
    MSL_STANDARD_TEMPERTURE,
    EARTH_GRAVITY_CONSTANT,
    STRATOPAUSE_BASE_ALTITUDE,
    STRATOPAUSE_CONSTANT_TEMPERATURE,
    STRATOSPHERE_BASE_ALTITUDE1,
    STRATOSPHERE_BASE_ALTITUDE2,
    STRATOSPHERE_BASE_TEMPERATURE1,
    STRATOSPHERE_BASE_TEMPERATURE2,
    STRATOSPHERE_LAPSE_RATE1,
    STRATOSPHERE_LAPSE_RATE2,
    TROPOPAUSE_BASE_ALTITUDE,
    TROPOPAUSE_CONSTANT_TEMPERATURE,
    TROPOSPHERE_BASE_ALTITUDE,
    TROPOSPHERE_LAPSE_RATE,
)


def isa_pressure(alt: float = 0.0) -> float:
    """
    Compute ISA (International Standard Atmosphere 1976) pressure at a given altitude.

    Parameters
    ----------
    alt : float, optional
        altitude above MSL (mean sea level) in meters, by default 0.0

    Returns
    -------
    float
        pressure in Pa

    Raises
    ------
    ValueError
        if altitude is not supported *

    *supported altitudes are between `ISA_MIN_ALTITUDE` and `ISA_MAX_ALTITUDE`
    """
    # troposphere
    if ISA_MIN_ALTITUDE <= alt <= TROPOPAUSE_BASE_ALTITUDE:
        pressure = MSL_STANDARD_PRESSURE * (
            1.0
            - TROPOSPHERE_LAPSE_RATE
            / MSL_STANDARD_TEMPERTURE
            * (alt - TROPOSPHERE_BASE_ALTITUDE)
        ) ** (EARTH_GRAVITY_CONSTANT / (DRY_AIR_GAS_CONSTANT * TROPOSPHERE_LAPSE_RATE))
    # tropopause
    elif TROPOPAUSE_BASE_ALTITUDE < alt <= STRATOSPHERE_BASE_ALTITUDE1:
        tropopause_base_pressure = isa_pressure(TROPOPAUSE_BASE_ALTITUDE)
        pressure = tropopause_base_pressure * np.exp(
            -EARTH_GRAVITY_CONSTANT
            / (DRY_AIR_GAS_CONSTANT * TROPOPAUSE_CONSTANT_TEMPERATURE)
            * (alt - TROPOPAUSE_BASE_ALTITUDE)
        )
    # stratosphere region 1
    elif STRATOSPHERE_BASE_ALTITUDE1 < alt <= STRATOSPHERE_BASE_ALTITUDE2:
        stratosphere_base_pressure1 = isa_pressure(STRATOSPHERE_BASE_ALTITUDE1)
        pressure = stratosphere_base_pressure1 * (
            1.0
            - STRATOSPHERE_LAPSE_RATE1
            / STRATOSPHERE_BASE_TEMPERATURE1
            * (alt - STRATOSPHERE_BASE_ALTITUDE1)
        ) ** (
            EARTH_GRAVITY_CONSTANT / (DRY_AIR_GAS_CONSTANT * STRATOSPHERE_LAPSE_RATE1)
        )
    # stratosphere region 2
    elif STRATOSPHERE_BASE_ALTITUDE2 < alt <= STRATOPAUSE_BASE_ALTITUDE:
        stratosphere_base_pressure2 = isa_pressure(STRATOSPHERE_BASE_ALTITUDE2)
        pressure = stratosphere_base_pressure2 * (
            1.0
            - STRATOSPHERE_LAPSE_RATE2
            / STRATOSPHERE_BASE_TEMPERATURE2
            * (alt - STRATOSPHERE_BASE_ALTITUDE2)
        ) ** (
            EARTH_GRAVITY_CONSTANT / (DRY_AIR_GAS_CONSTANT * STRATOSPHERE_LAPSE_RATE2)
        )
    # stratopause
    elif STRATOPAUSE_BASE_ALTITUDE < alt <= MESOSPHERE_BASE_ALTITUDE1:
        stratopause_base_pressure = isa_pressure(STRATOPAUSE_BASE_ALTITUDE)
        pressure = stratopause_base_pressure * np.exp(
            -EARTH_GRAVITY_CONSTANT
            / (DRY_AIR_GAS_CONSTANT * STRATOPAUSE_CONSTANT_TEMPERATURE)
            * (alt - STRATOPAUSE_BASE_ALTITUDE)
        )
    # mesosphere region 1
    elif MESOSPHERE_BASE_ALTITUDE1 < alt <= MESOSPHERE_BASE_ALTITUDE2:
        mesosphere_base_pressure1 = isa_pressure(MESOSPHERE_BASE_ALTITUDE1)
        pressure = mesosphere_base_pressure1 * (
            1.0
            - MESOSPHERE_LAPSE_RATE1
            / MESOSPHERE_BASE_TEMPERATURE1
            * (alt - MESOSPHERE_BASE_ALTITUDE1)
        ) ** (EARTH_GRAVITY_CONSTANT / (DRY_AIR_GAS_CONSTANT * MESOSPHERE_LAPSE_RATE2))
    # mesosphere region 2
    elif MESOSPHERE_BASE_ALTITUDE2 < alt <= ISA_MAX_ALTITUDE:
        mesosphere_base_pressure2 = isa_pressure(MESOSPHERE_BASE_ALTITUDE2)
        pressure = mesosphere_base_pressure2 * (
            1.0
            - MESOSPHERE_LAPSE_RATE2
            / MESOSPHERE_BASE_TEMPERATURE2
            * (alt - MESOSPHERE_BASE_ALTITUDE2)
        ) ** (EARTH_GRAVITY_CONSTANT / (DRY_AIR_GAS_CONSTANT * MESOSPHERE_LAPSE_RATE2))
    # unsuported altitude!
    else:
        raise ValueError(
            "ISA model is limited for altitudes between {} and {} meters above MSL (mean sea level)".format(
                ISA_MIN_ALTITUDE, ISA_MAX_ALTITUDE
            )
        )

    return pressure


def isa_temperature(alt: float = 0.0) -> float:
    """
    Compute ISA (International Standard Atmosphere 1976) temperature at a given altitude.

    Parameters
    ----------
    alt : float, optional
        altitude above MSL (mean sea level) in meters, by default 0.0

    Returns
    -------
    float
        temperature in °C

    Raises
    ------
    ValueError
        if altitude is not supported *

    *supported altitudes are between `ISA_MIN_ALTITUDE` and `ISA_MAX_ALTITUDE`
    """
    # troposphere
    if ISA_MIN_ALTITUDE <= alt <= TROPOPAUSE_BASE_ALTITUDE:
        temperature = MSL_STANDARD_TEMPERTURE - TROPOSPHERE_LAPSE_RATE * alt
    # tropopause
    elif TROPOPAUSE_BASE_ALTITUDE < alt <= STRATOSPHERE_BASE_ALTITUDE1:
        temperature = TROPOPAUSE_CONSTANT_TEMPERATURE
    # stratosphere region 1
    elif STRATOSPHERE_BASE_ALTITUDE1 < alt <= STRATOSPHERE_BASE_ALTITUDE2:
        temperature = STRATOSPHERE_BASE_TEMPERATURE1 - STRATOSPHERE_LAPSE_RATE1 * alt
    # stratosphere region 2
    elif STRATOSPHERE_BASE_ALTITUDE2 < alt <= STRATOPAUSE_BASE_ALTITUDE:
        temperature = STRATOSPHERE_BASE_TEMPERATURE2 - STRATOSPHERE_LAPSE_RATE2 * alt
    # stratopause
    elif STRATOPAUSE_BASE_ALTITUDE < alt <= MESOSPHERE_BASE_ALTITUDE1:
        temperature = STRATOPAUSE_CONSTANT_TEMPERATURE
    # mesosphere region 1
    elif MESOSPHERE_BASE_ALTITUDE1 < alt <= MESOSPHERE_BASE_ALTITUDE2:
        temperature = MESOSPHERE_BASE_TEMPERATURE1 - MESOSPHERE_LAPSE_RATE1 * alt
    # mesosphere region 2
    elif MESOSPHERE_BASE_ALTITUDE2 < alt <= ISA_MAX_ALTITUDE:
        temperature = MESOSPHERE_BASE_TEMPERATURE2 - MESOSPHERE_LAPSE_RATE2 * alt
    # unsuported altitude!
    else:
        raise ValueError(
            "ISA model is limited for altitudes between {} and {} meters above MSL (mean sea level)".format(
                ISA_MIN_ALTITUDE, ISA_MAX_ALTITUDE
            )
        )
    return temperature


def isa_density(alt: float = 0.0) -> float:
    return isa_pressure(alt) / (DRY_AIR_GAS_CONSTANT * isa_temperature(alt))


def isa_soundspeed(alt: float = 0.0) -> float:
    return np.sqrt(AIR_SPECIFIC_HEAT * DRY_AIR_GAS_CONSTANT * isa_temperature(alt))


if __name__ == "__main__":

    print("International Standard Atmosphere (ISA) Table")

    print(
        "==========================================================================================================================="
    )
    print(
        "| altitude (ft) | altitude (m) | temperature (°C) | temperature (K) | pressure (Pa) | density (kg/m^3) |"
        " soundspeed (m/s) |"
    )
    print(
        "---------------------------------------------------------------------------------------------------------------------------"
    )
    for h_ft in np.arange(-2000.0, 60000.0 + 1000.0, 1000.0):
        h_m = h_ft * FT2M
        t_K = isa_temperature(h_m)
        t_C = t_K - KELVIN0CELSIUS
        p = isa_pressure(h_m)
        rho = isa_density(h_m)
        a = isa_soundspeed(h_m)
        print(
            "|       {:5.0f}   |    {:7.1f}   |        {:7.2f}   |       {:7.2f}   |   {:9.2f}   |         {:6.4f}   | "
            "         {:5.1f}   |".format(h_ft, h_m, t_C, t_K, p, rho, a)
        )
    print(
        "==========================================================================================================================="
    )
