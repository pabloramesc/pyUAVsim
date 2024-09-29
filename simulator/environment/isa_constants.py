"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.environment.constants import EARTH_GRAVITY_CONSTANT

"""
General constants for the International Standard Atmosphere (ISA) 1976 model.
"""

ISA_MIN_ALTITUDE = -2e3
ISA_MAX_ALTITUDE = 86e3

# Mean Sea Level (MSL) constants
MSL_STANDARD_PRESSURE = 101325.0  # Units: Pa
MSL_STANDARD_TEMPERTURE = 288.15  # Units: K
MSL_STANDARD_DENSITY = 1.225  # Units: kg/m^3
MSL_STANDARD_SOUNDSPEED = 340.294  # Units: m/s

# Gas properties
UNIVERSAL_GAS_CONSTANT = 8.31432  # Units: J/(mol*K)
AIR_MOLAR_MASS = 0.0289644  # Units: kg/mol
AIR_GAS_CONSTANT = UNIVERSAL_GAS_CONSTANT / AIR_MOLAR_MASS  # Units: J/(mol*K)
AIR_SPECIFIC_WEIGHT = MSL_STANDARD_DENSITY * EARTH_GRAVITY_CONSTANT  # Units: N/m^3
AIR_SPECIFIC_HEAT = 1.400  # (adimensional)

# Troposphere (-6 to 11 km)
TROPOSPHERE_BASE_ALTITUDE = 0.0  # Units: m
TROPOPAUSE_BASE_ALTITUDE = 11e3  # Units: m
TROPOSPHERE_LAPSE_RATE = -6.5e-3  # Temperature rate by altitude. Units: K/m

# Tropopause (11 to 20 km)
TROPOPAUSE_CONSTANT_TEMPERATURE = 216.65  # Units: K

# Stratosphere 1st layer (20 to 32 km)
STRATOSPHERE1_BASE_ALTITUDE = 20e3  # Units: m
STRATOSPHERE1_BASE_TEMPERATURE = 216.65  # Units: K
STRATOSPHERE1_LAPSE_RATE = +1.0e-3  # Temperature rate by altitude. Units: K/m

# Stratosphere 2nd layer (32 to 47 km)
STRATOSPHERE2_BASE_ALTITUDE = 32e3  # Units: m
STRATOSPHERE2_BASE_TEMPERATURE = 228.65  # Units: K
STRATOSPHERE2_LAPSE_RATE = +2.8e-3  # Temperature rate by altitude. Units: K/m

# Stratopause (47 to 51 km)
STRATOPAUSE_BASE_ALTITUDE = 47e3  # Units: m
STRATOPAUSE_CONSTANT_TEMPERATURE = 270.65  # Units: K

# Mesosphere 1st layer (51 to 71 km)
MESOSPHERE1_BASE_ALTITUDE = 51e3  # Units: m
MESOSPHERE1_BASE_TEMPERATURE = 270.65  # Units: K
MESOSPHERE1_LAPSE_RATE = -2.8e-3  # Temperature rate by altitude. Units: K/m

# Mesosphere 2nd layer (71 to 86 km)
MESOSPHERE2_BASE_ALTITUDE = 71e3  # Units: m
MESOSPHERE2_BASE_TEMPERATURE = 214.65  # Units: K
MESOSPHERE2_LAPSE_RATE = -2.0e-3  # Temperature rate by altitude. Units: K/m

# Mesopause (86 km)
MESOPAUSE_BASE_ALTITUDE = 86e3  # Units: m
MESOPAUSE_BASE_TEMPERATURE = 186.95  # Units: K
