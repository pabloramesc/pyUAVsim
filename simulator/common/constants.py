"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np


### TRIGONOMETRY
PI = np.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
EPS = 1e-12  # a constant that is almost zero


### ESPHERICAL GEOGRAPHIC COORDINATES
GEO_DEG2NM = 60.0  # geographic distance in degrees to nautic miles
GEO_NM2DEG = 1.0 / GEO_DEG2NM
NM2KM = 1.852  # nautic mile to km
NM2M = 1852  # nautic mile to m
GEO_DEG2M = GEO_DEG2NM * NM2M  # geographic distance in degrees to meters
GEO_M2DEG = 1.0 / GEO_DEG2M


### WORLD GEODETIC SYSTEM 1984 (WGS84)
WGS84_EXCENTRICITY2 = 0.00669438
WGS84_EQUATORIAL_RADIUS = 6378137.0
WGS84_POLAR_RADIUS = 6356752.3

DEFAULT_HOME_COORDS = (
    40.4168,
    -3.7038,
    0.0,
)  # lat and long of Madrid in deg and altitude of 0 meters


### EARTH GRAVITY FIELD
EARTH_GRAVITY_CONSTANT = 9.80665  # Earth gravitational constant. Units: m/s^2
EARTH_GRAVITY_VECTOR = (
    np.array((0, 0, 1)) * EARTH_GRAVITY_CONSTANT
)  # Earth gravitational vector. Units: m/s^2

### WORLD MAGNETIC MODEL 2020 (WMM)
DEFAULT_HOME_WMM_NED = (
    25671.76,
    -142.34,
    36839.82,
)  # Madrid magnetic field vector North-East-Down components. Units: nT
DEFAULT_HOME_WMM_TID = (
    44993.7,
    55.1151,
    -0.3191,
)  # Madrid magnetic field vector expressed as Total field, inclination and declination. Units: nT, deg, deg


### INTERNATIONAL STANDARD ATMOSPHERE 1976 (ISA)
ISA_MIN_ALTITUDE = -2000.0
ISA_MAX_ALTITUDE = 86000.0

MSL_STANDARD_PRESSURE = (
    101325.0  # Standard pressure at sea level. Units: Pa (1 Pa = 1 N/m^2)
)
MSL_STANDARD_TEMPERTURE = 288.15  # Standard temperature at sea level. Units: K
MSL_STANDARD_DENSITY = 1.225  # Standard density at sea level. Units: kg/m^3
MSL_STANDARD_SOUNDSPEED = 340.294  # Standard sound speed at sea level. Units: m/s

TROPOSPHERE_BASE_ALTITUDE = 0.0  # Units: m
TROPOPAUSE_BASE_ALTITUDE = 11000.0  # Units: m
STRATOSPHERE_BASE_ALTITUDE1 = 20000.0  # Units: m
STRATOSPHERE_BASE_ALTITUDE2 = 32000.0  # Units: m
STRATOPAUSE_BASE_ALTITUDE = 47000.0  # Units: m
MESOSPHERE_BASE_ALTITUDE1 = 51000.0  # Units: m
MESOSPHERE_BASE_ALTITUDE2 = 71000.0  # Units: m
MESOPAUSE_BASE_ALTITUDE = 86000.0  # Units: m

TROPOPAUSE_CONSTANT_TEMPERATURE = 216.65  # Units: K
STRATOSPHERE_BASE_TEMPERATURE1 = 216.65  # Units: K
STRATOSPHERE_BASE_TEMPERATURE2 = 228.65  # Units: K
STRATOPAUSE_CONSTANT_TEMPERATURE = 270.65  # Units: K
MESOSPHERE_BASE_TEMPERATURE1 = 270.65  # Units: K
MESOSPHERE_BASE_TEMPERATURE2 = 214.65  # Units: K
MESOPAUSE_BASE_TEMPERATURE = 186.95  # Units: K

# TROPOSPHERE_BASE_PRESSURE = 0.0  # Units: Pa
# TROPOPAUSE_BASE_PRESSURE = 0.00  # Units: Pa
# STRATOSPHERE_BASE_PRESSURE1 = 0.00  # Units: Pa
# STRATOSPHERE_BASE_PRESSURE2 = 0.00  # Units: Pa
# STRATOPAUSE_BASE_PRESSURE = 0.00  # Units: Pa
# MESOSPHERE_BASE_PRESSURE1 = 0.00  # Units: Pa
# MESOSPHERE_BASE_PRESSURE2 = 0.00  # Units: Pa
# MESOPAUSE_BASE_PRESSURE = 0.00  # Units: Pa

TROPOSPHERE_LAPSE_RATE = 0.0065  # Temperature lapse rate by altitude between 0 and 11 km of altitude. Units: K/m
STRATOSPHERE_LAPSE_RATE1 = (
    -0.0010
)  # Temperature lapse rate by altitude between 20 and 47 km of altitude. Units: K/m
STRATOSPHERE_LAPSE_RATE2 = (
    -0.0028
)  # Temperature lapse rate by altitude between 20 and 47 km of altitude. Units: K/m
MESOSPHERE_LAPSE_RATE1 = 0.0028  # Temperature lapse rate by altitude between 51 and 71 km of altitude. Units: K/m
MESOSPHERE_LAPSE_RATE2 = 0.0020  # Temperature lapse rate by altitude between 71 and 84.852 km of altitude. Units: K/m

EARTH_ISA_RADIUS = 6356766.0  # Units: m

AIR_SPECIFIC_HEAT = 1.400  # Units: (adimensional)

UNIVERSAL_GAS_CONSTANT = 8.31432  # Universal gas constant. Units: J/(mol*K)
DRY_AIR_MOLAR_MASS = 0.0289644  # Dry air molar mass. Units: kg/mol
DRY_AIR_GAS_CONSTANT = (
    UNIVERSAL_GAS_CONSTANT / DRY_AIR_MOLAR_MASS
)  # Dry air gas constant. Units: J/(mol*K)

DRY_AIR_SPECIFIC_WEIGHT = (
    MSL_STANDARD_DENSITY * EARTH_GRAVITY_CONSTANT
)  # Air specific weight. Units: N/m^3


### PRESSURE UNITS CONVERSION FACTORS
ATM2PA = (
    101325.0  # Standard atmosphere to Pascal conversion. Units: Pa/atm (1 Pa = 1 N/m^2)
)
PA2ATM = (
    1.0 / ATM2PA
)  # Pascal to Standard atmosphere conversion. Units: atm/Pa (1 Pa = 1 N/m^2)
INHG2PA = 3386.39  # Inch of mercury to Pascal. Units: inchHg/Pa
PA2INHG = 1.0 / INHG2PA  # Pascal to Inch of mercury. Units: Pa/inchHg
MMHG2PA = 133.322  # Milimetre of mercury to Pascal. Units: mmHg/Pa
PA2MMHG = 1.0 / 133.0  # Pascal to milimetre of mercury. Units: Pa/mmHg


### TEMPERATURE UNITS CONVERSION FACTORS
KELVIN0CELSIUS = 273.15


### DISTANCE UNITS CONVERSION FACTORS
M2FT = 3.28084  # meters to feet
FT2M = 1.0 / M2FT  # feet to meters
NM2M = 1852.0  # nautic mile to meters
M2NM = 1.0 / NM2M  # meters to nautic mile


### MAGNETIC CONVERSION FACTORS
GAUSS2UT = 100  # gauss to micro-tesla
UT2GAUSS = 1.0 / GAUSS2UT  # micro-tesla to gauss
