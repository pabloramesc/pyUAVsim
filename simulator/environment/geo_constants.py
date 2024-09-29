"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.environment.constants import NM2M

"""
General constants for the International Standard Atmosphere (ISA) 1976 model.
"""

### ESPHERICAL GEOGRAPHIC COORDINATES
GEO_DEG2NM = 60.0  # geographic distance in degrees to nautic miles
GEO_NM2DEG = 1.0 / GEO_DEG2NM
GEO_DEG2M = GEO_DEG2NM * NM2M  # geographic distance in degrees to meters
GEO_M2DEG = 1.0 / GEO_DEG2M

### WORLD GEODETIC SYSTEM 1984 (WGS84)
WGS84_ECCENTRICITY2 = 6.69437999014e-3
WGS84_EQUATORIAL_RADIUS = 6378137.0
WGS84_POLAR_RADIUS = 6356752.3
DEFAULT_HOME_COORDS = (
    40.4168,
    -3.7038,
    0.0,
)  # lat and long of Madrid in deg, and altitude of 0 meters

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
