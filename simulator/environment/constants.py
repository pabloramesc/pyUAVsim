"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

### TRIGONOMETRY
PI = np.pi
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
EPSILON = 1e-12  # a constant that is almost zero

### EARTH'S GRAVITY FIELD
EARTH_GRAVITY_CONSTANT = 9.80665  # Earth's gravitational constant. Units: m/s^2
EARTH_GRAVITY_VECTOR = np.array(
    (0, 0, EARTH_GRAVITY_CONSTANT)
)  # Earth's gravitational vector in NED frame. Units: m/s^2

### PRESSURE UNITS CONVERSION FACTORS
ATM2PA = 101325.0  # Standard atmosphere to Pascal conversion. Units: Pa/atm
PA2ATM = 1.0 / ATM2PA  # Pascal to Standard atmosphere conversion. Units: atm/Pa
INHG2PA = 3386.39  # Inch of mercury to Pascal. Units: inchHg/Pa
PA2INHG = 1.0 / INHG2PA  # Pascal to Inch of mercury. Units: Pa/inchHg
MMHG2PA = 133.322  # Milimetre of mercury to Pascal. Units: mmHg/Pa
PA2MMHG = 1.0 / 133.0  # Pascal to milimetre of mercury. Units: Pa/mmHg

### TEMPERATURE UNITS CONVERSION FACTORS
KELVIN0CELSIUS = 273.15

### DISTANCE UNITS CONVERSION FACTORS
M2FT = 3.28084  # meters to feets
FT2M = 1.0 / M2FT  # feets to meters
NM2M = 1852.0  # nautic miles to meters
M2NM = 1.0 / NM2M  # meters to nautic miles

### MAGNETIC CONVERSION FACTORS
GAUSS2UT = 100  # gauss to micro-tesla
UT2GAUSS = 1.0 / GAUSS2UT  # micro-tesla to gauss
