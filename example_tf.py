"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
import numpy as np

from simulator.math.transfer_function import TransferFunction

# Example 1: 1st Order Transfer Function
tau = 1.0  # Time constant
num = [1.0]  # Numerator coefficients
den = [tau, 1.0]  # Denominator coefficients

# Example 2: 2nd Order Transfer Function
wn = 10.0  # Natural frequency
zeta = 0.7  # Damping ratio
num = [wn**2]  # Numerator coefficients
den = [1.0, 2 * zeta * wn, wn**2]  # Denominator coefficients

# Create a transfer function system
tf = TransferFunction(num, den)

# Calculate the step response
t, y = tf.step()

# Plot the step response
plt.figure()
plt.plot(t, y)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.grid()
plt.show()
