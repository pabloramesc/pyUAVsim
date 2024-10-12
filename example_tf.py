"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt

from simulator.math.transfer_function import TransferFunction
from simulator.math.root_locus import root_locus

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
plt.figure(1)
plt.plot(t, y)
plt.title("Step Response")
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.grid()

# Calculate the impulse response
t, y = tf.impulse()

# Plot the step response
plt.figure(2)
plt.plot(t, y)
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Response")
plt.grid()

# Calculate the bode plot
w, mag, phase = tf.bode()

# Plot Bode plot
plt.figure(3)

# Magnitude plot
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title('Bode Plot')
plt.ylabel('Magnitude (dB)')
plt.grid()

# Phase plot
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (rad/s)')
plt.grid()

plt.tight_layout()

plt.show()