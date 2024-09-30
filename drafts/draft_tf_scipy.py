import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step


def calculate_step_response(num, den):
    # Create a transfer function system
    system = TransferFunction(num, den)

    # Calculate the step response
    t, y = step(system, N=10**3)

    # Plot the step response
    plt.figure()
    plt.plot(t, y)
    plt.title("Step Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Response")
    plt.grid()
    plt.show()


# # Example 1: 1st Order Transfer Function
# tau = 1.0  # Time constant
# num = [1.0]  # Numerator coefficients
# den = [tau, 1.0]  # Denominator coefficients

# Example 2: 2nd Order Transfer Function
wn = 10.0  # Natural frequency
zeta = 0.7  # Damping ratio
num = [100.0, wn**2]  # Numerator coefficients
den = [1.0, 2 * zeta * wn, wn**2]  # Denominator coefficients

# Call the function to calculate and plot the step response
calculate_step_response(num, den)
