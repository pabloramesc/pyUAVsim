import numpy as np
from scipy.optimize import fsolve

def damping_ratio_from_overshoot(overshoot_percent):
    """Calculate damping ratio from percent overshoot."""
    def equation(zeta):
        return np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) - (overshoot_percent / 100)
    
    initial_guess = 0.5
    zeta_solution, = fsolve(equation, initial_guess)
    return zeta_solution

def natural_frequency_from_settling_time(settling_time, zeta):
    """Calculate natural frequency from settling time and damping ratio."""
    return 4 / (zeta * settling_time)

def main():
    # Example values
    overshoot_percent = 2  # percent overshoot
    settling_time_2_percent = 0.5  # settling time for 2% criterion

    # Calculate damping ratio
    zeta = damping_ratio_from_overshoot(overshoot_percent)
    print(f"Damping Ratio (ζ): {zeta:.4f}")

    # Calculate natural frequency
    omega_n = natural_frequency_from_settling_time(settling_time_2_percent, zeta)
    print(f"Natural Frequency (ω_n): {omega_n:.4f} rad/s")

if __name__ == "__main__":
    main()
