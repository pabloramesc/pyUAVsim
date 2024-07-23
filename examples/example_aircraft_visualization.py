import time

import numpy as np
from matplotlib import pyplot as plt

from simulator.visualization.aircraft_visualization import \
    AircraftVisualization

# Example usage:
visualization = AircraftVisualization()


# Simulate updating the state over time
t0 = time.time()
while True:
    t = time.time() - t0
    roll = np.pi / 4 * np.sin(0.1 * np.pi * t)
    pitch = np.pi / 6 * np.sin(0.05 * np.pi * t)
    yaw = np.pi / 8 * np.sin(0.02 * np.pi * t)
    north = 20 * np.sin(0.1 * np.pi * t)
    east = 10 * np.sin(0.05 * np.pi * t)
    down = 30 * np.sin(0.01 * np.pi * t)

    print(
        f"Time: {t:.2f} s, "
        f"North: {north:.2f} m, East: {east:.2f} m, Down: {down:.2f} m, "
        f"Roll: {np.rad2deg(roll):.2f} deg, Pitch: {np.rad2deg(pitch):.2f} deg, Yaw: {np.rad2deg(yaw):.2f} deg"
    )

    state = np.array([north, east, down, 0, 0, 0, roll, pitch, yaw, 0, 0, 0])
    visualization.update(state)
    plt.pause(0.01)
