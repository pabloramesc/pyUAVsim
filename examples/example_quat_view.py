"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft import AircraftState
from simulator.cli import SimConsole
from simulator.gui import AttitudePosition3DView
from simulator.math.rotation import euler2quat

cli = SimConsole()
gui = AttitudePosition3DView()

t0 = time.time()
while True:
    t = time.time() - t0

    roll = np.pi * np.sin(0.1 * np.pi * t)
    pitch = 0.0 #np.pi / 2 * np.sin(0.05 * np.pi * t)
    yaw = 0.0 #np.pi * np.sin(0.02 * np.pi * t)
    north = 20 * np.sin(0.1 * np.pi * t)
    east = 10 * np.sin(0.05 * np.pi * t)
    down = 30 * np.sin(0.01 * np.pi * t)

    # x = np.array([north, east, down, 0, 0, 0, roll, pitch, yaw, 0, 0, 0])
    # state = AircraftState(x)

    q = euler2quat(np.array([roll, pitch, yaw]))
    x = np.array([north, east, down, 0, 0, 0, q[0], q[1], q[2], q[3], 0, 0, 0])
    state = AircraftState(x, use_quat=True)

    cli.print_aircraft_state(t, state)
    gui.update(state, pause=0.01)
