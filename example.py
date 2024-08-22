"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.cli import SimConsole
from simulator.gui import AttitudePosition3DView
from simulator.utils import wait_animation

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
uav.trim(25.0, np.deg2rad(10.0), 500, update=True)
wait_animation(10.0)  # wait 10 seconds to visualize trim vars

cli = SimConsole()
gui = AttitudePosition3DView(use_blit=True)

t = 0.0  # simulation time
k = 0  # simulation steps
while True:
    t += dt
    k += 1

    uav.update()  # update simulation states

    if k % 10 == 0:  # update interface each 10 steps
        cli.print_state(t, uav.state)
        gui.update(uav.state, pause=0.01)
