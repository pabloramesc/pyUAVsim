"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.cli import SimConsole
from simulator.gui import AttitudePositionPanel
from simulator.utils import wait_animation

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
x0 = np.array(
    [
        0.0,  # pd
        0.0,  # pe
        0.0,  # pn
        5.0,  # u
        0.0,  # v
        0.0,  # w
        1.0,  # q0
        0.0,  # q1
        0.0,  # q2
        0.0,  # q3
        0.0,  # p
        0.0,  # q
        0.0,  # r
    ]
)
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True, x0=x0)

cli = SimConsole()
gui = AttitudePositionPanel(use_blit=True)

sim_time = 0.0  # simulation time
sim_iter = 0  # simulation steps
real_t0 = time.time()
while True:
    sim_time += dt
    sim_iter += 1

    uav.update()  # update simulation states
    
    gui.add_data(uav.state, sim_time)

    if sim_iter % 10 == 0:  # update interface each 10 steps
        real_time = time.time() - real_t0
        cli.print_state(sim_time, real_time, uav.state)
        gui.update(uav.state, pause=0.01)
