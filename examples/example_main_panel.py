"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.cli import SimConsole
from simulator.gui import MainStatusPanel

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params)
uav.trim(25.0, np.deg2rad(10.0), 500.0, update=True)

cli = SimConsole()
gui = MainStatusPanel(use_blit=True)

sim_step = 0
t_sim = 0.0
t0 = time.time()
while True:
    sim_step += 1
    t_sim += dt
    
    uav.update()

    gui.add_data(uav.state, t_sim)

    if sim_step % 10 == 0:
        t_real = time.time() - t0
        cli.print_aircraft_state(t_sim, t_real, uav.state)
        gui.update(uav.state, pause=0.01)