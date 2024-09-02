"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.autopilot import Autopilot
from simulator.cli import SimConsole
from simulator.gui import AttitudePositionPanel

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
uav.trim(25.0, np.deg2rad(10.0), 500, update=True, verbose=False)

ap = Autopilot(dt, aerosonde_params, uav.state)

cli = SimConsole()
gui = AttitudePositionPanel(use_blit=True, pos_3d=True)

t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()
while True:
    t_sim += dt
    k_sim += 1

    ap.control_course_altitude(course_target=np.deg2rad(180.0), altitude_target=1e3)
    uav.update(ap.control_deltas)  # update simulation states
    
    gui.add_data(state=uav.state)

    if k_sim % 10 == 0:  # update interface each 10 steps
        t_real = time.time() - t0
        cli.print_time(t_sim, t_real, dt, k_sim)
        cli.print_aircraft_state(uav.state)
        cli.print_control_deltas(uav.control_deltas)
        cli.print_autopilot_status(ap.status)
        gui.update(state=uav.state, pause=0.01)
