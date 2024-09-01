"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.cli import SimConsole
from simulator.gui import AttitudePosition3DView

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params)

gui = AttitudePosition3DView()

cli = SimConsole()

t = 0.0
while True:
    t += dt
    fx = 100.0
    fy = 0.0
    fz = 0.0
    l = 0.0
    m = 1.0
    n = 0.0

    u = np.array([fx, fy, fz, l, m, n])
    state = uav.kinematics_dynamics(uav.state.x, u)
    uav.state.update(state)

    cli.print_aircraft_state(t, uav.state, style="table")

    gui.update(uav.state.x, pause=0.01)
