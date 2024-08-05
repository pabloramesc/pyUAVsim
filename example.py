"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.cli import SimConsole
from simulator.gui import AttitudePosition3DView

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params)
uav.trim(25.0, 0.0, 100.0, update=True)
time.sleep(10.0)

cli = SimConsole()
gui = AttitudePosition3DView()

t = 0.0
while True:
    t += dt

    uav.update()

    cli.print_state(t, uav.state)
    gui.update(uav.state.x, pause=0.01)
