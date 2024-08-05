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
from simulator.gui import AltitudeTimeLog, AttitudeView, Position2DPlot

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params)
uav.trim(10.0, np.deg2rad(10.0), np.inf, update=True)
time.sleep(10.0)

cli = SimConsole()

plt.ion()
fig = plt.figure(figsize=(12, 6))
att_view = AttitudeView(fig, 121)
pos_plot = Position2DPlot(fig, 222)
alt_log = AltitudeTimeLog(fig, 224)

t = 0.0
while True:
    t += dt

    uav.update()

    cli.print_state(t, uav.state, style="table")

    att_view.update(uav.state)
    pos_plot.update(uav.state)
    alt_log.update(uav.state, time=t)

    plt.draw()
    plt.pause(0.01)
