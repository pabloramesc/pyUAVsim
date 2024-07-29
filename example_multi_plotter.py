"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft import Aircraft, load_airframe_parameters_from_yaml
from simulator.visualization import MultiPlotter, AttitudeView, Position2DPlot, AltitudeTimeLog

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.1
state0 = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
deltas0 = np.array([0.0, 0.1, 0.1, 0.5])
aircraft = Aircraft(dt, aerosonde_params, state0=state0, deltas0=deltas0)

plt.ion()
fig = plt.figure(figsize=(12, 6))
att_view = AttitudeView(fig, 121)
pos_plot = Position2DPlot(fig, 222)
alt_log = AltitudeTimeLog(fig, 224)

t = 0.0
while True:
    t += dt

    aircraft.update_state()

    print(f"Time: {t:.2f} s")
    print(
        f"Position (NED):      pn: {aircraft.state.pn:.2f}, pe: {aircraft.state.pe:.2f}, pd: {aircraft.state.pd:.2f}"
    )
    print(
        f"Velocity (Body):     u: {aircraft.state.u:.2f}, v: {aircraft.state.v:.2f}, w: {aircraft.state.w:.2f}"
    )
    print(
        f"Attitude (Radians):  roll: {aircraft.state.roll:.2f}, pitch: {aircraft.state.pitch:.2f}, yaw: {aircraft.state.yaw:.2f}"
    )
    print(
        f"Angular Rates:       p: {aircraft.state.p:.2f}, q: {aircraft.state.q:.2f}, r: {aircraft.state.r:.2f}"
    )
    print("-" * 50)

    att_view.update(aircraft.state)
    pos_plot.update(aircraft.state)
    alt_log.update(aircraft.state, time=t)

    plt.draw()
    plt.pause(0.01)

