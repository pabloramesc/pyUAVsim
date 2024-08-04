"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft import Aircraft, load_airframe_parameters_from_yaml, Trim
from simulator.visualization.attitude_position_view import AttitudePositionView

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

trim = Trim(aerosonde_params)
state0, delta0 = trim.calculate_trim(15.0, np.deg2rad(0.0), 500)

dt = 0.01
aircraft = Aircraft(dt, aerosonde_params, state0=state0, delta0=delta0)

visualization = AttitudePositionView()

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

    visualization.update(aircraft.state.x, pause=0.01)
