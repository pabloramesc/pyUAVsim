import numpy as np
import time

from simulator.aircraft import Aircraft, load_airframe_parameters_from_yaml
from simulator.visualization.aircraft_visualization import AircraftVisualization

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
aircraft = Aircraft(dt, aerosonde_params)

visualization = AircraftVisualization()

t0 = time.time()
while True:
    t = time.time() - t0

    fx = 100.0
    fy = 0.0
    fz = 0.0

    l = 0.0
    m = 1.0
    n = 0.0

    forces = np.array([fx, fy, fz])
    moments = np.array([l, m, n])

    aircraft.update_state(forces, moments)

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

    visualization.update(aircraft.state.state, pause=0.01)
