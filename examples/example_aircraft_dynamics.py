import numpy as np
import time

from simulator.aircraft import Aircraft, AirframeParameters, load_airframe_parameters_from_yaml
from simulator.visualization.aircraft_visualization import AircraftVisualization

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
aircraft = Aircraft(dt, aerosonde_params)
state = aircraft.state

visualization = AircraftVisualization()

t0 = time.time()
while True:
    t = time.time() - t0

    fx = 100.0
    fy = 0.0
    fz = 0.0

    l = 0.2 * np.sin(2*np.pi*0.02*t)
    m = 1.0 * np.sin(2*np.pi*0.01*t)
    n = 0.0

    forces = np.array([fx, fy, fx])
    moments = np.array([l, m, n])

    state = aircraft.dynamics(forces, moments) # get updated state
    aircraft.state = state # update aircraft state

    print(f"Time: {t:.2f} s")
    print(f"Position (NED):      pn: {state[0]:.2f}, pe: {state[1]:.2f}, pd: {state[2]:.2f}")
    print(f"Velocity (Body):     u: {state[3]:.2f}, v: {state[4]:.2f}, w: {state[5]:.2f}")
    print(f"Attitude (Radians):  roll: {state[6]:.2f}, pitch: {state[7]:.2f}, yaw: {state[8]:.2f}")
    print(f"Angular Rates:       p: {state[9]:.2f}, q: {state[10]:.2f}, r: {state[11]:.2f}")
    print("-" * 50)

    visualization.update(state, pause=0.01)
