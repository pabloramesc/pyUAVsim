"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time

import numpy as np
from scipy import signal

from simulator.aircraft import (
    AircraftDynamics,
    ControlDeltas,
    load_airframe_parameters_from_yaml,
)
from simulator.autopilot import Autopilot, LineFollower
from simulator.cli import SimConsole
from simulator.gui import AttitudePositionPanel, FlightControlPanel
from simulator.utils import wait_animation

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
x0 = np.array(
    [
        0.0,  # pd
        0.0,  # pe
        0.0,  # pn
        25.0,  # u
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
x_trim, delta_trim = uav.trim(Va=25.0)

autopilot = Autopilot(dt, aerosonde_params, uav.state)
follower = LineFollower(autopilot.config)

cli = SimConsole()
gui = AttitudePositionPanel(use_blit=True, pos_3d=True)
# gui = FlightControlPanel(use_blit=True)

t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()
while True:
    t_sim += dt
    k_sim += 1

    roll_cmd = np.deg2rad(0.0) * signal.square(2*np.pi*0.1*t_sim)
    pitch_cmd = np.deg2rad(30.0) * signal.square(2*np.pi*0.1*t_sim)

    Va_cmd = 25.0 + 0.0 * signal.sawtooth(2*np.pi*0.05*t_sim, width=0.5)
    h_cmd = 0.0 * signal.sawtooth(2*np.pi*0.01*t_sim, width=0.5)
    X_cmd = np.deg2rad(60.0) * signal.square(2*np.pi*0.005*t_sim)

    # autopilot.control_pitch_roll(roll_target=roll_cmd, pitch_target=pitch_cmd, airspeed_target=Va_cmd)
    autopilot.control_course_altitude(altitude_target=h_cmd, course_target=X_cmd, airspeed_target=25.0)

    uav.update(autopilot.control_deltas)  # update simulation states

    gui.add_data(time=t_sim, state=uav.state, ap_status=autopilot.status)

    if k_sim % 100 == 0:  # update interface each 10 steps
        t_real = time.time() - t0
        cli.print_time(t_sim, t_real, dt, k_sim, style='simple')
        cli.print_aircraft_state(uav.state, style='simple')
        cli.print_control_deltas(uav.control_deltas, style='simple')
        cli.print_autopilot_status(autopilot.status, style='simple')
        gui.update(state=uav.state, pause=0.01)
