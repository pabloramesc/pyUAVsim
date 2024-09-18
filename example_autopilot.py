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
from simulator.autopilot import Autopilot, LineFollower, OrbitFollower
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
line_follower = LineFollower(autopilot.config)
line_follower.set_path(np.array([0, 500, -1e3]), np.array([1, 0, -0.2]))
orbit_follower = OrbitFollower(autopilot.config)
orbit_follower.set_path(np.array([0, 0, -2e3]), 200.0, 1)

cli = SimConsole()
gui1 = AttitudePositionPanel(use_blit=False, pos_3d=True)
gui2 = FlightControlPanel(use_blit=True)

t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()
while True:
    t_sim += dt
    k_sim += 1
    
    uav.update(autopilot.control_deltas)  # update simulation states

    # roll_cmd = np.deg2rad(0.0) * signal.square(2*np.pi*0.1*t_sim)
    # roll_cmd = np.deg2rad(45.0) * np.sin(2*np.pi*(t_sim/60.0)*t_sim)
    # pitch_cmd = np.deg2rad(0.0) * signal.square(2*np.pi*0.1*t_sim)

    Va_cmd = 25.0 + 0.0 * signal.sawtooth(2*np.pi*0.05*t_sim, width=0.5)
    # h_cmd = 0.0 * signal.sawtooth(2*np.pi*0.01*t_sim, width=0.5)
    # X_cmd = np.deg2rad(90.0) * signal.square(2*np.pi*0.01*t_sim)

    # X_cmd, h_cmd = line_follower.guidance(uav.state.ned_position, uav.state.course_angle)
    X_cmd, h_cmd = orbit_follower.guidance(uav.state.ned_position, uav.state.course_angle)

    autopilot.update(dt, uav.state)
    # autopilot.control_roll_pitch_airspeed(roll_cmd, pitch_cmd, Va_cmd)
    autopilot.control_course_altitude_airspeed(X_cmd, h_cmd, Va_cmd)

    gui1.add_data(state=uav.state)
    gui2.add_data(time=t_sim, ap_status=autopilot.status)

    if k_sim % 10 == 0:  # update interface each 10 steps
        t_real = time.time() - t0
        cli.print_time(t_sim, t_real, dt, k_sim, style='simple')
        cli.print_aircraft_state(uav.state, style='simple')
        cli.print_control_deltas(uav.control_deltas, style='simple')
        cli.print_autopilot_status(autopilot.status, style='simple')
        gui1.update(state=uav.state, pause=0.01)
        gui2.update(pause=0.01)


