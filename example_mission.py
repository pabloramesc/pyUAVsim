"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.autopilot.waypoints import load_waypoints_from_txt
from simulator.autopilot.mission_control import MissionControl

import time

import numpy as np
from scipy import signal

from simulator.aircraft import (
    AircraftDynamics,
    ControlDeltas,
    load_airframe_parameters_from_yaml,
)
from simulator.autopilot import Autopilot
from simulator.cli import SimConsole
from simulator.gui import AttitudePositionPanel

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
x_trim, delta_trim = uav.trim(Va=25.0)

autopilot = Autopilot(dt, aerosonde_params, uav.state)

waypoints_file = r"config/waypoints_example.wp"
waypoints_list = load_waypoints_from_txt(waypoints_file)
mission = MissionControl(dt, autopilot.config)
mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)

cli = SimConsole()
gui = AttitudePositionPanel(use_blit=False, pos_3d=True)

t_sim = 0.0  # simulation time
k_sim = 0  # simulation steps
t0 = time.time()
while True:
    t_sim += dt
    k_sim += 1

    uav.update(autopilot.control_deltas)  # update simulation states

    course_ref, altitude_ref = mission.update(
        uav.state.ned_position, uav.state.course_angle
    )
    autopilot.status.update_aircraft_state(uav.state)
    autopilot.control_course_altitude_airspeed(course_ref, altitude_ref, airspeed=25.0)

    gui.add_data(state=uav.state)

    if k_sim % 1000 == 0:  # update interface each 10 steps
        t_real = time.time() - t0
        cli.print_time(t_sim, t_real, dt, k_sim, style="simple")
        cli.print_aircraft_state(uav.state, style="simple")
        cli.print_control_deltas(uav.control_deltas, style="simple")
        cli.print_autopilot_status(autopilot.status, style="simple")
        cli.print_mission_status(mission)
        gui.update(state=uav.state, pause=0.01)
