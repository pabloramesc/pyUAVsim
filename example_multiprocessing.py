"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time
from multiprocessing import Process, Queue

from simulator.simulation_data import SimulationData
from simulator.aircraft import (
    AircraftDynamics,
    load_airframe_parameters_from_yaml,
)
from simulator.autopilot import Autopilot
from simulator.autopilot.mission_control import MissionControl
from simulator.autopilot.waypoints import load_waypoints_from_txt
from simulator.cli import SimConsole


def simulation_process(q: Queue):

    dt = 0.01

    params_file = r"config/aerosonde_parameters.yaml"
    aerosonde_params = load_airframe_parameters_from_yaml(params_file)
    uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
    x_trim, delta_trim = uav.trim(Va=25.0)

    autopilot = Autopilot(dt, aerosonde_params, uav.state)

    waypoints_file = r"config/go_waypoint.wp"
    waypoints_list = load_waypoints_from_txt(waypoints_file)
    mission = MissionControl(dt, autopilot.config)
    mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)

    t = 0.0  # simulation time
    k = 0  # simulation steps
    while True:
        t += dt
        k += 1

        uav.update(autopilot.control_deltas)  # update simulation states

        flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
        autopilot.status.update_aircraft_state(uav.state)
        autopilot.control_course_altitude_airspeed(
            flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
        )

        if k % 100 == 0:  # update interface each 100 steps
            q.put(
                SimulationData(
                    dt,
                    t,
                    k,
                    uav.state,
                    uav.control_deltas,
                    autopilot.status,
                    mission,
                    mission.route_manager,
                )
            )


def console_process(q: Queue):
    cli = SimConsole()
    while True:
        sim_data: SimulationData = q.get()
        # print("queue size:", q.qsize())
        cli.update(sim_data)
        time.sleep(0.01)

if __name__ == "__main__":
    q = Queue()
    sim_process = Process(target=simulation_process, args=(q,))
    sim_process.start()
    console_process(q)
    sim_process.join()
