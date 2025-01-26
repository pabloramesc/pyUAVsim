"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import time
from multiprocessing import Process, Queue

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from simulator.gui import AttitudePositionPanel

from simulator.utils.simulation_data import SimulationData, SimulationDataConnector
from simulator.aircraft import (
    AircraftDynamics,
    load_airframe_parameters_from_yaml,
)
from simulator.autopilot import Autopilot
from simulator.autopilot.mission_control import MissionControl
from simulator.autopilot.waypoints import load_waypoints_from_txt
from simulator.cli import SimConsole
from simulator.gui.attitude_position_animation import AttitudePositionAnimation


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

    t0 = time.time()
    sim_t = 0.0  # simulation time
    sim_k = 0  # simulation steps
    while True:
        sim_t += dt
        sim_k += 1

        uav.update(autopilot.control_deltas)  # update simulation states

        flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
        autopilot.status.update_aircraft_state(uav.state)
        autopilot.control_course_altitude_airspeed(
            flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
        )

        q.put(
            SimulationData(
                dt,
                sim_t,
                sim_k,
                uav.state,
                uav.control_deltas,
                autopilot.status,
                mission,
                mission.route_manager,
            )
        )

        if sim_t > time.time() - t0:
            time.sleep(max(dt, sim_t - (time.time() - t0)))

def visualization_process(q: Queue):
    cli = SimConsole()
    gui = AttitudePositionPanel()

    while True:
        sim_data: SimulationData = None
        while True:
            try:
                sim_data = q.get_nowait()
                gui.add_data(sim_data.uav_state)
            except:
                break

        if sim_data:
            cli.update(sim_data, q.qsize())
            gui.update(sim_data.uav_state, pause=0.01)

def visualization_process2(q: Queue):
    # Crear la figura y el eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Posición en tiempo real")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Height (m)")

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-100, 100)

    # Inicializar las listas de datos para la posición
    pos_log = ([], [], [])
    line, = ax.plot([], [], [])

    def update(frame):
        sim_data: SimulationData = None
        while True:
            try:
                sim_data = q.get_nowait()
                pos_log[0].append(sim_data.uav_state.pe)
                pos_log[1].append(sim_data.uav_state.pn)
                pos_log[2].append(-sim_data.uav_state.pd)
            except:
                break
            
        if sim_data:
            print(q.qsize())
            line.set_data(pos_log[0], pos_log[1])
            line.set_3d_properties(pos_log[2])

        if len(pos_log[0]) > 1:
            ax.set_xlim(min(pos_log[0]), max(pos_log[0]))
            ax.set_ylim(min(pos_log[1]), max(pos_log[1]))
            ax.set_zlim(min(pos_log[2]), max(pos_log[2]))

        # fig.canvas.flush_events()
        plt.pause(0.01)  # Pausa para permitir que se renderice el gráfico

        return line,

    while True:
        update(None)

    # ani = FuncAnimation(fig, update, frames=1, blit=True)
    # plt.show()


if __name__ == "__main__":
    q = Queue()

    sim_process = Process(target=simulation_process, args=(q,))
    vis_process = Process(target=visualization_process2, args=(q,))

    sim_process.start()
    vis_process.start()

    sim_process.join()
    vis_process.join()

