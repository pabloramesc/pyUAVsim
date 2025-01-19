import time
from simulator.aircraft import AircraftDynamics
from simulator.autopilot import Autopilot
from simulator.autopilot import MissionControl, load_waypoints_from_txt

def simulation_process(queue, dt, aerosonde_params, waypoints_file):
    uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
    x_trim, delta_trim = uav.trim(Va=25.0)
    autopilot = Autopilot(dt, aerosonde_params, uav.state)
    waypoints_list = load_waypoints_from_txt(waypoints_file)
    mission = MissionControl(dt, autopilot.config)
    mission.initialize(waypoints_list, Va=25.0, h=0.0, chi=0.0)

    t_sim = 0.0  # simulation time
    k_sim = 0  # simulation steps

    while True:
        t_sim += dt
        k_sim += 1

        uav.update(autopilot.control_deltas)  # update simulation states

        flight_cmd = mission.update(uav.state.ned_position, uav.state.course_angle)
        autopilot.status.update_aircraft_state(uav.state)
        autopilot.control_course_altitude_airspeed(
            flight_cmd.course, flight_cmd.altitude, flight_cmd.airspeed
        )

        queue.put(uav.state)