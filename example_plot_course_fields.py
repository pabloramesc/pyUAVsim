
import numpy as np

from matplotlib import pyplot as plt

from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.path_following import LineFollower, OrbitFollower

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
uav.trim(25.0, np.deg2rad(10.0), 500, update=True)

autopilot_config = AutopilotConfig()
autopilot_config.calculate_control_gains(aerosonde_params, uav.state)

line_follower = LineFollower(autopilot_config)
line_follower.set_line(np.array([2e3, 1e3, -50.0]), np.array([1.0, 1.0, 0.0]))
line_follower.plot_course_field()

orbit_follower = OrbitFollower(autopilot_config)
orbit_follower.set_orbit(np.array([2e3, 1e3, -50.0]), 500.0, +1)
orbit_follower.plot_course_field()