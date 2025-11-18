"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import tqdm
import numpy as np
from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.sensors.sensor_system import SensorSystem
from simulator.estimation.model_inversion import ModelInversionFilter

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
x_trim, delta_trim = uav.trim(Va=25.0, R_orb=500, gamma=np.deg2rad(10.0))

sensors = SensorSystem(uav.state)
sensors.initialize(t=0.0)

filter = ModelInversionFilter()

# Create arrays to store simulation data
sim_steps = int(100 / dt)
uav_states_size = len(uav.state.x)  # pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r
uav_states = np.zeros((sim_steps, uav_states_size))
estimated_states_size = 13  # roll, pitch, yaw, p, q, r, Va, h, pn, pe, pd, Vg, heading
estimated_states = np.zeros((sim_steps, estimated_states_size))

t = 0.0
for k in tqdm.tqdm(range(sim_steps)):
    t += dt
    uav.update()

    sensors.update(t)
    readings = sensors.read(t)
    estimated_state = filter.update(readings)

    uav_states[k, :13] = uav.state.x
    estimated_states[k, :] = estimated_state.as_array()

# Plotting the results
import matplotlib.pyplot as plt
from simulator.math.rotation import quat2euler

time = np.arange(0, sim_steps * dt, dt)

# Plot true vs estimated roll, pitch, yaw
true_euler = np.rad2deg(quat2euler(uav_states[:, 6:10]))
estimated_euler = np.rad2deg(estimated_states[:, 0:3])
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_euler[:, 0], label="True Roll")
plt.plot(
    time, estimated_euler[:, 0], label="Estimated Roll", linestyle="--"
)
plt.ylabel("Roll (deg)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_euler[:, 1], label="True Pitch")
plt.plot(
    time, estimated_euler[:, 1], label="Estimated Pitch", linestyle="--"
)
plt.ylabel("Pitch (deg)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_euler[:, 2] % 360, label="True Yaw")
plt.plot(time, estimated_euler[:, 2], label="Estimated Yaw", linestyle="--")
plt.ylabel("Yaw (deg)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()

# Plot true vs estimated angular rates p, q, r
true_rates = np.rad2deg(uav_states[:, 10:13])
estimated_rates = np.rad2deg(estimated_states[:, 3:6])
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_rates[:, 0], label="True p")
plt.plot(
    time, estimated_rates[:, 0], label="Estimated p", linestyle="--"
)
plt.ylabel("p (deg/s)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_rates[:, 1], label="True q")
plt.plot(
    time, estimated_rates[:, 1], label="Estimated q", linestyle="--"
)
plt.ylabel("q (deg/s)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_rates[:, 2], label="True r")
plt.plot(
    time, estimated_rates[:, 2], label="Estimated r", linestyle="--"
)
plt.ylabel("r (deg/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()



plt.show()
