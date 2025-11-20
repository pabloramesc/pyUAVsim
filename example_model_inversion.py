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
x_trim, delta_trim = uav.trim(Va=25.0, R_orb=100.0, gamma=np.deg2rad(10.0))

sensors = SensorSystem(uav.state)
sensors.initialize(t=0.0)

filter = ModelInversionFilter(dt)

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
plt.plot(time, estimated_euler[:, 0], label="Estimated Roll", linestyle="--")
plt.ylabel("Roll (deg)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_euler[:, 1], label="True Pitch")
plt.plot(time, estimated_euler[:, 1], label="Estimated Pitch", linestyle="--")
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
plt.plot(time, estimated_rates[:, 0], label="Estimated p", linestyle="--")
plt.ylabel("p (deg/s)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_rates[:, 1], label="True q")
plt.plot(time, estimated_rates[:, 1], label="Estimated q", linestyle="--")
plt.ylabel("q (deg/s)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_rates[:, 2], label="True r")
plt.plot(time, estimated_rates[:, 2], label="Estimated r", linestyle="--")
plt.ylabel("r (deg/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()

# Plot true vs estimated airspeed and altitude
true_Va = np.linalg.norm(uav_states[:, 3:6], axis=1)
estimated_Va = estimated_states[:, 6]
true_h = -uav_states[:, 2]
estimated_h = estimated_states[:, 7]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, true_Va, label="True Airspeed")
plt.plot(time, estimated_Va, label="Estimated Airspeed", linestyle="--")
plt.ylabel("Airspeed (m/s)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, true_h, label="True Altitude")
plt.plot(time, estimated_h, label="Estimated Altitude", linestyle="--")
plt.ylabel("Altitude (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()

# Plot true vs estimated position (pn, pe, pd)
true_pn = uav_states[:, 0]
true_pe = uav_states[:, 1]
true_pd = uav_states[:, 2]
estimated_pn = estimated_states[:, 8]
estimated_pe = estimated_states[:, 9]
estimated_pd = estimated_states[:, 10]
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, true_pn, label="True pn")
plt.plot(time, estimated_pn, label="Estimated pn", linestyle="--")
plt.ylabel("pn (m)")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time, true_pe, label="True pe")
plt.plot(time, estimated_pe, label="Estimated pe", linestyle="--")
plt.ylabel("pe (m)")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(time, true_pd, label="True pd")
plt.plot(time, estimated_pd, label="Estimated pd", linestyle="--")
plt.ylabel("pd (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()

# Plot true vs estimated ground speed and course
from simulator.math.rotation import multi_rotation
ned_velocities = multi_rotation(angles=np.deg2rad(true_euler), values=uav_states[:, 3:6], reverse=True)
true_Vg = np.linalg.norm(ned_velocities, axis=1)
true_course = np.arctan2(ned_velocities[:, 1], ned_velocities[:, 0])
estimated_Vg = estimated_states[:, 11]
estimated_course = estimated_states[:, 12]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, true_Vg, label="True Ground Speed")
plt.plot(time, estimated_Vg, label="Estimated Ground Speed", linestyle="--")
plt.ylabel("Ground Speed (m/s)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(true_course) % 360, label="True Course")
plt.plot(
    time,
    np.rad2deg(estimated_course) % 360,
    label="Estimated Course",
    linestyle="--",
)
plt.ylabel("Course (deg)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()

# Show all plots
plt.show()
