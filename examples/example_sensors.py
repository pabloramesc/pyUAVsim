"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import tqdm
import numpy as np
from simulator.aircraft import AircraftDynamics, load_airframe_parameters_from_yaml
from simulator.sensors.sensor_system import SensorSystem

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

dt = 0.01
uav = AircraftDynamics(dt, aerosonde_params, use_quat=True)
x_trim, delta_trim = uav.trim(Va=25.0, R_orb=100.0, gamma=np.deg2rad(10.0))

sensors = SensorSystem(uav.state)
sensors.initialize(t=0.0)

# Create arrays to store simulation data
sim_steps = int(100 / dt)
uav_states_size = (
    len(uav.state.x) + 3
)  # pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r, ax, ay, az
uav_states = np.zeros((sim_steps, uav_states_size))
sensor_readings_size = len(
    sensors.read(0.0).as_array()
)  # ax, ay, az, p, q, r, abs pressure, diff pressure, heading, pn, pe, h, Vg, chi
sensor_readings = np.zeros((sim_steps, sensor_readings_size))

t = 0.0
for k in tqdm.tqdm(range(sim_steps)):
    t += dt
    uav.update()

    sensors.update(t)
    readings = sensors.read(t)

    uav_states[k, :13] = uav.state.x
    uav_states[k, 13:] = uav.state.body_acceleration
    sensor_readings[k, :] = readings.as_array()

# Plotting the results
import matplotlib.pyplot as plt
from simulator.math.rotation import quat2euler

time = np.arange(0, sim_steps * dt, dt)

# Plot true accelerations vs accelerometer readings
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1.plot(time, uav_states[:, 13], label="True ax (g)")
ax1.plot(time, sensor_readings[:, 0], label="Accel ax (g)", alpha=0.7)
ax1.set_ylabel("Acceleration X (g)")
ax1.legend()
ax2 = fig.add_subplot(312)
ax2.plot(time, uav_states[:, 14], label="True ay (g)")
ax2.plot(time, sensor_readings[:, 1], label="Accel ay (g)", alpha=0.7)
ax2.set_ylabel("Acceleration Y (g)")
ax2.legend()
ax3 = fig.add_subplot(313)
ax3.plot(time, uav_states[:, 15], label="True az (g)")
ax3.plot(time, sensor_readings[:, 2], label="Accel az (g)", alpha=0.7)
ax3.set_ylabel("Acceleration Z (g)")
ax3.set_xlabel("Time (s)")
ax3.legend()
plt.tight_layout()

# Plot true angular rates vs gyro readings
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1.plot(time, uav_states[:, 10], label="True p (rad/s)")
ax1.plot(time, sensor_readings[:, 3], label="Gyro p (rad/s)", alpha=0.7)
ax1.set_ylabel("Roll Rate (rad/s)")
ax1.legend()
ax2 = fig.add_subplot(312)
ax2.plot(time, uav_states[:, 11], label="True q (rad/s)")
ax2.plot(time, sensor_readings[:, 4], label="Gyro q (rad/s)", alpha=0.7)
ax2.set_ylabel("Pitch Rate (rad/s)")
ax2.legend()
ax3 = fig.add_subplot(313)
ax3.plot(time, uav_states[:, 12], label="True r (rad/s  )")
ax3.plot(time, sensor_readings[:, 5], label="Gyro r (rad/s)", alpha=0.7)
ax3.set_ylabel("Yaw Rate (rad/s)")
ax3.set_xlabel("Time (s)")
ax3.legend()
plt.tight_layout()

# Plot true altitude and airspeed vs barometric readings
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax1.plot(time, -uav_states[:, 2], label="True Altitude (m)")
h = (101.325 - sensor_readings[:, 6]) * 1e3 / (
    1.225 * 9.81
)  # Convert pressure to altitude
ax1.plot(time, h, label="Baro Altitude (m)", alpha=0.7)
ax1.set_ylabel("Altitude (m)")
ax1.legend()
ax2 = fig.add_subplot(212)
Va = np.sqrt(uav_states[:, 3] ** 2 + uav_states[:, 4] ** 2 + uav_states[:, 5] ** 2)
ax2.plot(time, Va, label="True Airspeed (m/s)")
Va = np.sqrt(
    2 / 1.225 * sensor_readings[:, 7] * 1e3
)  # Convert dynamic pressure to airspeed
ax2.plot(time, Va, label="Sensor Airspeed (m/s)", alpha=0.7)
ax2.set_ylabel("Airspeed (m/s)")
ax2.set_xlabel("Time (s)")
plt.tight_layout()

# Plot true position vs GPS readings
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1.plot(time, uav_states[:, 0], label="True pn (m)")
ax1.plot(time, sensor_readings[:, 9], label="GPS pn (m)", alpha=0.7)
ax1.set_ylabel("Position North (m)")
ax1.legend()
ax2 = fig.add_subplot(312)
ax2.plot(time, uav_states[:, 1], label="True pe (m)")
ax2.plot(time, sensor_readings[:, 10], label="GPS pe (m)", alpha=0.7)
ax2.set_ylabel("Position East (m)")
ax2.legend()
ax3 = fig.add_subplot(313)
ax3.plot(time, -uav_states[:, 2], label="True h (m)")
ax3.plot(time, sensor_readings[:, 11], label="GPS h (m)", alpha=0.7)
ax3.set_ylabel("Position Down (m)")
ax3.set_xlabel("Time (s)")
ax3.legend()
plt.tight_layout()

# Plot true heading and ground speed vs compass and GPS ground speed readings
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
true_heading = np.rad2deg(quat2euler(uav_states[:, 6:10])[:, 2]) % 360.0
ax1.plot(time, true_heading, label="True Heading (deg)")
ax1.plot(time, sensor_readings[:, 8], label="Compass Heading (deg)")
ax1.plot(time, sensor_readings[:, 13], label="GPS Heading (deg)", alpha=0.7)
ax1.set_ylabel("Heading (deg)")
ax1.legend()
ax2 = fig.add_subplot(212)
Vg = np.sqrt(uav_states[:, 3] ** 2 + uav_states[:, 4] ** 2)
ax2.plot(time, Vg, label="True Ground Speed (m/s)")
ax2.plot(time, sensor_readings[:, 12], label="GPS Ground Speed (m/s)", alpha=0.7)
ax2.set_ylabel("Ground Speed (m/s)")
ax2.set_xlabel("Time (s)")
ax2.legend()
plt.tight_layout()

plt.show()
