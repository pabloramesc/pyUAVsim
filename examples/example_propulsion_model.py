"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator.aircraft import load_airframe_parameters_from_yaml, PropulsionModel, AircraftState, ControlDeltas

# Load parameters from yaml file
params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)
Vmax = aerosonde_params.Vmax

# Initialize the ForcesMoments model
model = PropulsionModel(aerosonde_params)

# Define airspeed range and throttle settings
airspeed = np.linspace(0.0, 30.0, 1000)
delta_throttle = np.linspace(0.1, 1.0, 8)

# Initialize arrays to store thrust and torque results
thrust_results = []
torque_results = []

# Calculate thrust and torque for each throttle setting and airspeed
for dt in delta_throttle:
    thrust = []
    torque = []
    deltas = ControlDeltas(np.array([0.0, 0.0, 0.0, dt]))
    for va in airspeed:
        state = AircraftState(np.array([0.0, 0.0, 0.0, va, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        f = model.propulsion_force(state, deltas)
        m = model.propulsion_moment(state, deltas)
        thrust.append(f)
        torque.append(m)
    thrust_results.append(thrust)
    torque_results.append(torque)


plt.figure(figsize=(10, 5))

# Plotting thrust versus airspeed
plt.subplot(1, 2, 1)
for i, dt in enumerate(delta_throttle):
    plt.plot(airspeed, thrust_results[i], label=f'{dt*Vmax:.1f} V')
plt.xlabel('airspeed (m/s)')
plt.ylabel('thrust (N)')
plt.legend(title='Voltage')
plt.title('Propeller thrust vs airspeed')
plt.grid(True)

# Plotting torque versus airspeed
plt.subplot(1, 2, 2)
for i, dt in enumerate(delta_throttle):
    plt.plot(airspeed, torque_results[i], label=f'{dt*Vmax:.1f} V')
plt.xlabel('airspeed (m/s)')
plt.ylabel('torque (Nm)')
plt.legend(title='Voltage')
plt.title('Motor torque vs airspeed')
plt.grid(True)

plt.tight_layout()
plt.show()
