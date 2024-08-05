"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator.aircraft import load_airframe_parameters_from_yaml, AerodynamicModel

params_file = r"config/aerosonde_parameters.yaml"
aerosonde_params = load_airframe_parameters_from_yaml(params_file)

model = AerodynamicModel(aerosonde_params)

# Define alpha range from -90 to 90 degrees
alpha_degrees = np.linspace(-90, 90, 360)
alpha_radians = np.radians(alpha_degrees)

# Calculate CL and CD for both models
CL_accurate = [
    model.lift_coefficient_vs_alpha(alpha, model="accurate") for alpha in alpha_radians
]
CL_linear = [
    model.lift_coefficient_vs_alpha(alpha, model="linear") for alpha in alpha_radians
]
CD_accurate = [
    model.drag_coefficient_vs_alpha(alpha, model="quadratic") for alpha in alpha_radians
]
CD_linear = [
    model.drag_coefficient_vs_alpha(alpha, model="linear") for alpha in alpha_radians
]


plt.figure(figsize=(10, 5))

# Plot CL vs alpha
plt.subplot(1, 2, 1)
plt.plot(alpha_degrees, CL_accurate, label="Accurate Model")
plt.plot(alpha_degrees, CL_linear, label="Linear Model")
plt.xlabel("Alpha (degrees)")
plt.ylabel("CL")
plt.title("CL vs Alpha")
plt.legend()
plt.grid()

# Plot CD vs alpha
plt.subplot(1, 2, 2)
plt.plot(alpha_degrees, CD_accurate, label="Quadratic Model")
plt.plot(alpha_degrees, CD_linear, label="Linear Model")
plt.xlabel("Alpha (degrees)")
plt.ylabel("CD")
plt.title("CD vs Alpha")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
