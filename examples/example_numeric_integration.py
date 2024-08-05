"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator.math.numeric_integration import euler, rk4, crank_nicolson

# Time vector
t0 = 0.0
tf = 60.0
dt = 0.01
t = np.arange(t0, tf, dt)

# State arrays to store results
y_eu = np.zeros((t.size, 2))
y_rk = np.zeros((t.size, 2))
y_cn = np.zeros((t.size, 2))

# Set initial conditions
y0 = 0.0
y_eu[0, 0] = y0
y_rk[0, 0] = y0
y_cn[0, 0] = y0

# Define input array
u = np.ones(t.size)

# Define the parameters of the second-order system
wn = 1.0  # Natural frequency (rad/s)
zeta = 0.1  # Damping ratio

# Define the second-order system differential equation:
#
# Transfer function:
#   Y(s)/U(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
#
# Laplace domain representation:
#   s^2*Y(s) + 2*zeta*wn*s*Y(s) + wn^2*Y(s) = wn^2*U(s)
#
# Differential equation:
#   d^2y/dt^2 + 2*zeta*wn * dy/dt + wn^2 * y = wn^2 * u(t)
#
# State-space representation:
#   Let x1 = y (output variable), x2 = dy/dt (derivative of y)
#   dx1/dt = x2
#   dx2/dt = -2*zeta*wn*x2 - wn^2*x1 + wn^2*u(t)
#
# Matrix form (dx/dt = A*x + B*u):
#   A = [[0, 1],
#        [-wn^2, -2*zeta*wn]]
#   B = [[0],
#        [wn^2]]
#
func = lambda x, u: np.array([x[1], -2 * zeta * wn * x[1] - wn**2 * x[0] + wn**2 * u])

# Perform the integration
for k, tk in enumerate(t[1:], 1):
    dfdt = lambda t, y: func(y, u[k])
    dy_eu = euler(dfdt, tk, y_eu[k - 1], dt)
    dy_rk = rk4(dfdt, tk, y_rk[k - 1], dt)
    dy_cn = crank_nicolson(dfdt, tk, y_cn[k - 1], dt)
    y_eu[k] = y_eu[k - 1] + dy_eu
    y_rk[k] = y_rk[k - 1] + dy_rk
    y_cn[k] = y_cn[k - 1] + dy_cn

# Compute theoretical overshoot and rise time
overshoot = 1.0 + np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
rise_time = (np.pi - np.arccos(zeta)) / (wn * np.sqrt(1 - zeta**2))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, y_eu[:, 0], label="Euler")
plt.plot(t, y_rk[:, 0], label="Runge-Kutta 4")
plt.plot(t, y_cn[:, 0], label="Crank-Nicolson")
plt.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label="Final value")
plt.axhline(y=overshoot, color='r', linestyle='--', linewidth=1, label=f'Overshoot: {overshoot*100.0:.2f}%')
plt.axvline(x=rise_time, color='b', linestyle='--', linewidth=1, label=f'Rise Time: {rise_time:.2f}s')
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.title("Step Response of a Second-Order System")
plt.legend()
plt.grid(True)
plt.show()
