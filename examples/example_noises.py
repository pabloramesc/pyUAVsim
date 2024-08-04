"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from matplotlib import pyplot as plt
import numpy as np
from simulator.sensors.noise_models import get_white_noise, get_brown_noise, get_pink_noise

print("Testing noise functions...")

dt = 1e-3
t = np.arange(0.0, 60.0, dt)
N = t.size

Nd = 8.4553e-03  # noise density
Bi = 2.6309e-03  # bias instability
Rw = 5.7262e-05  # random walk
Tc = 1.6e03  # correlation time

# white noise test
wn1 = np.zeros(N)
for k in range(N):
    wn1[k] = get_white_noise(Nd, dt, nlen=1)
wn2 = get_white_noise(Nd, dt, nlen=N)
plt.plot(t, wn1, label="wn1")
plt.plot(t, wn2, label="wn2")
plt.legend()
plt.show()

# pink noise test
pn1 = np.zeros(N)
pn_prev = 0.0
for k in range(N):
    pn1[k] = get_pink_noise(Bi, Tc, dt, pn_prev, nlen=1)
    pn_prev = pn1[k]
pn2 = get_pink_noise(Bi, Tc, dt, 0.0, nlen=N)
plt.plot(t, pn1, label="pn1")
plt.plot(t, pn2, label="pn2")
plt.legend()
plt.show()

# brown noise test
bn1 = np.zeros(N)
bn_prev = 0.0
for k in range(N):
    bn1[k] = get_brown_noise(Rw, dt, bn_prev, nlen=1)
    bn_prev = bn1[k]
bn2 = get_brown_noise(Rw, dt, 0.0, nlen=N)
plt.plot(t, bn1, label="bn1")
plt.plot(t, bn2, label="bn2")
plt.legend()
plt.show()
