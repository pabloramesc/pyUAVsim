"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time

from simulator.common.angles import rot_matrix_zyx, ned2xyz

SPAN = 14.0
CHORD = 2.0
LENGTH = 10.0
HEIGHT = 1.0
TAIL_HEIGHT = 2.0
TAIL_SPAN = 4.0
TAIL_ROOT = 2.0
TAIL_TIP = 1.0

# Enable interactive mode
plt.ion()

# Create a figure with a specific layout
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Real-Time Aircraft Orientation")
ax.set_xlabel("E (East)")
ax.set_ylabel("N (North)")
ax.set_zlabel("H (Height)")
ax.set_xlim(-0.6 * SPAN, 0.6 * SPAN)
ax.set_ylim(-0.6 * SPAN, 0.6 * SPAN)
ax.set_zlim(-0.6 * SPAN, 0.6 * SPAN)

# Define the vertices of the aircraft components
wing = np.array(
    [
        [-CHORD, -0.5 * SPAN, 0],
        [0, -0.5 * SPAN, 0],
        [0, +0.5 * SPAN, 0],
        [-CHORD, +0.5 * SPAN, 0],
    ]
)
body = np.array(
    [
        [0.2 * LENGTH, 0, 0],
        [0.2 * LENGTH, 0, HEIGHT],
        [-0.8 * LENGTH, 0, HEIGHT],
        [-0.8 * LENGTH, 0, 0],
    ]
)
htail = np.array(
    [
        [-0.8 * LENGTH + TAIL_ROOT, 0, 0],
        [-0.8 * LENGTH, 0, 0],
        [-0.8 * LENGTH, 0, -TAIL_HEIGHT],
        [-0.8 * LENGTH + TAIL_TIP, 0, -TAIL_HEIGHT],
    ]
)
vtail = np.array(
    [
        [-0.8 * LENGTH + TAIL_ROOT, 0, 0],
        [-0.8 * LENGTH + TAIL_TIP, -0.5 * TAIL_SPAN, 0],
        [-0.8 * LENGTH, -0.5 * TAIL_SPAN, 0],
        [-0.8 * LENGTH, +0.5 * TAIL_SPAN, 0],
        [-0.8 * LENGTH + TAIL_TIP, +0.5 * TAIL_SPAN, 0],
        [-0.8 * LENGTH + TAIL_ROOT, 0, 0],
    ]
)

# Create a collection of the aircraft polygons
poly = Poly3DCollection(
    [ned2xyz(body), ned2xyz(wing), ned2xyz(htail), ned2xyz(vtail)]
)
poly.set_facecolors(["w", "r", "r", "r"])
poly.set_edgecolor("k")
poly.set_alpha(0.8)
ax.add_collection(poly)

# Show the plots
plt.show()

# Update the plots interactively
t0 = time.time()
while True:
    # Update the angles
    t = time.time() - t0
    roll = 0.0 + np.pi * np.sin(2*np.pi*0.04*t)
    pitch = 0.0 + 0.5 * np.pi * np.sin(2 * np.pi * 0.02 * t)
    yaw = 0.0 + np.pi * np.sin(2 * np.pi * 0.01 * t)

    # Rotate the components
    R_vb = rot_matrix_zyx(np.array([roll, pitch, yaw]))
    rotated_body = body.dot(R_vb)
    rotated_wing = wing.dot(R_vb)
    rotated_htail = htail.dot(R_vb)
    rotated_vtail = vtail.dot(R_vb)

    # Update the polygon vertices
    poly.set_verts([
        ned2xyz(rotated_body),
        ned2xyz(rotated_wing),
        ned2xyz(rotated_htail),
        ned2xyz(rotated_vtail)
    ])

    # Update the figure
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(
        f"t: {t:.2f} s , roll: {np.rad2deg(roll):.2f} deg , pitch: {np.rad2deg(pitch):.2f} deg , yaw: {np.rad2deg(yaw):.2f} deg"
    )

    plt.pause(0.01)  # Pause to update the plot