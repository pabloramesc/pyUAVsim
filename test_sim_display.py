"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
import numpy as np

# Enable interactive mode
plt.ion()

# Create a figure with a specific layout
fig = plt.figure()

# Add the 3D plot
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_title("3D Helix")
ax1.set_xlim(-1.0, +1.0)
ax1.set_ylim(-1.0, +1.0)
ax1.set_zlim(0.0, 100.0)

# Add the top perspective (XY plane) 2D plot
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Top View (XY plane)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_xlim(-1.0, +1.0)
ax2.set_ylim(-1.0, +1.0)

# Add the frontal perspective (XZ plane) 2D plot
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_title("Frontal View (XZ plane)")
ax3.set_xlabel("X")
ax3.set_ylabel("Z")
ax3.set_xlim(-1.0, +1.0)
ax3.set_ylim(0.0, 100.0)

# Sample data for the helix
N = 1000
t = np.linspace(0, 10 * 2 * np.pi, N)
x = np.sin(t)
y = np.cos(t)
z = t

# Plotting the initial data
(line1,) = ax1.plot(x[0], y[0], z[0], label="3D Helix")

(line2,) = ax2.plot(x[0], y[0], "r-", label="XY plane")

(line3,) = ax3.plot(x[0], z[0], "b-", label="XZ plane")

# Show the plots
plt.show()

# Update the plots interactively
for k in range(1, N):
    # Update the 3D plot
    line1.set_data(x[:k], y[:k])
    line1.set_3d_properties(z[:k])

    # Update the top perspective plot (XY plane)
    line2.set_data(x[:k], y[:k])

    # Update the frontal perspective plot (XZ plane)
    line3.set_data(x[:k], z[:k])

    # Update the figure
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.01)  # Pause to update the plot
