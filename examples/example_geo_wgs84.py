"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator.environment.geo_constants import *
from simulator.environment.geo import geo_to_wgs84

# Define some points in (lat, long) format with their corresponding names
points = [
    (34.0, -118.0, "Los Angeles"),  # Los Angeles
    (40.7, -74.0, "New York"),  # New York
    (51.5, -0.1, "London"),  # London
    (35.7, 139.7, "Tokyo"),  # Tokyo
]

# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = WGS84_EQUATORIAL_RADIUS * np.outer(np.cos(u), np.sin(v))
y = WGS84_EQUATORIAL_RADIUS * np.outer(np.sin(u), np.sin(v))
z = WGS84_EQUATORIAL_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

# Set up the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, color="lightblue", alpha=0.5)

# Convert geodetic points to ECEF coordinates and plot them with labels
for lat, long, name in points:
    coords = geo_to_wgs84(lat, long)
    ax.scatter(coords[0], coords[1], coords[2], color="blue", s=50)  # Point size 50
    ax.text(
        coords[0], coords[1], coords[2], name, color="black", fontsize=10, ha="right"
    )

# Draw the equator (latitude = 0)
equator_lat = 0  # Latitude = 0 for the equator
equator_long = np.linspace(0, 360, 100)  # Full circle around the globe
equator_coords = np.array(
    [geo_to_wgs84(equator_lat, long) for long in equator_long]
).T  # Convert to ECEF
ax.plot(
    equator_coords[0],
    equator_coords[1],
    equator_coords[2],
    color="red",
    linewidth=2,
    label="Equator",
)

# Draw the Greenwich meridian (longitude = 0)
meridian_lat = np.linspace(-90, 90, 100)  # From -90 to 90 degrees latitude
meridian_long = 0  # Longitude = 0 (Greenwich)
meridian_coords = geo_to_wgs84(meridian_lat, meridian_long)  # Convert to ECEF
ax.plot(
    meridian_coords[0],
    meridian_coords[1],
    meridian_coords[2],
    color="green",
    linewidth=2,
    label="Greenwich Meridian",
)

# Draw Earth's axis
axis_x = np.zeros(100)
axis_y = np.zeros_like(axis_x)
axis_z = np.linspace(-WGS84_EQUATORIAL_RADIUS, WGS84_EQUATORIAL_RADIUS, 100)  # z varies
ax.plot(axis_x, axis_y, axis_z, color="black", linewidth=2, label="Earth's Axis")

# Labels and title
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.set_title(
    "3D Sphere with Geodetic Points, Equator, Greenwich Meridian, and Earth's Axis"
)
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is equal
ax.legend()

plt.show()
