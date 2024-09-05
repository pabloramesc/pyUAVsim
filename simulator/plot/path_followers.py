"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.autopilot.line_follower import LineFollower
from simulator.autopilot.orbit_follower import OrbitFollower
from simulator.plot.base_plotter import BasePlotter


class LineFollowerPlotter(BasePlotter):
    """Class for plotting the course field of a LineFollower."""

    def __init__(self, line_follower: LineFollower, ax: plt.Axes = None) -> None:
        """
        Initialize the LineFollowerPlotter with a LineFollower instance.

        Parameters
        ----------
        line_follower : LineFollower
            The LineFollower instance for which the course field will be plotted.
        ax : plt.Axes, optional
            _description_, by default None
        """
        super().__init__(ax, is_3d=False)

        self.line_follower = line_follower

        self.ax.set_title("Line Following Course Field")
        self.ax.set_ylabel("North position (m)")
        self.ax.set_xlabel("East position (m)")
        self.ax.set_aspect("equal")

    def plot_course_field(self, density: int = 20, distance: float = 100.0) -> None:
        """
        Plot the course field for line following.

        Parameters
        ----------
        density : int, optional
            The density of the grid for plotting (default is 20).
        distance : float, optional
            The distance range from the path origin for plotting (default is 100.0).
        """
        # Extract initial position and direction
        pn0, pe0, _ = self.line_follower.path_origin
        qn0, qe0, _ = self.line_follower.path_direction

        # Create position grids for pn and pe
        pn = np.linspace(pn0 - distance, pn0 + distance, density)
        pe = np.linspace(pe0 - distance, pe0 + distance, density)
        pn, pe = np.meshgrid(pn, pe)

        # Compute the course for each grid point
        course = np.array(
            [
                self.line_follower.lateral_guidance(np.array([pni, pei, 0.0]))
                for pni, pei in zip(pn.flatten(), pe.flatten())
            ]
        ).reshape(density, density)

        # Calculate the direction vectors
        qn, qe = np.cos(course), np.sin(course)

        # Plotting
        self.ax.plot(pe0, pn0, "ro")  # Plot initial position
        self.ax.plot(
            [pe0, pe0 + distance * qe0], [pn0, pn0 + distance * qn0], "r-"
        )  # Line path
        self.ax.plot(
            [pe0, pe0 - distance * qe0], [pn0, pn0 - distance * qn0], "r--"
        )  # Opposite line path
        self.ax.quiver(pe, pn, qe, qn)  # Plot vector arrows


class OrbitFollowerPlotter(BasePlotter):
    """Class for plotting the course field of an OrbitFollower."""

    def __init__(self, orbit_follower: OrbitFollower, ax: plt.Axes = None) -> None:
        """
        Initialize the OrbitFollowerPlotter with an OrbitFollower instance.

        Parameters
        ----------
        orbit_follower : OrbitFollower
            The OrbitFollower instance for which the course field will be plotted.
        ax : plt.Axes, optional
            _description_, by default None
        """
        super().__init__(ax, is_3d=False)

        self.orbit_follower = orbit_follower

        self.ax.set_title("Orbit Following Course Field")
        self.ax.set_ylabel("North position (m)")
        self.ax.set_xlabel("East position (m)")
        self.ax.set_aspect("equal")

    def plot_course_field(self, density: int = 20) -> None:
        """
        Plot the course field for orbit following.

        Parameters
        ----------
        density : int, optional
            The density of the grid for plotting (default is 20).
        """
        # Extract orbit center and radius
        cn, ce, _ = self.orbit_follower.orbit_center
        r = self.orbit_follower.orbit_radius

        # Create position grids for pn and pe
        pn, pe = np.meshgrid(
            np.linspace(cn - 2 * r, cn + 2 * r, density),
            np.linspace(ce - 2 * r, ce + 2 * r, density),
        )
        d = np.sqrt((pn - cn) ** 2 + (pe - ce) ** 2)
        psi = np.arctan2(pe - ce, pn - cn)

        # Compute the course for each grid point
        course = np.array(
            [
                self.orbit_follower.lateral_guidance(np.array([pni, pei, 0.0]))
                for pni, pei in zip(pn.flatten(), pe.flatten())
            ]
        ).reshape(density, density)

        # Calculate the direction vectors
        qn, qe = np.cos(course), np.sin(course)

        # Plotting
        self.ax.plot(ce, cn, "ro")  # Plot orbit center
        self.plot_horizontal_circle([cn, ce], r, style="r--")  # Orbit path
        self.ax.quiver(pe, pn, qn, qe)  # Plot vector arrows
        plt.show()
