import numpy as np
from matplotlib import pyplot as plt

from simulator.autopilot.route_manager import RouteManager
from simulator.plot.base_plotter import BasePlotter


class WaypointsManagerPlotter(BasePlotter):

    def __init__(
        self, wps_manager: RouteManager, ax: plt.Axes = None, is_3d: bool = True
    ) -> None:
        super().__init__(ax, is_3d)

        self.wps_manager = wps_manager

        self.ax.set_title("Waypoint positions in NED frame")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.grid(True)

    def plot_waypoints(self, wp_areas: bool = False, style: str = "ro") -> None:
        """
        Plot the waypoints on a 2D graph.
        """
        wps = self.wps_manager.wp_coords
        if wps is None:
            raise ValueError("Waypoints have not been set.")

        self.ax.plot(wps[:, 0], wps[:, 1], wps[:, 2] if self.is_3d else None, style)

        if wp_areas:
            r = self.wps_manager.config.wp_default_radius
            for wp in wps:
                self.plot_horizontal_circle(wp, r, style="y-")

    def plot_end_orbit(self) -> None:
        c = self.wps_manager.get_waypoint_coords(-1)
        r = self.wps_manager.config.wait_orbit_radius
        self.plot_horizontal_circle(c, r, style="r--")
