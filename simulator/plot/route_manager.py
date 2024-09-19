import numpy as np
from matplotlib import pyplot as plt

from simulator.autopilot.route_manager import RouteManager
from simulator.plot.base_plotter import BasePlotter
from simulator.math.rotation import ned2xyz


class RouteManagerPlotter(BasePlotter):

    def __init__(
        self, route_manager: RouteManager, ax: plt.Axes = None, is_3d: bool = True
    ) -> None:
        super().__init__(ax, is_3d)

        self.route_manager = route_manager

        self.ax.set_title("Waypoint positions in NED frame")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        if self.is_3d:
            self.ax.set_zlabel("Height (m)")
        self.ax.grid(True)

    def plot_waypoints(
        self,
        paths: bool = True,
        wp_areas: bool = False,
        wp_style: str = "ro",
        path_style: str = "r--",
    ) -> None:
        """
        Plot the waypoints on a 2D graph.
        """
        wps = self.route_manager.wp_coords
        if wps is None:
            raise ValueError("Waypoints have not been set.")

        xyz = ned2xyz(wps)
        if paths:
            self.ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2] if self.is_3d else None, wp_style
            )
        self.ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2] if self.is_3d else None, path_style
        )

        if wp_areas:
            r = self.route_manager.config.wp_default_radius
            for wp in wps:
                self.plot_horizontal_circle(wp, r, style="y-")

        self.ax.set_box_aspect(
            (np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 20*np.ptp(xyz[:, 2])) if self.is_3d else None
        )  # aspect ratio is 1:1:20 to see better the height

    def plot_end_orbit(self) -> None:
        c = self.route_manager.get_waypoint_coords(-1)
        r = self.route_manager.config.wait_orbit_radius
        self.plot_horizontal_circle(c, r, style="r--")
