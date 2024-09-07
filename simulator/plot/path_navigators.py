"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from simulator.autopilot.route_manager import RouteManager
from simulator.autopilot.path_navigator import (
    PathNavigator,
    LinePathNavigator,
    FilletPathNavigator,
    DubinPathNavigator,
)
from simulator.plot.base_plotter import BasePlotter
from simulator.plot.waypoints_manager import WaypointsManagerPlotter


class PathNavigatorPlotter(WaypointsManagerPlotter):

    def __init__(
        self, path_nav: PathNavigator, ax: plt.Axes = None, is_3d: bool = True
    ) -> None:
        super().__init__(path_nav.wps_manager, ax, is_3d)
        self.path_nav = path_nav

    @abstractmethod
    def plot_paths(self) -> None:
        pass

    def _get_waypoints(self) -> np.ndarray:
        wps = self.wps_manager.wp_coords
        if wps is None:
            raise ValueError("Waypoints have not been set.")
        return wps


class LinePathNavigatorPlotter(PathNavigatorPlotter):

    def __init__(
        self, path_nav: LinePathNavigator, ax: plt.Axes = None, is_3d: bool = True
    ) -> None:
        super().__init__(path_nav, ax, is_3d)

    def plot_paths(self) -> None:
        self.plot_waypoints(style="ro-")


class FilletPathNavigatorPlotter(PathNavigatorPlotter):

    def __init__(
        self,
        path_nav: FilletPathNavigator,
        ax: plt.Axes = None,
        is_3d: bool = True,
    ) -> None:
        super().__init__(path_nav, ax, is_3d)

    def plot_paths(self) -> None:
        raise NotImplementedError


class DubinPathNavigatorPlotter(PathNavigatorPlotter):

    def __init__(
        self,
        path_nav: DubinPathNavigator,
        ax: plt.Axes = None,
        is_3d: bool = True,
    ) -> None:
        super().__init__(path_nav, ax, is_3d)

    def plot_paths(self) -> None:
        raise NotImplementedError
