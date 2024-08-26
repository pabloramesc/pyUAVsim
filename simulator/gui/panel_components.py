"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from simulator.aircraft import AircraftState
from simulator.utils.logger import DataLogger

class PanelComponent(ABC):
    def __init__(
        self,
        fig: plt.Figure,
        pos: int | tuple,
        is_3d: bool = False,
    ):
        self.fig = fig
        if isinstance(pos, int):
            self.ax = self.fig.add_subplot(pos, projection='3d' if is_3d else None)
        else:
            self.ax = self.fig.add_subplot(*pos, projection='3d' if is_3d else None)
        self.is_3d = is_3d
        
        self.use_blit = True
        self.background = None
        self.render_objects = []

    def _render_normal(self) -> None:
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()

    def _render_blit(self) -> None:
        self.fig.canvas.restore_region(self.background)
        for obj in self.render_objects:
            self.ax.draw_artist(obj)
        self.fig.canvas.blit(self.ax.bbox)
        
    def setup_blit(self, render_objects: list) -> None:
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.render_objects = render_objects

    def render(self) -> None:
        if self.use_blit:
            self._render_blit()
        else:
            self._render_normal()


class View(PanelComponent):
    def __init__(self, fig: plt.Figure, pos: int, is_3d: bool = False):
        super().__init__(fig, pos, is_3d)

    @abstractmethod
    def update(self, values: np.ndarray) -> None:
        """Interface method to update the state of the view"""
        pass


class Plot(PanelComponent):
    def __init__(
        self,
        fig: plt.Figure,
        pos: int = 111,
        is_3d: bool = False,
        nvars: int = 1,
    ):
        if not isinstance(nvars, int) and nvars <= 0:
            raise ValueError("nvars must be an integer greater than 0!")
        
        super().__init__(fig, pos, is_3d)

        self.logger = DataLogger(nvars)

    def add_data(self, values: np.ndarray, time: float = None) -> None:
        self.logger.update(time, values)

    @abstractmethod
    def update_plot(self) -> None:
        """Interface method to update the plot view"""
        pass

    def update(self, values: np.ndarray, time: float = None) -> None:
        self.add_data(values, time)
        self.update_plot()
        
        
