"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.panel_components import Plot


class TimeSeriesPlot(Plot):

    def __init__(
        self,
        fig: plt.Figure,
        pos: int = 111,
        nvars: int = 1,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        labels: list[str] = None,
    ) -> None:
        super().__init__(fig, pos, False, nvars)

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid()

        if labels is None:
            self.labels = []
            for k in range(self.logger.nvars):
                self.labels.append(f"var{k+1}")
        else:
            self.labels = labels

        self.lines = []
        for k in range(self.logger.nvars):
            line, = self.ax.plot([], [], linewidth=2.0, label=self.labels[k])
            self.lines.append(line)
            
        self.setup_blit(self.lines)

    def update_plot(self) -> None:
        """
        Updates the plot with the latest logged data.

        The method retrieves the logged data and updates the plot's lines,
        adjusting the axes to fit the new data.
        """
        log = self.logger.as_array()
        for k in range(self.logger.nvars):
            self.lines[k].set_data(log[:, 0], log[:, k + 1])
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.render()