"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.panel_components import Plot


class TimeSeriesPlot(Plot):
    """
    A class for creating and managing time series plots.

    This class extends the Plot class and provides functionality for plotting
    multiple time series on the same axes.

    Attributes
    ----------
    lines : list of matplotlib.lines.Line2D
        A list of line objects representing each variable in the plot.
    labels : list of str
        Labels for each variable plotted.
    """

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
        """
        Initializes the TimeSeriesPlot component.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this plot belongs.
        pos : int, optional
            The position of the subplot within the figure, by default 111.
        nvars : int, optional
            The number of variables to be plotted, by default 1.
        title : str, optional
            The title of the plot, by default "".
        xlabel : str, optional
            The label for the x-axis, by default "".
        ylabel : str, optional
            The label for the y-axis, by default "".
        labels : list of str, optional
            A list of labels for each variable. If None, default labels will be used, by default None.
        """
        super().__init__(fig, pos, False, nvars)

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid()

        if labels is None:
            self.labels = [f"var{k+1}" for k in range(self.logger.nvars)]
        else:
            self.labels = labels

        self.lines: list[plt.Line2D] = []
        for k in range(self.logger.nvars):
            line, = self.ax.plot([], [], linewidth=2.0, label=self.labels[k])
            self.lines.append(line)
            
        self.setup_blit(self.lines)

    def update_plot(self) -> None:
        """
        Updates the plot with the latest logged data.

        The method retrieves the logged data from the DataLogger and updates
        the lines in the plot, adjusting the axes to fit the new data.
        """
        log = self.logger.as_array()
        for k in range(self.logger.nvars):
            self.lines[k].set_data(log[:, 0], log[:, k + 1])
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.render()
