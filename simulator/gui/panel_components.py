"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from simulator.utils.logger import DataLogger


class PanelComponent(ABC):
    """
    Base class for panel components used in rendering plots within a figure.

    Attributes
    ----------
    fig : plt.Figure
        The matplotlib figure object that contains the panel.
    ax : plt.Axes
        The axes object of the subplot.
    is_3d : bool
        A flag indicating whether the subplot is 3D.
    use_blit : bool
        Indicates whether to use blitting for rendering optimization.
    background : np.ndarray or None
        The background image for blitting.
    render_objects : list
        List of objects to render on the axes.
    """

    def __init__(
        self,
        fig: plt.Figure,
        pos: int | tuple,
        is_3d: bool = False,
    ):
        """
        Initializes the PanelComponent.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this panel belongs.
        pos : int or tuple
            The position of the subplot within the figure. Can be an integer for 
            single subplot or a tuple for grid positioning.
        is_3d : bool, optional
            Whether the subplot is 3D, by default False.
        """
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
        """
        Renders the figure using the standard redraw method.
        """
        # self.ax.figure.canvas.draw()
        # self.ax.figure.canvas.flush_events()

    def _render_blit(self) -> None:
        """
        Renders the figure using blitting for performance optimization.
        """
        self.fig.canvas.restore_region(self.background)
        for obj in self.render_objects:
            self.ax.draw_artist(obj)
        self.fig.canvas.blit(self.ax.bbox)
        
    def setup_blit(self, render_objects: list) -> None:
        """
        Sets up blitting by capturing the background and specifying render objects.

        Parameters
        ----------
        render_objects : list
            A list of matplotlib artists that need to be redrawn.
        """
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.render_objects = render_objects

    def render(self) -> None:
        """
        Renders the figure either with blitting or normal render method based on `use_blit`.
        """
        if self.use_blit:
            self._render_blit()
        else:
            self._render_normal()


class View(PanelComponent):
    """
    Abstract base class for a view within the panel, extending PanelComponent.

    This class is intended to be subclassed with specific implementations
    for updating the view with new data.
    """

    def __init__(self, fig: plt.Figure, pos: int, is_3d: bool = False):
        """
        Initializes the View component.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this view belongs.
        pos : int
            The position of the subplot within the figure.
        is_3d : bool, optional
            Whether the subplot is 3D, by default False.
        """
        super().__init__(fig, pos, is_3d)

    @abstractmethod
    def update_view(self, values: np.ndarray) -> None:
        """
        Interface method to update the state of the view.

        Parameters
        ----------
        values : np.ndarray
            An array of new values to update the view with.
        """
        pass


class Plot(PanelComponent):
    """
    Class for creating and managing plots within a panel.

    Attributes
    ----------
    logger : DataLogger
        Logger instance for recording plot data.
    """

    def __init__(
        self,
        fig: plt.Figure,
        pos: int = 111,
        is_3d: bool = False,
        nvars: int = 1,
    ):
        """
        Initializes the Plot component.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this plot belongs.
        pos : int, optional
            The position of the subplot within the figure, by default 111.
        is_3d : bool, optional
            Whether the subplot is 3D, by default False.
        nvars : int, optional
            Number of variables to log and plot, by default 1.
        
        Raises
        ------
        ValueError
            If `nvars` is not an integer greater than 0.
        """
        if not isinstance(nvars, int) or nvars <= 0:
            raise ValueError("nvars must be an integer greater than 0!")
        
        super().__init__(fig, pos, is_3d)

        self.logger = DataLogger(nvars)

    def add_data(self, values: np.ndarray, time: float = None) -> None:
        """
        Adds data to the logger.

        Parameters
        ----------
        values : np.ndarray
            The values to be added to the log.
        time : float, optional
            The timestamp corresponding to the values, by default None.
        """
        self.logger.update(time, values)

    @abstractmethod
    def update_plot(self) -> None:
        """
        Interface method to update the plot view.
        """
        pass

    def update(self, values: np.ndarray, time: float = None) -> None:
        """
        Updates the plot with new data and renders the changes.

        Parameters
        ----------
        values : np.ndarray
            The values to be added to the plot.
        time : float, optional
            The timestamp corresponding to the values, by default None.
        """
        self.add_data(values, time)
        self.update_plot()
