"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Any, List

from matplotlib import pyplot as plt

from simulator.gui.panel_components import PanelComponent


class Panel(ABC):
    """
    Abstract base class for a panel that visualizes aircraft data.

    Attributes
    ----------
    fig : plt.Figure
        The matplotlib figure object.
    use_blit : bool
        Whether to use blitting for faster rendering.
    components : List[PanelComponent]
        A list of panel components to be displayed.
    """

    def __init__(self, figsize=(12, 6), use_blit: bool = False, **kwargs: Any) -> None:
        """
        Initialize the Panel with a figure size and blitting option.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (12, 6).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        """
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.use_blit = use_blit
        self.components: List[PanelComponent] = []

    @abstractmethod
    def add_data(self, **kwargs: Any) -> None:
        """
        Interface method for adding data to the panel components.
        """
        pass

    @abstractmethod
    def update_plots(self) -> None:
        """
        Update the plots within the panel.
        """
        pass

    @abstractmethod
    def update_views(self, **kwargs: Any) -> None:
        """
        Interface method for updating the views within the panel.
        """
        pass

    def add_components(self, components: List[PanelComponent]) -> None:
        """
        Add components to the panel.

        Parameters
        ----------
        components : List[PanelComponent]
            List of components to add.
        """
        for component in components:
            self.components.append(component)
            component.use_blit = self.use_blit

    def update(self, pause: float = 0.01, **kwargs: Any) -> None:
        """
        Update the panel by updating views and plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        **kwargs : Any
            Keyword arguments for updates.
        """
        self.update_views(**kwargs)
        self.update_plots()
        plt.pause(pause)


