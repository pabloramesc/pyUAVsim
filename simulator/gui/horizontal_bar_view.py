"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from matplotlib.figure import Figure
import numpy as np

from simulator.gui.panel_components import View


class HorizontalBarView(View):
    """
    A class for creating and managing a horizontal bar view to visualize a single value.

    This class extends the View class and provides a horizontal bar representation
    of a value with a customizable label and axis limits.

    Attributes
    ----------
    label : str
        The label for the horizontal bar.
    value : float
        The current value represented by the horizontal bar.
    bar : matplotlib.patches.Rectangle
        The rectangle object representing the horizontal bar.
    text : matplotlib.text.Text
        The text object displaying the value next to the bar.
    """

    def __init__(
        self,
        fig: Figure,
        pos: int,
        label: str = "Value",
        xlim: tuple[float] = (0.0, 1.0),
    ) -> None:
        """
        Initializes the HorizontalBarView component.

        Parameters
        ----------
        fig : Figure
            The matplotlib figure object to which this view belongs.
        pos : int
            The position of the subplot within the figure.
        label : str, optional
            The label for the bar, by default "Value".
        xlim : tuple of float, optional
            The limits for the x-axis, by default (0.0, 1.0).
        """
        super().__init__(fig, pos, False)
        self.ax.set_xlim(xlim)
        self.label = label
        self.value = 0.0
        (self.bar,) = self.ax.barh(self.label, self.value, color="blue")

        x_text_position = self.ax.get_xlim()[1] + 0.1 * np.sum(
            np.abs(self.ax.get_xlim())
        )
        self.text = self.ax.text(
            x_text_position, 0, f"{0.0:.2f}", va="center", ha="right"
        )

        self.setup_blit([self.bar, self.text])

    def update_view(self, value: float) -> None:
        """
        Updates the horizontal bar with a new value.

        The method adjusts the width of the bar and updates the associated text
        to reflect the new value.

        Parameters
        ----------
        value : float
            The new value to update the bar with.
        """
        self.value = value
        self.bar.set_width(self.value)
        self.text.set_text(f"{self.value:.2f}")
        self.render()
