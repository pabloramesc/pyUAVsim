"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from matplotlib.figure import Figure
import numpy as np

from simulator.gui.panel_components import View


class HorizontalBarView(View):

    def __init__(
        self,
        fig: Figure,
        pos: int,
        label: str = "Value",
        xlim: tuple[float] = (0.0, 1.0),
    ):
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

    def update(self, value: float) -> None:
        self.value = value
        self.bar.set_width(self.value)
        self.text.set_text(f"{self.value:.2f}")
        self.render()
