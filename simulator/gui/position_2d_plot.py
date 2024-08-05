"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.general_plotter import GeneralPlotter
from simulator.aircraft.aircraft_state import AircraftState


class Position2DPlot(GeneralPlotter):
    def __init__(self, fig: plt.Figure, pos: int = 111) -> None:
        ax = fig.add_subplot(pos)
        super().__init__(fig, ax, pos, is_3d=False)

        self.ax.set_title("Position in NED Frame")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.grid()

        (self.line,) = self.ax.plot([], [], color="b", linewidth=2.0)
        self.position_history = []

    def update(self, state: AircraftState, time: float = None) -> None:
        self.position_history.append(state.ned_position[0:2])
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            self.line.set_data(positions[:, 1], positions[:, 0])
        self.ax.relim()
        self.ax.autoscale_view()
