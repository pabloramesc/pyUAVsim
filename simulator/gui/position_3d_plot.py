"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.general_plotter import GeneralPlotter
from simulator.aircraft import AircraftState


class Position3DPlot(GeneralPlotter):
    def __init__(self, ax=None, hlim=200.0, vlim=50.0):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        super().__init__(ax, is_3d=True)
        self.ax.set_title("Position in NED Frame")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.set_zlabel("Height (m)")
        self.ax.set_xlim(-hlim, hlim)
        self.ax.set_ylim(-hlim, hlim)
        self.ax.set_zlim(-vlim, vlim)
        self.line, = self.ax.plot([], [], [], color='b', linewidth=2.0)
        self.position_history = []

    def update(self, state: AircraftState, time: float = None):
        self.position_history.append(state.ned_position)
        if len(self.position_history) > 1:
            positions = np.array(self.position_history)
            self.line.set_data(positions[:, 1], positions[:, 0])
            self.line.set_3d_properties(positions[:, 2])
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()
