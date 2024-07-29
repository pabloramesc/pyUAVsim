"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.visualization import GeneralPlotter
from simulator.aircraft import AircraftState


class AltitudeTimeLog(GeneralPlotter):
    def __init__(self, ax: plt.Axes = None) -> None:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        super().__init__(ax, is_3d=False)
        self.ax.set_title("Altitude vs Time Logger")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Altitude (m)")
        self.line, = self.ax.plot([], [], color='b', linewidth=2.0)
        self.altitude_log= []

    def update(self, state: AircraftState, time: float):
        self.altitude_log.append([time, state.altitude])
        log = np.array(self.altitude_log)
        self.line.set_data(log[:, 0], log[:, 1])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()