"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt

from simulator.aircraft import AircraftState
from simulator.gui.general_plotter import GeneralPlotter


class MultiPlotter:
    def __init__(self, figsize=(12, 12)):
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.plotters = []

    def add_subplot(self, position: tuple, plotter: GeneralPlotter):
        nrow, ncol, index = position
        ax = self.fig.add_subplot(nrow, ncol, index, projection='3d' if plotter.is_3d else None)
        plotter.ax = ax
        self.plotters.append(plotter)
        plt.draw()

    def update(self, state: AircraftState, time: float = None, pause=0.0):
        for plotter in self.plotters:
            plotter.update(state, time)
        plt.pause(pause)
