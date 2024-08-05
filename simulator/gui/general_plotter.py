"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from simulator.aircraft import AircraftState


class GeneralPlotter(ABC):
    def __init__(self, fig: plt.Figure, ax: plt.Axes, pos: int = 111, is_3d: bool = False):
        self.fig = fig
        self.ax = ax
        self.pos = pos
        self.is_3d = is_3d

    @abstractmethod
    def update(self, state: AircraftState, time: float = None):
        """Interface method for updating the plotter

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft
        time : float, optional
            The current time value for the simulation, by default None
        """
        pass
