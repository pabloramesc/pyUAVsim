"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from matplotlib import pyplot as plt

from simulator.aircraft import AircraftState


class AttitudePositionAnimation:

    def __init__(self) -> None:
        self.fig = plt.figure()
        self.attitude_ax = self.fig.add_subplot(121, projection="3d")
        self.position_ax = self.fig.add_subplot(122, projection="3d")
        # initialize position plot
        self.position_log = ([], [], [])
        self.position_line = self.position_ax.plot([], [], [])[0]

    def update_position_data(self, state: AircraftState) -> None:
        self.position_log[0].append(state.pe)  # px = pe
        self.position_log[1].append(state.pn)  # py = pn
        self.position_log[2].append(-state.pd)  # pz = -pd

    def update_position_plot(self) -> None:
        self.position_line.set_data(self.position_log[0], self.position_log[1])
        self.position_line.set_3d_properties(self.position_log[2])

        self.position_ax.set_xlim(min(self.position_log[0]), max(self.position_log[0]))
        self.position_ax.set_ylim(min(self.position_log[1]), max(self.position_log[1]))
        self.position_ax.set_zlim(min(self.position_log[2]), max(self.position_log[2]))
