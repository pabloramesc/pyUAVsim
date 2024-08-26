"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.panel_components import Plot
from simulator.math.rotation import ned2xyz


class PositionPlot(Plot):
    def __init__(self, fig: plt.Figure, pos: int = 111, is_3d: bool = False):
        super().__init__(fig, pos, is_3d, nvars=3)
                
        self.ax.set_title("Position in NED Frame")
        self.ax.grid()
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        if is_3d:
            self.ax.set_zlabel("Height (m)")
            self.line, = self.ax.plot([], [], [], color='b', linewidth=2.0)
        else:
            self.line, = self.ax.plot([], [], color="b", linewidth=2.0)
        self.ax.set_aspect('auto')
        
        self.logger.labels = ["pn", "pe", "pd"]
            
        self.setup_blit([self.line])

    def update_plot(self):
        pos = self.logger.as_dict()
        
        self.line.set_data(pos["pe"], pos["pn"])
        if self.is_3d:
            self.line.set_3d_properties(-pos["pd"])
        
        # Autoscale axis limits
        self.ax.set_xlim(min(pos["pe"]), max(pos["pe"]))
        self.ax.set_ylim(min(pos["pn"]), max(pos["pn"]))
        if self.is_3d:
            self.ax.set_zlim(min(-pos["pd"]), max(-pos["pd"]))
            
        self.render()
