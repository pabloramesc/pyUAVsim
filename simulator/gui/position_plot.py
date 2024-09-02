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
    """
    A class to create and manage a plot for visualizing position data 
    in the North-East-Down (NED) coordinate frame.

    Attributes
    ----------
    line : matplotlib.lines.Line2D
        The line object representing the position in the plot.
    """

    def __init__(self, fig: plt.Figure, pos: int = 111, is_3d: bool = False) -> None:
        """
        Initializes the PositionPlot component.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this plot belongs.
        pos : int, optional
            The position of the subplot within the figure, by default 111.
        is_3d : bool, optional
            Whether the subplot is 3D, by default False.
        """
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
        """
        Updates the plot with the latest position data from the logger.

        The method fetches the position data from the logger and updates
        the line object in the plot. It also adjusts the axis limits to fit
        the new data, including adjustments for 3D plots if necessary.
        """
        pos = self.logger.as_dict()
        
        self.line.set_data(pos["pe"], pos["pn"])
        if self.is_3d:
            self.line.set_3d_properties(-pos["pd"])
        
        # Autoscale axis limits
        self.ax.set_xlim(np.min(pos["pe"]), np.max(pos["pe"]))
        self.ax.set_ylim(np.min(pos["pn"]), np.max(pos["pn"]))
        if self.is_3d:
            self.ax.set_zlim(np.min(-pos["pd"]), np.max(-pos["pd"]))
            
        self.render()
