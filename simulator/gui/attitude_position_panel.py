"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Any, override

from simulator.aircraft import AircraftState
from simulator.gui.attitude_view import AttitudeView
from simulator.gui.panel_base import Panel
from simulator.gui.position_plot import PositionPlot


class AttitudePositionPanel(Panel):
    """
    Panel that displays attitude and position data of the aircraft.

    Attributes
    ----------
    attitude_view : AttitudeView
        The attitude view component.
    position_plot : PositionPlot
        The position plot component.
    """

    def __init__(self, figsize=(10, 5), use_blit: bool = False, pos_3d: bool = True) -> None:
        """
        Initialize the AttitudePositionPanel with attitude and position views.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (10, 5).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        pos_3d : bool, optional
            Whether to display the position in 3D, by default True.
        """
        super().__init__(figsize, use_blit)

        self.attitude_view = AttitudeView(self.fig, pos=121)
        self.position_plot = PositionPlot(self.fig, pos=122, is_3d=pos_3d)

        self.add_components([self.attitude_view, self.position_plot])
        self.fig.tight_layout(pad=2)

    def add_data(self, state: AircraftState) -> None:
        """
        Add data to the attitude and position views.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft.

        Raises
        ------
        ValueError
            If the required keyword argument 'state' is not provided.
        """
        self.position_plot.add_data(state.ned_position)

    def update_plots(self) -> None:
        """
        Update the position plot.
        """
        self.position_plot.update_plot()

    def update_views(self, state: AircraftState) -> None:
        """
        Update the attitude view.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft.

        Raises
        ------
        ValueError
            If the required keyword argument 'state' is not provided.
        """
        self.attitude_view.update_view(state.attitude_angles)

    @override
    def update(self, state: AircraftState, pause: float = 0.01) -> None:
        """
        Update the attitude view and the position plot.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft.
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01

        Raises
        ------
        ValueError
            If the required keyword argument 'state' is not provided.
        """
        return super().update(state=state, pause=pause)
