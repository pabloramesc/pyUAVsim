"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Any, override

from simulator.aircraft import AircraftState, ControlDeltas
from simulator.gui.attitude_view import AttitudeView
from simulator.gui.horizontal_bar_view import HorizontalBarView
from simulator.gui.panel_base import Panel
from simulator.gui.position_plot import PositionPlot
from simulator.gui.time_series_plot import TimeSeriesPlot


class MainStatusPanel(Panel):
    """
    Panel that displays the main status of the aircraft including attitude,
    position, control deltas, altitude, and airspeed.

    Attributes
    ----------
    attitude_view : AttitudeView
        The attitude view component.
    position_plot : PositionPlot
        The position plot component.
    delta_a_view : HorizontalBarView
        The horizontal bar view for aileron deflection.
    delta_e_view : HorizontalBarView
        The horizontal bar view for elevator deflection.
    delta_r_view : HorizontalBarView
        The horizontal bar view for rudder deflection.
    delta_t_view : HorizontalBarView
        The horizontal bar view for throttle position.
    altitude_plot : TimeSeriesPlot
        The time series plot for altitude.
    airspeed_plot : TimeSeriesPlot
        The time series plot for airspeed.
    """

    def __init__(self, figsize=(10, 6), use_blit: bool = False, **kwargs) -> None:
        """
        Initialize the MainStatusPanel with various views and plots.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (10, 6).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        """
        super().__init__(figsize, use_blit)
        self.attitude_view = AttitudeView(self.fig, pos=221)
        self.position_plot = PositionPlot(self.fig, pos=222, is_3d=False)
        self.delta_a_view = HorizontalBarView(
            self.fig, (8, 2, 9), "Aileron", (-1.0, +1.0)
        )
        self.delta_e_view = HorizontalBarView(
            self.fig, (8, 2, 11), "Elevator", (-1.0, +1.0)
        )
        self.delta_r_view = HorizontalBarView(
            self.fig, (8, 2, 13), "Rudder", (-1.0, +1.0)
        )
        self.delta_t_view = HorizontalBarView(
            self.fig, (8, 2, 15), "Throttle", (0.0, 1.0)
        )
        self.altitude_plot = TimeSeriesPlot(
            self.fig, pos=426, ylabel="Altitude (m)", title="Altitude & Airspeed Log"
        )
        self.airspeed_plot = TimeSeriesPlot(
            self.fig, pos=428, ylabel="Airspeed (m/s)", xlabel="Time (s)"
        )
        self.add_components(
            [
                self.attitude_view,
                self.position_plot,
                self.delta_a_view,
                self.delta_e_view,
                self.delta_r_view,
                self.delta_t_view,
                self.altitude_plot,
                self.airspeed_plot,
            ]
        )
        self.fig.subplots_adjust(wspace=0.4, hspace=0.4)

    def add_data(self, **kwargs: Any) -> None:
        """
        Add data to the main status views and plots.

        Parameters
        ----------
        time : float
            The current time.
        state : AircraftState
            The current state of the aircraft.

        Raises
        ------
        ValueError
            If the required keyword argument 'time' or 'state' is not provided.
        """
        time: float = kwargs.get("time")
        state: AircraftState = kwargs.get("state")

        if time is None:
            raise ValueError("Missing required keyword argument 'time'")
        if state is None:
            raise ValueError("Missing required keyword argument 'state'")

        self.position_plot.add_data(state.ned_position)
        self.altitude_plot.add_data(state.altitude, time)
        self.airspeed_plot.add_data(state.airspeed, time)

    def update_plots(self) -> None:
        """
        Update the position, altitude, and airspeed plots.
        """
        self.position_plot.update_plot()
        self.altitude_plot.update_plot()
        self.airspeed_plot.update_plot()

    def update_views(self, **kwargs: Any) -> None:
        """
        Update the attitude and control deltas views.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft.
        deltas : ControlDeltas
            The current control surface deflections.

        Raises
        ------
        ValueError
            If the required keyword argument 'state' or 'deltas' is not provided.
        """
        state: AircraftState = kwargs.get("state")
        deltas: ControlDeltas = kwargs.get("deltas")

        if state is None:
            raise ValueError("Missing required keyword argument 'state'")
        if deltas is None:
            raise ValueError("Missing required keyword argument 'deltas'")

        self.attitude_view.update_view(state.attitude_angles)
        self.delta_a_view.update_view(deltas.delta_a)
        self.delta_e_view.update_view(deltas.delta_e)
        self.delta_r_view.update_view(deltas.delta_r)
        self.delta_t_view.update_view(deltas.delta_t)

    @override
    def update(self, pause: float = 0.01, **kwargs: Any) -> None:
        """
        Update the attitude and control deltas view;
        and the position, altitude, and airspeed plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        state : AircraftState
            The current state of the aircraft.
        deltas : ControlDeltas
            The current control surface deflections.

        Raises
        ------
        ValueError
            If the required keyword argument 'state' or 'deltas' is not provided.
        """
        state: AircraftState = kwargs.get("state")
        deltas: ControlDeltas = kwargs.get("deltas")

        if state is None:
            raise ValueError("Missing required keyword argument 'state'")
        if deltas is None:
            raise ValueError("Missing required keyword argument 'deltas'")

        return super().update(pause, state=state, deltas=deltas)
