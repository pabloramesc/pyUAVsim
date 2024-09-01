"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from matplotlib import pyplot as plt

from simulator.aircraft import AircraftState, ControlDeltas
from simulator.autopilot import AutopilotStatus
from simulator.gui.attitude_view import AttitudeView
from simulator.gui.horizontal_bar_view import HorizontalBarView
from simulator.gui.panel_components import PanelComponent
from simulator.gui.position_plot import PositionPlot
from simulator.gui.time_series_plot import TimeSeriesPlot


class Panel(ABC):
    """
    Abstract base class for a panel that visualizes aircraft data.

    Attributes
    ----------
    fig : plt.Figure
        The matplotlib figure object.
    use_blit : bool
        Whether to use blitting for faster rendering.
    components : List[PanelComponent]
        A list of panel components to be displayed.
    """

    def __init__(self, figsize=(12, 6), use_blit: bool = False) -> None:
        """
        Initialize the Panel with a figure size and blitting option.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (12, 6).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        """
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.use_blit = use_blit
        self.components: List[PanelComponent] = []

    @abstractmethod
    def add_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Add data to the panel components.

        Parameters
        ----------
        *args : Any
            Positional arguments for data.
        **kwargs : Any
            Keyword arguments for data.
        """
        pass

    @abstractmethod
    def update_plots(self) -> None:
        """
        Update the plots within the panel.
        """
        pass

    @abstractmethod
    def update_views(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the views within the panel.

        Parameters
        ----------
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
        """
        pass

    def add_components(self, components: List[PanelComponent]) -> None:
        """
        Add components to the panel.

        Parameters
        ----------
        components : List[PanelComponent]
            List of components to add.
        """
        for component in components:
            self.components.append(component)
            component.use_blit = self.use_blit

    def update(self, pause: float = 0.01, *args: Any, **kwargs: Any) -> None:
        """
        Update the panel by updating views and plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        *args : Any
            Positional arguments for updates.
        **kwargs : Any
            Keyword arguments for updates.
        """
        self.update_views(*args, **kwargs)
        self.update_plots()
        plt.pause(pause)


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

    def __init__(
        self, figsize=(10, 5), use_blit: bool = False, pos_3d: bool = True
    ) -> None:
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

    def add_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Add data to the attitude and position views.

        Parameters
        ----------
        *args : Any
            Positional arguments for data.
        **kwargs : Any
            Keyword arguments for data.
            - state : AircraftState
                The current state of the aircraft.
        """
        state: AircraftState = kwargs.get("state")
        if state:
            self.position_plot.add_data(state.ned_position)

    def update_plots(self) -> None:
        """
        Update the position plot.
        """
        self.position_plot.update_plot()

    def update_views(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the attitude view.

        Parameters
        ----------
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
            - state : AircraftState
                The current state of the aircraft.
        """
        state: AircraftState = kwargs.get("state")
        if state:
            self.attitude_view.update(state.attitude_angles)

    def update(self, pause: float = 0.01, *args: Any, **kwargs: Any) -> None:
        """
        Update the attitude view and the position plot.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
            - state : AircraftState
                The current state of the aircraft.
        """
        return super().update(pause, *args, **kwargs)


class FlightControlPanel(Panel):
    """
    Panel that displays flight control data using time series plots.

    Attributes
    ----------
    roll_plot : TimeSeriesPlot
        The time series plot for roll.
    beta_plot : TimeSeriesPlot
        The time series plot for beta.
    course_plot : TimeSeriesPlot
        The time series plot for course.
    pitch_plot : TimeSeriesPlot
        The time series plot for pitch.
    airspeed_plot : TimeSeriesPlot
        The time series plot for airspeed.
    altitude_plot : TimeSeriesPlot
        The time series plot for altitude.
    """

    def __init__(self, figsize=(10, 5), use_blit: bool = False) -> None:
        """
        Initialize the FlightControlPanel with time series plots.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (12, 6).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        """
        super().__init__(figsize, use_blit)
        self.roll_plot = TimeSeriesPlot(
            self.fig, pos=231, ylabel="Roll (deg)", nvars=2, labels=["real", "target"]
        )
        self.beta_plot = TimeSeriesPlot(
            self.fig, pos=232, ylabel="Beta (deg)", nvars=2, labels=["real", "target"]
        )
        self.course_plot = TimeSeriesPlot(
            self.fig, pos=233, ylabel="Course (deg)", nvars=2, labels=["real", "target"]
        )
        self.pitch_plot = TimeSeriesPlot(
            self.fig, pos=234, ylabel="Pitch (deg)", nvars=2, labels=["real", "target"]
        )
        self.airspeed_plot = TimeSeriesPlot(
            self.fig,
            pos=235,
            ylabel="Airspeed (m/s)",
            nvars=2,
            labels=["real", "target"],
        )
        self.altitude_plot = TimeSeriesPlot(
            self.fig, pos=236, ylabel="Altitude (m)", nvars=2, labels=["real", "target"]
        )

        self.add_components(
            [
                self.roll_plot,
                self.beta_plot,
                self.course_plot,
                self.pitch_plot,
                self.airspeed_plot,
                self.altitude_plot,
            ]
        )
        self.fig.tight_layout(pad=2)

    def add_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Add autopilot data to the time series plots.

        Parameters
        ----------
        *args : Any
            Positional arguments for data.
        **kwargs : Any
            Keyword arguments for data.
            - time : float
                The current time.
            - autopilot_status : AutopilotState
                The current state of the autopilot.
        """
        time: float = kwargs.get("time")
        autopilot_status: AutopilotStatus = kwargs.get("autopilot_status")
        if autopilot_status and time is not None:
            self.roll_plot.add_data(
                np.rad2deg([autopilot_status.roll, autopilot_status.roll_target]), time
            )
            self.beta_plot.add_data(
                np.rad2deg([autopilot_status.beta, autopilot_status.beta_target]), time
            )
            self.course_plot.add_data(
                np.rad2deg([autopilot_status.course, autopilot_status.course_target]),
                time,
            )
            self.pitch_plot.add_data(
                np.rad2deg([autopilot_status.pitch, autopilot_status.pitch_target]), time
            )
            self.airspeed_plot.add_data(
                np.array([autopilot_status.airspeed, autopilot_status.airspeed_target]),
                time,
            )
            self.altitude_plot.add_data(
                np.array([autopilot_status.altitude, autopilot_status.altitude_target]),
                time,
            )

    def update_plots(self) -> None:
        """
        Update all time series plots.
        """
        self.roll_plot.update_plot()
        self.beta_plot.update_plot()
        self.course_plot.update_plot()
        self.pitch_plot.update_plot()
        self.airspeed_plot.update_plot()
        self.altitude_plot.update_plot()

    def update_views(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the views within the panel.
        This panel does not use additional views beyond plots.

        Parameters
        ----------
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
        """
        pass

    def update(self, pause: float = 0.01, *args: Any, **kwargs: Any) -> None:
        """
        Update the plots within the panel.
        This panel does not use additional views beyond plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
        """
        return super().update(pause, *args, **kwargs)


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

    def __init__(self, figsize=(10, 6), use_blit: bool = False) -> None:
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

    def add_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Add data to the main status views and plots.

        Parameters
        ----------
        *args : Any
            Positional arguments for data.
        **kwargs : Any
            Keyword arguments for data.
            - time : float
                The current time.
            - state : AircraftState
                The current state of the aircraft.
        """
        time: float = kwargs.get("time")
        state: AircraftState = kwargs.get("state")
        if state:
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

    def update_views(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the attitude and control deltas views.

        Parameters
        ----------
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
            - state : AircraftState
                The current state of the aircraft.
            - deltas : ControlDeltas
                The current control surface deflections.
        """
        state: AircraftState = kwargs.get("state")
        deltas: ControlDeltas = kwargs.get("deltas")
        if state:
            self.attitude_view.update(state.attitude_angles)
        if deltas:
            self.delta_a_view.update(deltas.delta_a)
            self.delta_e_view.update(deltas.delta_e)
            self.delta_r_view.update(deltas.delta_r)
            self.delta_t_view.update(deltas.delta_t)

    def update(self, pause: float = 0.01, *args: Any, **kwargs: Any) -> None:
        """
        Update the attitude and control deltas view;
        and the position, altitude, and airspeed plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        *args : Any
            Positional arguments for view updates.
        **kwargs : Any
            Keyword arguments for view updates.
            - state : AircraftState
                The current state of the aircraft.
            - deltas : ControlDeltas
                The current control surface deflections.
        """
        return super().update(pause, *args, **kwargs)