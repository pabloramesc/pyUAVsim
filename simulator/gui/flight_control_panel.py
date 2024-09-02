"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from typing import Any, override

import numpy as np

from simulator.autopilot import AutopilotStatus
from simulator.gui.panel_base import Panel
from simulator.gui.time_series_plot import TimeSeriesPlot


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

    def __init__(self, figsize=(10, 5), use_blit: bool = False, **kwargs: Any) -> None:
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
        time : float
            The current time.
        ap_status : AutopilotStatus
            The current state of the autopilot.

        Raises
        ------
        ValueError
            If the required keyword argument 'time' or 'ap_status' is not provided.
        """
        time: float = kwargs.get("time")
        ap_status: AutopilotStatus = kwargs.get("ap_status")

        if time is None:
            raise ValueError("Missing required keyword argument 'time'")
        if ap_status is None:
            raise ValueError("Missing required keyword argument 'ap_status'")

        self.roll_plot.add_data(
            np.rad2deg([ap_status.roll, ap_status.roll_target]), time
        )
        self.beta_plot.add_data(
            np.rad2deg([ap_status.beta, ap_status.beta_target]), time
        )
        self.course_plot.add_data(
            np.rad2deg([ap_status.course, ap_status.course_target]),
            time,
        )
        self.pitch_plot.add_data(
            np.rad2deg([ap_status.pitch, ap_status.pitch_target]),
            time,
        )
        self.airspeed_plot.add_data(
            np.array([ap_status.airspeed, ap_status.airspeed_target]),
            time,
        )
        self.altitude_plot.add_data(
            np.array([ap_status.altitude, ap_status.altitude_target]),
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

    def update_views(self, **kwargs: Any) -> None:
        """
        Update the views within the panel.
        This panel does not use additional views beyond plots.
        """
        pass

    @override
    def update(self, pause: float = 0.01, **kwargs: Any) -> None:
        """
        Update the plots within the panel.
        This panel does not use additional views beyond plots.

        Parameters
        ----------
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        """
        return super().update(pause)
