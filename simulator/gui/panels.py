import numpy as np

from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from simulator.gui.panel_components import PanelComponent
from simulator.gui.attitude_view import AttitudeView
from simulator.gui.position_plot import PositionPlot
from simulator.gui.horizontal_bar_view import HorizontalBarView
from simulator.gui.time_series_plot import TimeSeriesPlot
from simulator.aircraft import AircraftState


class Panel(ABC):
    def __init__(self, figsize=(12, 6), use_blit: bool = False) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.use_blit = use_blit
        self.components: list[PanelComponent] = []

    @abstractmethod
    def add_data(self, state: AircraftState, time: float = None) -> None:
        pass

    @abstractmethod
    def update_plots(self) -> None:
        pass

    @abstractmethod
    def update_views(self, state: AircraftState) -> None:
        pass

    def add_components(self, components: list[PanelComponent]) -> None:
        for component in components:
            self.components.append(component)
            component.use_blit = self.use_blit

    def update(self, state: AircraftState, pause: float = 0.0) -> None:
        self.update_views(state)
        self.update_plots()
        # plt.draw()
        plt.pause(pause)


class AttitudePositionPanel(Panel):
    def __init__(
        self, figsize=(10, 5), use_blit: bool = False, pos_3d: bool = True
    ) -> None:
        super().__init__(figsize, use_blit)
        self.attitude_view = AttitudeView(self.fig, pos=121)
        self.position_plot = PositionPlot(self.fig, pos=122, is_3d=pos_3d)
        self.add_components([self.attitude_view, self.position_plot])
        self.fig.tight_layout(pad=2)

    def add_data(self, state: AircraftState, time: float = None) -> None:
        self.position_plot.add_data(state.ned_position)

    def update_plots(self) -> None:
        self.position_plot.update_plot()

    def update_views(self, state: AircraftState) -> None:
        self.attitude_view.update(state.attitude_angles)


class MainStatusPanel(Panel):
    def __init__(self, figsize=(10, 10), use_blit: bool = False) -> None:
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

    def add_data(self, state: AircraftState, time: float = None) -> None:
        self.position_plot.add_data(state.ned_position)
        self.altitude_plot.add_data(state.altitude, time)
        self.airspeed_plot.add_data(state.airspeed, time)

    def update_plots(self) -> None:
        self.position_plot.update_plot()
        self.altitude_plot.update_plot()
        self.airspeed_plot.update_plot()

    def update_views(self, state: AircraftState) -> None:
        self.attitude_view.update(state.attitude_angles)
        self.delta_a_view.update(state.control_deltas.delta_a)
        self.delta_e_view.update(state.control_deltas.delta_e)
        self.delta_r_view.update(state.control_deltas.delta_r)
        self.delta_t_view.update(state.control_deltas.delta_t)
