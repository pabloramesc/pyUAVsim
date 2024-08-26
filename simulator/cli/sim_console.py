"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from rich.console import Console
from rich.table import Table

from simulator.aircraft.aircraft_state import AircraftState
from simulator.utils.readable import seconds_to_hhmmss, seconds_to_dhms


class SimConsole:
    def __init__(self) -> None:
        self.console = Console()

    def print_state(self, t_sim: float, t_real: float, state: AircraftState, style="simple") -> None:
        # Clear the screen
        self.console.clear()

        if style == "simple":
            self._print_state_simple(t_sim, t_real, state)
        elif style == "table":
            self._print_state_table(t_sim, t_real, state)
        else:
            raise ValueError(
                "Not valid style parameter! Valid oprtions are 'simple' or 'table'."
            )

    def _print_state_simple(self, t_sim: float, t_real: float, state: AircraftState) -> None:
        t_sim_str = seconds_to_dhms(t_sim)
        t_real_str = seconds_to_dhms(t_real)
        # Print updated content
        self.console.print(f"Sim Time: {t_sim_str}, Real Time: {t_real_str}")
        self.console.print(
            f"NED position (m): pn: {state.pn:.2f}, pe: {state.pe:.2f}, pd: {state.pd:.2f}"
        )
        self.console.print(
            f"Body velocity (m/s): u: {state.u:.2f}, v: {state.v:.2f}, w: {state.w:.2f}"
        )
        self.console.print(
            f"Attitude (degs): roll: {np.rad2deg(state.roll):.2f}, pitch: {np.rad2deg(state.pitch):.2f}, yaw: {np.rad2deg(state.yaw):.2f}"
        )
        self.console.print(
            f"Angular Rates (rads/s): p: {state.p:.2f}, q: {state.q:.2f}, r: {state.r:.2f}"
        )

    def _print_state_table(self, t_sim: float, t_real: float, state: AircraftState) -> None:
        # Create a table
        table = Table()

        # Define columns
        table.add_column("Aircraft State", justify="left", style="cyan", no_wrap=True)
        table.add_column("X Value", style="magenta")
        table.add_column("Y Value", style="green")
        table.add_column("Z Value", style="blue")

        # Add rows to the table
        table.add_row(
            "NED Position (m)",
            f"pn: {state.pn:.2f}",
            f"pe: {state.pe:.2f}",
            f"pd: {state.pd:.2f}",
        )
        table.add_row(
            "Body Velocity (m/s)",
            f"u: {state.u:.2f}",
            f"v: {state.v:.2f}",
            f"w: {state.w:.2f}",
        )
        table.add_row(
            "Attitude (deg)",
            f"roll: {np.rad2deg(state.roll):.2f}",
            f"pitch: {np.rad2deg(state.pitch):.2f}",
            f"yaw: {np.rad2deg(state.yaw):.2f}",
        )
        table.add_row(
            "Angular Rates (rad/s)",
            f"p: {state.p:.2f}",
            f"q: {state.q:.2f}",
            f"r: {state.r:.2f}",
        )

        # Print the table
        self.console.print(f"Sim Time: {t_sim:.2f} s, Real Time: {t_real:.2f} s")
        self.console.print(table)
